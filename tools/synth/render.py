"""Renders one jigsaw piece into a camera frame so it looks like printed cardboard.

Why each effect exists (all of them change what a segmenter or matcher sees):
- print colour shift: a puzzle print never matches the box art or the screen: limited black and
  white levels, saturation, gamma and ink balance differ (per puzzle, tiny per-piece drift);
- paper micro-noise and optional linen emboss (the "structure" finish of many brands), so flat
  sky pieces are not mathematically flat;
- a thin light cut-edge rim: the die crushes the print layer along the outline;
- cardboard thickness: the side wall is visible on the side facing the camera because the top
  face is nearer to the lens (parallax), and more so on a tilted table;
- a soft drop shadow plus contact darkening, so pieces sit on the table;
- gloss glare (added by scene.py over all piece tops) and face-down pieces (plain back).

Geometry contract: the *top face* (printed surface) is exactly the ground-truth outline; the
instance mask also covers the visible side wall, since that is what a segmenter must cut out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .common import fbm, raster_alpha, smooth_grid, translate, upsample, warp_prefiltered

# Cardboard thickness as a fraction of the core side. Measured on real pieces: ~2 mm board for
# 17-22 mm cores (Ravensburger, Clementoni, Educa) -> 0.09-0.12.
THICKNESS = (0.085, 0.12)
# Light cut-edge rim width, as a fraction of the core side, clamped to 1-3 px at typical sizes.
# Guessed from close-up photos (~0.3 mm on a 20 mm piece).
RIM_FRACTION = 0.013
RIM_PX = (0.9, 3.0)


@dataclass
class PrintStyle:
    """How one physical puzzle was printed and cut. Fixed per puzzle."""
    black: float            # darkest printable level (0..1)
    white: float            # brightest printable level
    sat: float              # saturation factor
    gamma: float
    gains: np.ndarray       # per-channel ink balance
    back: np.ndarray        # face-down cardboard colour
    wall: np.ndarray        # cut-edge (side wall) colour
    rim: np.ndarray         # crushed-print rim colour
    rim_strength: float
    gloss: float            # 0 = matte, 1 = glossy (scales glare)
    linen: float            # linen emboss amplitude (0 = smooth print)
    paper_noise: float      # micro-noise amplitude
    thickness: float        # board thickness / core side


def make_print_style(rng: np.random.Generator) -> PrintStyle:
    backs = [(0.62, 0.64, 0.68), (0.55, 0.6, 0.7), (0.7, 0.68, 0.62), (0.75, 0.74, 0.72), (0.45, 0.5, 0.6),
             (0.8, 0.76, 0.68), (0.58, 0.58, 0.58)]
    back = np.array(backs[int(rng.integers(len(backs)))], np.float32) * rng.uniform(0.92, 1.06)
    wall = np.clip(back * rng.uniform(0.55, 0.8) + rng.uniform(-0.04, 0.04, 3), 0, 1).astype(np.float32)
    return PrintStyle(
        black=rng.uniform(0.03, 0.09), white=rng.uniform(0.9, 0.98), sat=rng.uniform(0.85, 1.08),
        gamma=rng.uniform(0.9, 1.12), gains=(1 + rng.uniform(-0.035, 0.035, 3)).astype(np.float32),
        back=back, wall=wall, rim=np.array([0.93, 0.91, 0.87], np.float32) * rng.uniform(0.9, 1.02),
        rim_strength=rng.uniform(0.25, 0.6), gloss=rng.uniform(0.2, 1.0),
        linen=rng.uniform(0.015, 0.04) if rng.random() < 0.45 else 0.0,
        paper_noise=rng.uniform(0.008, 0.025), thickness=rng.uniform(*THICKNESS),
    )


def apply_print(motif_u8: np.ndarray, style: PrintStyle) -> np.ndarray:
    """Motif as it came out of the printer (uint8 RGB): levels, saturation, gamma, ink balance."""
    img = motif_u8.astype(np.float32) / 255.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.addWeighted(img, style.sat, cv2.merge([gray, gray, gray]), 1.0 - style.sat, 0.0)
    img = np.clip(img, 0, 1) ** style.gamma
    img = style.black + (style.white - style.black) * img
    img = img * style.gains
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def warp_surface(printed: np.ndarray, M_frame_from_motif: np.ndarray, motif_poly: np.ndarray,
                 bbox: tuple[int, int, int, int]) -> np.ndarray:
    """The printed surface under `motif_poly`, warped into frame bbox (x0, y0, x1, y1): float RGB.

    Only a crop around the piece is warped; the registration to the ground-truth outline is exact
    (see common.warp_prefiltered).
    """
    x0, y0, x1, y1 = bbox
    mh, mw = printed.shape[:2]
    cx0 = max(0, int(math.floor(motif_poly[:, 0].min())) - 3)
    cy0 = max(0, int(math.floor(motif_poly[:, 1].min())) - 3)
    cx1 = min(mw, int(math.ceil(motif_poly[:, 0].max())) + 4)
    cy1 = min(mh, int(math.ceil(motif_poly[:, 1].max())) + 4)
    T = translate(-x0, -y0) @ M_frame_from_motif @ translate(cx0, cy0)
    out = warp_prefiltered(printed[cy0:cy1, cx0:cx1], T, (x1 - x0, y1 - y0))
    return out.astype(np.float32) / 255.0


def surface_finish(rng: np.random.Generator, rgb: np.ndarray, alpha_top: np.ndarray, style: PrintStyle,
                   core_px: float, angle_deg: float, face_up: bool) -> np.ndarray:
    """Paper noise, linen emboss and the light cut-edge rim on a piece's top face (float RGB)."""
    h, w = alpha_top.shape
    if not face_up:
        fib = fbm(rng, h, w, feature_px=rng.uniform(1.5, 3.0), octaves=3, aspect=rng.uniform(1, 4))
        low = fbm(rng, h, w, feature_px=max(4.0, core_px / 2), octaves=2)
        shade = 1.0 + 0.03 * fib + 0.025 * low
        rgb = style.back[None, None, :] * shade[..., None]
    rgb = rgb * float(rng.uniform(0.985, 1.015))
    noise = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.6) * 2.2
    rgb = rgb * (1.0 + style.paper_noise * noise)[..., None]
    if style.linen > 0 and face_up:
        ys = np.arange(h, dtype=np.float32)[:, None]
        xs = np.arange(w, dtype=np.float32)[None, :]
        a = math.radians(angle_deg)
        period = max(2.2, core_px / 40.0)  # ~0.5 mm emboss pitch on a 20 mm piece
        u = (xs * math.cos(a) + ys * math.sin(a)) * (2 * math.pi / period)
        v = (-xs * math.sin(a) + ys * math.cos(a)) * (2 * math.pi / period)
        rgb = rgb * (1.0 + style.linen * np.sin(u) * np.sin(v))[..., None]
    rim_w = float(np.clip(core_px * RIM_FRACTION, *RIM_PX))
    inside = (alpha_top >= 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(inside, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    rim = np.clip(1.0 - (dist - 0.5) / rim_w, 0.0, 1.0) * inside
    rim_col = style.rim if face_up else np.clip(style.back * 1.12, 0, 1)
    k = (style.rim_strength * rim)[..., None]
    return rgb * (1 - k) + rim_col[None, None, :] * k


def wall_colour(rng: np.random.Generator, style: PrintStyle, facing_light: float) -> np.ndarray:
    """Side-wall colour; walls facing the light are brighter."""
    return np.clip(style.wall * (0.75 + 0.35 * facing_light) * rng.uniform(0.92, 1.08), 0, 1)


def sweep_alpha(alpha: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """Union of `alpha` translated by t*disp for t in (0, 1]: the swept side wall."""
    n = int(min(8, math.ceil(float(np.hypot(*disp)) / 1.0)))
    out = np.zeros_like(alpha)
    h, w = alpha.shape
    for i in range(1, n + 1):
        t = i / n
        M = np.array([[1, 0, disp[0] * t], [0, 1, disp[1] * t]], np.float32)
        out = np.maximum(out, cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0))
    return out


def glare_field(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Specular reflection of a lamp or window: an elongated highlight somewhere in the frame."""
    yy, xx, _, _ = smooth_grid(h, w)
    cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
    a = rng.uniform(0, math.pi)
    u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
    v = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
    L = max(h, w) * rng.uniform(0.15, 0.5)
    S = max(h, w) * rng.uniform(0.02, 0.08)
    g = np.exp(-(u / L) ** 2 - (v / S) ** 2)
    if rng.random() < 0.4:  # window: a second parallel bar
        off = S * rng.uniform(2.5, 4.0)
        g = g + 0.8 * np.exp(-(u / L) ** 2 - ((v - off) / S) ** 2)
    return upsample(g * rng.uniform(0.35, 0.9), h, w)


def contact_and_drop_shadow(foot_poly: np.ndarray, shadow_poly: np.ndarray, bbox: tuple[int, int, int, int],
                            blur_px: float) -> tuple[np.ndarray, np.ndarray]:
    """(drop shadow alpha, contact darkening alpha) over the bbox, both float 0..1."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    drop = raster_alpha([shadow_poly, foot_poly], x0, y0, w, h, ss=2)
    drop = cv2.GaussianBlur(drop, (0, 0), max(0.6, blur_px))
    foot = raster_alpha([foot_poly], x0, y0, w, h, ss=2)
    contact = cv2.GaussianBlur(foot, (0, 0), 1.2)
    return drop, contact
