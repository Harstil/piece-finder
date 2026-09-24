"""Simulated box reference: what the app gets when the user photographs (or uploads) the box lid.

Why: the engine never sees motif.png. It rectifies a phone photo of the lid, whose picture
differs from the puzzle in ways that matter for matching:
- the box is printed separately (its own colour shift) and the lid adds a coloured frame, a
  logo block and a "500 PIECES" badge that often cover the picture's corners;
- the box art is cropped or extended by up to MISMATCH per side relative to the actual puzzle;
- the photo has perspective (up to MAX_PHOTO_TILT_DEG), lid side walls, a glare streak from a
  glossy lid, blur, white-balance error and JPEG — or it is a clean digital image, or a soft
  video-frame grab.
`referenceCorners` are the TL, TR, BR, BL corners of the *motif area* (the true puzzle picture,
pixel-edge rectangle [-0.5, W-0.5] x [-0.5, H-0.5] of motif.png) mapped into reference.jpg — even
when a corner is hidden under a badge or cropped away by the box art.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .backgrounds import make_background
from .camera import apply_camera, lighting_map
from .common import (apply_h, camera_rotation_homography, choice_weighted, clip_bbox, hsv_to_rgb, poly_bbox,
                     raster_alpha, rot_mat, scale_mat, smooth_grid, to_u8, translate, upsample, warp_prefiltered)
from .render import apply_print, make_print_style

KINDS = ("photo", "digital", "video")
KIND_WEIGHTS = (0.6, 0.2, 0.2)       # guessed: most users photograph the lid
MISMATCH = 0.02                      # box art vs puzzle, per side (from the task spec)
MAX_PHOTO_TILT_DEG = 30.0            # from the task spec
MAX_VIDEO_TILT_DEG = 20.0
PHOTO_LONG_SIDE = (3000, 4000)       # from the task spec (12 MP phone stills are 4032 wide)
VIDEO_LONG_SIDE = 1920
DIGITAL_LONG_SIDE = (1200, 3000)
BOX_HEIGHT = (0.07, 0.14)            # lid wall height / lid width. Measured: 5-6 cm on a 37-50 cm lid.
# Generic lid words only: the synthetic boxes must never imitate a real brand.
LOGO_WORDS = ("JIGSAW", "PUZZLE", "CLASSIC PUZZLE", "PREMIUM JIGSAW", "FAMILY PUZZLE", "ART PUZZLE")


def make_reference(rng: np.random.Generator, motif: np.ndarray, n_pieces: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """(reference uint8 RGB, referenceCorners (4, 2), info dict with kind and JPEG quality)."""
    kind = choice_weighted(rng, KINDS, KIND_WEIGHTS)
    box_style = make_print_style(rng)
    box_style.sat = float(np.clip(box_style.sat * rng.uniform(1.0, 1.1), 0.8, 1.2))  # box inks are punchier
    printed = apply_print(motif, box_style)
    lid, motif_to_lid, art_rect = _lid(rng, printed, n_pieces)
    Hm, Wm = motif.shape[:2]
    motif_corners = np.array([[-0.5, -0.5], [Wm - 0.5, -0.5], [Wm - 0.5, Hm - 0.5], [-0.5, Hm - 0.5]])
    corners_lid = apply_h(motif_to_lid, motif_corners)
    info: dict = {"referenceKind": kind}

    if kind == "digital":
        if rng.random() < 0.5:  # a shop image of the picture alone (with its crop/extension)
            x0, y0, x1, y1 = art_rect
            src = lid[int(round(y0)):int(round(y1)), int(round(x0)):int(round(x1))]
            off = translate(-round(x0), -round(y0))
        else:  # a flat scan / shop image of the whole lid
            src, off = lid, np.eye(3)
        long_side = int(rng.integers(DIGITAL_LONG_SIDE[0], DIGITAL_LONG_SIDE[1] + 1))
        f = long_side / max(src.shape[:2])
        size = (max(8, int(round(src.shape[1] * f))), max(8, int(round(src.shape[0] * f))))
        fx, fy = size[0] / src.shape[1], size[1] / src.shape[0]
        img = cv2.resize(src, size, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
        # cv2.resize maps x -> (x + 0.5) * f - 0.5
        R = np.array([[fx, 0, 0.5 * fx - 0.5], [0, fy, 0.5 * fy - 0.5], [0, 0, 1.0]])
        corners = apply_h(R @ off, corners_lid)
        info["jpegQuality"] = int(rng.integers(88, 97))
        return img, corners, info

    video = kind == "video"
    img, H = _photo(rng, lid, video)
    corners = apply_h(H, corners_lid)
    info["jpegQuality"] = int(rng.integers(70, 89)) if video else int(rng.integers(80, 96))
    return img, corners, info


# ----------------------------------------------------------------------------------------------
# The lid
# ----------------------------------------------------------------------------------------------


def _lid(rng: np.random.Generator, printed: np.ndarray, n_pieces: int) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Lid artwork (uint8 RGB), the motif -> lid homography, and the visible art rect (x0, y0, x1, y1)."""
    Hm, Wm = printed.shape[:2]
    # Per-side mismatch: > 0 crops the art (the puzzle extends beyond the box picture),
    # < 0 extends it (the box shows a little more than the puzzle).
    dl, dt, dr, db = rng.uniform(-MISMATCH, MISMATCH, 4)
    vis = np.array([-0.5 + dl * Wm, -0.5 + dt * Hm, Wm - 0.5 - dr * Wm, Hm - 0.5 - db * Hm])
    art_long = int(min(3600, max(Wm, Hm)))
    s = art_long / max(vis[2] - vis[0], vis[3] - vis[1])
    aw, ah = (vis[2] - vis[0]) * s, (vis[3] - vis[1]) * s
    m = art_long * rng.uniform(0.03, 0.09)
    band = art_long * rng.uniform(0.06, 0.16) if rng.random() < 0.55 else m
    band_top = rng.random() < 0.3
    top = band if band_top else m
    bottom = m if band_top else band
    W = int(math.ceil(aw + 2 * m))
    H = int(math.ceil(ah + top + bottom))
    ax0, ay0 = m, top
    motif_to_lid = translate(ax0, ay0) @ scale_mat(s) @ translate(-vis[0], -vis[1])

    brand = hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.35, 0.95), rng.uniform(0.25, 0.9))
    if rng.random() < 0.25:
        brand = np.array([0.96, 0.96, 0.95], np.float32) if rng.random() < 0.6 else np.array([0.08, 0.08, 0.1], np.float32)
    lid = np.empty((H, W, 3), np.uint8)
    lid[:] = to_u8(brand)
    art = warp_prefiltered(printed, motif_to_lid, (W, H), border=cv2.BORDER_REFLECT_101)
    # motif_to_lid maps the art's pixel-edge rectangle onto exactly this (continuous) rectangle.
    rect = np.array([[ax0, ay0], [ax0 + aw, ay0], [ax0 + aw, ay0 + ah], [ax0, ay0 + ah]])
    a = raster_alpha([rect], 0, 0, W, H, ss=2)[..., None]
    lid = (lid * (1 - a) + art * a + 0.5).astype(np.uint8)
    if rng.random() < 0.4:  # thin frame line around the picture
        c = (245, 245, 240) if rng.random() < 0.7 else (200, 170, 90)
        t = max(2, int(art_long * 0.003))
        cv2.rectangle(lid, (int(ax0) - t, int(ay0) - t), (int(ax0 + aw) + t, int(ay0 + ah) + t), c, t, cv2.LINE_AA)
    ink = (250, 250, 250) if float(np.mean(brand)) < 0.55 else (20, 20, 25)
    _logo(rng, lid, ax0, ay0, aw, ah, art_long, ink)
    _badge(rng, lid, ax0, ay0, aw, ah, art_long, n_pieces)
    _title(rng, lid, ax0, ay0 + ah if not band_top else 0, aw, bottom if not band_top else top, art_long, ink)
    return lid, motif_to_lid, (ax0, ay0, ax0 + aw, ay0 + ah)


def _logo(rng, lid, ax0, ay0, aw, ah, L, ink) -> None:
    """A logo-like block at a top corner, overlapping the picture's corner half of the time."""
    word = LOGO_WORDS[int(rng.integers(len(LOGO_WORDS)))]
    scale = L / 900 * rng.uniform(0.8, 1.4)
    thick = max(2, int(scale * 2.2))
    (tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
    pad = th * 0.6
    bw, bh = tw + 2 * pad, th + 2 * pad
    over = rng.random() < 0.5    # True: the block sits on the picture's corner
    left = rng.random() < 0.5
    x = ax0 if left else ax0 + aw - bw
    y = ay0 if over else max(0.0, ay0 - bh - L * 0.004)
    col = to_u8(hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.5, 1.0), rng.uniform(0.4, 0.9)))
    cv2.rectangle(lid, (int(x), int(y)), (int(x + bw), int(y + bh)), tuple(int(v) for v in col), -1, cv2.LINE_AA)
    txt = (255, 255, 255) if float(np.mean(col)) < 140 else (15, 15, 20)
    cv2.putText(lid, word, (int(x + pad), int(y + pad + th)), cv2.FONT_HERSHEY_DUPLEX, scale, txt, thick, cv2.LINE_AA)


def _badge(rng, lid, ax0, ay0, aw, ah, L, n_pieces) -> None:
    """The "500 PIECES" badge, at a bottom corner, often covering the picture corner."""
    r = L * rng.uniform(0.04, 0.07)
    corner = int(rng.integers(4))
    over = rng.random() < 0.55
    inset = r * (0.6 if over else -1.2)
    cx = ax0 + inset if corner in (0, 3) else ax0 + aw - inset
    cy = ay0 + inset if corner in (0, 1) else ay0 + ah - inset
    cx = float(np.clip(cx, r * 1.3, lid.shape[1] - r * 1.3))  # stay on the lid
    cy = float(np.clip(cy, r * 1.3, lid.shape[0] - r * 1.3))
    col = tuple(int(v) for v in to_u8(hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.6, 1.0), rng.uniform(0.6, 0.95))))
    if rng.random() < 0.6:
        cv2.circle(lid, (int(cx), int(cy)), int(r), col, -1, cv2.LINE_AA)
        cv2.circle(lid, (int(cx), int(cy)), int(r), (255, 255, 255), max(2, int(r * 0.06)), cv2.LINE_AA)
    else:
        cv2.rectangle(lid, (int(cx - r * 1.2), int(cy - r * 0.8)), (int(cx + r * 1.2), int(cy + r * 0.8)), col, -1, cv2.LINE_AA)
    num = str(n_pieces)
    font = cv2.FONT_HERSHEY_DUPLEX
    (w1, h1), _ = cv2.getTextSize(num, font, 1.0, 2)
    s1 = min(1.3 * r / w1, 0.7 * r / h1)          # the number fills the badge but stays inside it
    t1 = max(2, int(round(s1 * 2)))
    (tw, th), _ = cv2.getTextSize(num, font, s1, t1)
    txt = (255, 255, 255) if float(np.mean(col)) < 150 else (20, 20, 20)
    cv2.putText(lid, num, (int(cx - tw / 2), int(cy + th * 0.3)), font, s1, txt, t1, cv2.LINE_AA)
    (w2, h2), _ = cv2.getTextSize("PIECES", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
    s2 = 0.95 * r / w2
    t2 = max(1, int(round(s2 * 2)))
    (tw2, th2), _ = cv2.getTextSize("PIECES", cv2.FONT_HERSHEY_SIMPLEX, s2, t2)
    cv2.putText(lid, "PIECES", (int(cx - tw2 / 2), int(cy + th * 0.3 + th2 * 1.9)), cv2.FONT_HERSHEY_SIMPLEX, s2, txt,
                t2, cv2.LINE_AA)


def _title(rng, lid, x0, y0, aw, band_h, L, ink) -> None:
    """A made-up picture title in the lid band (text-like marks only; never a real product name)."""
    if band_h < L * 0.05:
        return
    letters = "ABCDEFGHIJKLMNOPRSTUVW"
    words = [("".join(letters[int(i)] for i in rng.integers(0, len(letters), int(rng.integers(3, 8))))).capitalize()
             for _ in range(int(rng.integers(1, 4)))]
    scale = band_h / 70.0
    thick = max(1, int(scale * 1.6))
    cv2.putText(lid, " ".join(words), (int(x0 + aw * 0.05), int(y0 + band_h * 0.65)), cv2.FONT_HERSHEY_TRIPLEX,
                scale, ink, thick, cv2.LINE_AA)


# ----------------------------------------------------------------------------------------------
# The photo
# ----------------------------------------------------------------------------------------------


def _blend_poly(canvas: np.ndarray, poly: np.ndarray, colour: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Paint `colour` (a colour or a full-size image) inside `poly`, anti-aliased, touching only its bbox.

    Returns (alpha, bbox) so callers can reuse the coverage.
    """
    h, w = canvas.shape[:2]
    x0, y0, x1, y1 = clip_bbox(poly_bbox(poly, 1.0), w, h)
    a = raster_alpha([poly], x0, y0, x1 - x0, y1 - y0, ss=2)
    src = colour[y0:y1, x0:x1] if colour.shape[0] == h and colour.shape[1] == w else colour
    region = canvas[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1] = region * (1 - a[..., None]) + src * a[..., None]
    return a, (x0, y0, x1, y1)


def _photo(rng: np.random.Generator, lid: np.ndarray, video: bool) -> tuple[np.ndarray, np.ndarray]:
    """Photograph the lid on a table. Returns (uint8 RGB photo, lid -> photo homography)."""
    lh, lw = lid.shape[:2]
    landscape = lw >= lh
    if video:
        pw, ph = (VIDEO_LONG_SIDE, 1080) if landscape else (1080, VIDEO_LONG_SIDE)
    else:
        long_side = int(rng.integers(PHOTO_LONG_SIDE[0], PHOTO_LONG_SIDE[1] + 1))
        pw, ph = (long_side, round(long_side * 0.75)) if landscape else (round(long_side * 0.75), long_side)
    f = max(pw, ph) * rng.uniform(0.72, 0.85)
    tilt = float(min(MAX_VIDEO_TILT_DEG if video else MAX_PHOTO_TILT_DEG, abs(rng.normal(0, 12))))
    tilt_dir = rng.uniform(0, 360)
    # The lid lies where the tilted camera looks (plane = straight-down view from the camera's
    # position), so an angled photo sees the lid from the side, with its near wall visible.
    Ht = camera_rotation_homography(pw, ph, f, tilt, tilt_dir)
    look = apply_h(np.linalg.inv(Ht), np.array([[(pw - 1) / 2, (ph - 1) / 2]]))[0]
    fill = rng.uniform(0.7, 0.9)
    s = min(pw * fill / lw, ph * fill / lh)
    A = translate(look[0] + rng.uniform(-0.04, 0.04) * pw, look[1] + rng.uniform(-0.04, 0.04) * ph) @ \
        rot_mat(rng.uniform(-7, 7)) @ scale_mat(s) @ translate(-lw / 2, -lh / 2)
    lid_quad = np.array([[-0.5, -0.5], [lw - 0.5, -0.5], [lw - 0.5, lh - 0.5], [-0.5, lh - 0.5]])
    O = np.array([(pw - 1) / 2, (ph - 1) / 2])
    D = f
    box_h = BOX_HEIGHT[0] + (BOX_HEIGHT[1] - BOX_HEIGHT[0]) * rng.random()
    t = box_h * lw * s                       # wall height in plane px
    top_plane = apply_h(A, lid_quad)
    foot_plane = O + (top_plane - O) * ((D - t) / D)
    q = apply_h(Ht, np.concatenate([top_plane, foot_plane]))
    margin = rng.uniform(0.03, 0.08)
    lo, hi = q.min(axis=0), q.max(axis=0)
    k = min(pw * (1 - 2 * margin) / (hi[0] - lo[0]), ph * (1 - 2 * margin) / (hi[1] - lo[1])) * rng.uniform(0.82, 1.0)
    fit = translate(pw / 2, ph / 2) @ scale_mat(k) @ translate(-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2)
    Hp = fit @ Ht                              # plane -> photo
    top_f = apply_h(Hp, top_plane)
    foot_f = apply_h(Hp, foot_plane)

    bg, _ = make_background(rng, ph // 2, pw // 2)
    canvas = cv2.resize(bg, (pw, ph), interpolation=cv2.INTER_LINEAR)
    # Soft shadow of the box on the table.
    light = rng.uniform(0, 2 * math.pi)
    off = np.array([math.cos(light), math.sin(light)]) * t * k * rng.uniform(0.2, 0.8)
    # Soft box shadow, computed at 1/8 size (it is blurred by tens of px anyway).
    _, _, hl, wl = smooth_grid(ph, pw)
    sx, sy = (wl - 1) / (pw - 1), (hl - 1) / (ph - 1)
    small = np.array([sx, sy])
    sh = raster_alpha([(foot_f + off) * small, foot_f * small], 0, 0, wl, hl, ss=2)
    sh = cv2.GaussianBlur(sh, (0, 0), max(0.5, t * k * sx * rng.uniform(0.2, 0.5)))
    canvas = canvas * (1 - rng.uniform(0.25, 0.5) * upsample(sh, ph, pw))[..., None]
    # Lid side walls: the quads between the top edge and the footprint edge.
    wall_base = lid[lh // 2, 2].astype(np.float32) / 255.0
    for i in range(4):
        j = (i + 1) % 4
        quad = np.array([top_f[i], top_f[j], foot_f[j], foot_f[i]])
        _blend_poly(canvas, quad, (wall_base * rng.uniform(0.55, 0.85))[None, None, :])
    lid_img = warp_prefiltered(lid, Hp @ A, (pw, ph), border=cv2.BORDER_CONSTANT).astype(np.float32) / 255.0
    a_top = _blend_poly(canvas, top_f, lid_img)
    canvas *= lighting_map(rng, ph, pw)[..., None]
    if rng.random() < 0.6:  # glare streak on the glossy lid
        yy, xx, _, _ = smooth_grid(ph, pw)
        c = top_f.mean(axis=0) + rng.uniform(-0.3, 0.3, 2) * (hi - lo)
        ang = rng.uniform(0, math.pi)
        u = (xx - c[0]) * math.cos(ang) + (yy - c[1]) * math.sin(ang)
        v = -(xx - c[0]) * math.sin(ang) + (yy - c[1]) * math.cos(ang)
        g = np.exp(-(u / (max(pw, ph) * rng.uniform(0.2, 0.5))) ** 2 - (v / (max(pw, ph) * rng.uniform(0.01, 0.05))) ** 2)
        x0, y0, x1, y1 = a_top[1]
        g = upsample(g * rng.uniform(0.3, 0.85), ph, pw)[y0:y1, x0:x1, None] * a_top[0][..., None]
        region = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = region + (1 - region) * g
    img, _ = apply_camera(rng, np.clip(canvas, 0, 1), tilt_dir_deg=tilt_dir, tilt_deg=tilt, strength=1.0 if video else 0.6)
    return img, Hp @ A
