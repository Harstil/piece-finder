"""Phone-camera simulation shared by scene frames and box-lid photos.

Why: the engine only ever sees phone pixels. A clean render would let it rely on razor edges and
exact colours that no iPhone frame has. This module reproduces, in order, what happens between
the light and the JPEG: scene lighting, white balance and exposure, lens defocus (varying across
a tilted table), hand motion blur, vignetting, sensor noise, the phone's own denoise + sharpen
(which leaves halos), and finally JPEG (done by the caller when writing).

All parameters are drawn from the caller's random stream and returned as a small dict so the
scene JSON can record them for per-condition evaluation.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .common import fbm, smooth_grid, to_u8, upsample

# Parameter ranges. All guessed from looking at iPhone 1080p video frames of a table under room
# light; the eval harness reports accuracy per range so they can be tightened with real data.
DEFOCUS_SIGMA = (0.0, 1.6)        # px, Gaussian stand-in for the lens blur at the focus point
DEFOCUS_TILT_EXTRA = (0.0, 1.4)   # px of extra blur at the far edge of a tilted table
P_MOTION = 0.25                   # share of frames with hand shake
MOTION_LEN = (2.0, 9.0)           # px
VIGNETTE = (0.0, 0.35)            # darkening at the corners
READ_NOISE = (0.5, 2.8)           # 8-bit levels (after the phone's own denoising)
SHOT_NOISE = (0.01, 0.06)         # sqrt-signal gain (8-bit levels at full white ~= 16x this)
WB_SPREAD = 0.10                  # max per-channel white-balance gain error
EXPOSURE = (0.78, 1.22)


def lighting_map(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Multiplicative illumination over the frame: a lamp hotspot, a gradient, soft variation."""
    yy, xx, hl, wl = smooth_grid(h, w)
    ang = rng.uniform(0, 2 * math.pi)
    grad = ((xx - w / 2) * math.cos(ang) + (yy - h / 2) * math.sin(ang)) / max(h, w)
    lx, ly = rng.uniform(-0.3, 1.3) * w, rng.uniform(-0.3, 1.3) * h
    spot = np.exp(-((xx - lx) ** 2 + (yy - ly) ** 2) / (max(h, w) * rng.uniform(0.5, 1.2)) ** 2)
    soft = fbm(rng, hl, wl, feature_px=max(hl, wl) / 5.0, octaves=2)
    m = 1.0 + rng.uniform(0.0, 0.35) * grad + rng.uniform(0.0, 0.3) * (spot - 0.5) + rng.uniform(0.0, 0.04) * soft
    return upsample(m, h, w)


def phone_shadow(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """The soft shadow a hovering phone and hand cast on the table (multiplicative, <= 1)."""
    _, _, hl, wl = smooth_grid(h, w)
    fx, fy = (wl - 1) / (w - 1), (hl - 1) / (h - 1)
    cx, cy = rng.uniform(-0.2, 1.2) * w * fx, rng.uniform(0.3, 1.3) * h * fy
    ax, ay = rng.uniform(0.25, 0.6) * w * fx, rng.uniform(0.2, 0.5) * h * fy
    m = np.zeros((hl, wl), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(ax)), max(1, int(ay))), rng.uniform(0, 180), 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(hl, wl) * rng.uniform(0.05, 0.12))
    return upsample(1.0 - rng.uniform(0.12, 0.35) * m, h, w)


def apply_camera(rng: np.random.Generator, img: np.ndarray, tilt_dir_deg: float | None = None,
                 tilt_deg: float = 0.0, strength: float = 1.0) -> tuple[np.ndarray, dict]:
    """Float RGB 0..1 (already lit) -> uint8 RGB as the phone would store it before JPEG.

    `tilt_dir_deg` / `tilt_deg` make defocus grow toward the far side of a tilted table.
    `strength` scales blur and noise (box photos taken as stills use < 1, video frames 1).
    """
    h, w = img.shape[:2]
    info: dict = {}
    gains = 1.0 + rng.uniform(-WB_SPREAD, WB_SPREAD, 3)
    gains = gains / gains.mean()
    exposure = rng.uniform(*EXPOSURE)
    img = img * (gains * exposure).astype(np.float32)
    info["exposure"] = round(float(exposure), 3)

    sigma = rng.uniform(*DEFOCUS_SIGMA) * strength
    extra = rng.uniform(*DEFOCUS_TILT_EXTRA) * strength * min(1.0, tilt_deg / 20.0)
    if sigma > 0.3 or extra > 0.3:
        near = cv2.GaussianBlur(img, (0, 0), max(sigma, 0.3))
        if extra > 0.3 and tilt_dir_deg is not None:
            far = cv2.GaussianBlur(img, (0, 0), max(sigma + extra, 0.3))
            yy, xx, _, _ = smooth_grid(h, w)
            a = math.radians(tilt_dir_deg)
            t = ((xx - w / 2) * math.cos(a) + (yy - h / 2) * math.sin(a)) / (0.5 * math.hypot(w, h))
            t = upsample(np.clip(t, 0, 1), h, w)[..., None]
            img = near * (1 - t) + far * t
        else:
            img = near
    info["blurSigma"] = round(float(sigma), 2)

    if rng.random() < P_MOTION * strength:
        length = rng.uniform(*MOTION_LEN)
        ang = rng.uniform(0, 180)
        k = int(math.ceil(length)) | 1
        kern = np.zeros((k, k), np.float32)
        c = (k - 1) / 2
        dx, dy = math.cos(math.radians(ang)) * length / 2, math.sin(math.radians(ang)) * length / 2
        cv2.line(kern, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1, cv2.LINE_AA)
        kern /= kern.sum()
        img = cv2.filter2D(img, -1, kern, borderType=cv2.BORDER_REFLECT)
        info["motionBlurPx"] = round(float(length), 1)

    v = rng.uniform(*VIGNETTE)
    if v > 0.01:
        yy, xx, _, _ = smooth_grid(h, w)
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        img = img * upsample(1.0 - v * r2 ** 1.5, h, w)[..., None]

    # Sensor noise in 8-bit units: signal-dependent shot noise plus read noise, mostly luma.
    img = np.clip(img, 0, 1) * 255.0
    read = rng.uniform(*READ_NOISE) * strength
    shot = rng.uniform(*SHOT_NOISE) * strength
    gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    sd = np.sqrt(read ** 2 + (shot ** 2) * 255.0 * gray)
    luma = rng.standard_normal((h, w), dtype=np.float32)
    chroma = rng.standard_normal((h, w, 3), dtype=np.float32)
    img = img + (sd * 0.85 * luma)[..., None] + sd[..., None] * 0.25 * chroma
    info["noise"] = round(float(read), 2)

    # Phone ISP: light denoise, then sharpening with a halo.
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 0.8))
    amount = rng.uniform(0.0, 0.9)
    if amount > 0.1:
        blur = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 1.6))
        img = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    return to_u8(img / 255.0), info
