"""Shared building blocks for the synthetic data generator.

Why one module: every stage (cut, motifs, render, scene, reference, preview) needs the same few
things, and they must agree exactly or the ground truth drifts from the pixels:
- deterministic random streams keyed by name (no global random state, so a dataset is a pure
  function of its seed and arguments, independent of generation order or worker count);
- the pixel-centre coordinate convention of src/engine/types.ts (pixel (i, j) is centred on the
  integer point (i, j) and covers [i-0.5, i+0.5]) applied identically when rasterising polygons;
- homography maths for the camera and box-photo simulations;
- noise fields (fBm) used by motifs, backgrounds and piece surfaces;
- byte-stable image and JSON writers.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

# ----------------------------------------------------------------------------------------------
# Random streams
# ----------------------------------------------------------------------------------------------


def _key_to_int(key: object) -> int:
    if isinstance(key, str):
        return zlib.crc32(key.encode("utf-8"))
    return int(key) & 0xFFFFFFFF


def rng_for(seed: int, *keys: object) -> np.random.Generator:
    """An independent random stream for (seed, *keys).

    Each puzzle / scene / sub-step draws from its own stream, so changing how many scenes are
    generated, or generating them in parallel, never changes any other item's output.
    """
    entropy = [int(seed) & 0xFFFFFFFF] + [_key_to_int(k) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def choice_weighted(rng: np.random.Generator, options: Sequence[str], weights: Sequence[float]) -> str:
    w = np.asarray(weights, dtype=np.float64)
    return options[int(rng.choice(len(options), p=w / w.sum()))]


# ----------------------------------------------------------------------------------------------
# Polygons
# ----------------------------------------------------------------------------------------------


def signed_area(poly: np.ndarray) -> float:
    """Shoelace area. Positive means clockwise as seen on screen (y axis points down)."""
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


# Supersampling factor for anti-aliased rasterisation. 4x4 sub-samples give coverage in steps of
# 1/16, which is finer than 8-bit JPEG can show at an edge. Guessed, and good enough.
AA_SUPERSAMPLE = 4
_FIXED_SHIFT = 4  # cv2.fillPoly sub-pixel bits


def raster_alpha(polys: Iterable[np.ndarray], x0: int, y0: int, w: int, h: int,
                 ss: int = AA_SUPERSAMPLE) -> np.ndarray:
    """Anti-aliased coverage (float32 0..1, shape h x w) of the union of `polys`.

    Pixel (i, j) of the result is the image pixel centred at (x0 + i, y0 + j), so polygons are
    given in the same absolute pixel-centre coordinates as the dataset's ground truth.
    """
    if w <= 0 or h <= 0:
        return np.zeros((max(h, 0), max(w, 0)), np.float32)
    canvas = np.zeros((h * ss, w * ss), np.uint8)
    scale = float(1 << _FIXED_SHIFT)
    pts = []
    for p in polys:
        q = (np.asarray(p, np.float64) - (x0, y0) + 0.5) * ss - 0.5
        pts.append(np.round(q * scale).astype(np.int32).reshape(-1, 1, 2))
    if pts:
        cv2.fillPoly(canvas, pts, 255, lineType=cv2.LINE_8, shift=_FIXED_SHIFT)
    if ss == 1:
        return canvas.astype(np.float32) / 255.0
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def raster_binary(polys: Iterable[np.ndarray], x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """Pixel-centre sampled (non anti-aliased) fill, uint8 0/1."""
    canvas = np.zeros((h, w), np.uint8)
    scale = float(1 << _FIXED_SHIFT)
    pts = [np.round((np.asarray(p, np.float64) - (x0, y0)) * scale).astype(np.int32).reshape(-1, 1, 2)
           for p in polys]
    if pts:
        cv2.fillPoly(canvas, pts, 1, lineType=cv2.LINE_8, shift=_FIXED_SHIFT)
    return canvas


def poly_bbox(poly: np.ndarray, pad: float = 0.0) -> tuple[int, int, int, int]:
    """Integer pixel bbox (x0, y0, x1, y1), x1/y1 exclusive, covering the polygon plus `pad`."""
    x0 = int(math.floor(poly[:, 0].min() - pad))
    y0 = int(math.floor(poly[:, 1].min() - pad))
    x1 = int(math.ceil(poly[:, 0].max() + pad)) + 1
    y1 = int(math.ceil(poly[:, 1].max() + pad)) + 1
    return x0, y0, x1, y1


def clip_bbox(b: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    return max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h)


# ----------------------------------------------------------------------------------------------
# Homographies
# ----------------------------------------------------------------------------------------------


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def scale_mat(sx: float, sy: float | None = None) -> np.ndarray:
    return np.diag([sx, sx if sy is None else sy, 1.0])


def rot_mat(deg: float) -> np.ndarray:
    """Rotation by `deg` clockwise as seen on screen (y down)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def camera_rotation_homography(w: int, h: int, f: float, tilt_deg: float, tilt_dir_deg: float,
                               roll_deg: float = 0.0) -> np.ndarray:
    """H = K R K^-1: frontal (straight-down) view -> view of a camera rotated about its centre.

    Plane coordinates are the pixels the same camera would see looking straight down, with the
    principal point at the image centre. The table plane stays at the same place; only the camera
    turns, which is exactly how a phone held at an angle sees a flat table.
    """
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    a = math.radians(tilt_dir_deg)
    axis = np.array([math.cos(a), math.sin(a), 0.0])
    R_tilt, _ = cv2.Rodrigues(axis * math.radians(tilt_deg))
    R_roll, _ = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(roll_deg)]))
    R = R_roll @ R_tilt
    return K @ R @ np.linalg.inv(K)


def warp_prefiltered(src: np.ndarray, M: np.ndarray, dsize: tuple[int, int],
                     border: int = cv2.BORDER_REFLECT) -> np.ndarray:
    """cv2.warpPerspective(src, M, dsize) that pre-filters with INTER_AREA when M shrinks a lot.

    The exact pixel-centre mapping of cv2.resize is folded into M, so the result stays registered
    to geometry computed with M to a small fraction of a pixel.
    """
    J = M[:2, :2] / M[2, 2]
    scale = math.sqrt(abs(float(np.linalg.det(J))))
    if scale < 0.7:
        f = min(1.0, scale * 1.5)
        size = (max(2, int(round(src.shape[1] * f))), max(2, int(round(src.shape[0] * f))))
        fx, fy = size[0] / src.shape[1], size[1] / src.shape[0]
        src = cv2.resize(src, size, interpolation=cv2.INTER_AREA)
        M = M @ np.array([[1 / fx, 0, 0.5 / fx - 0.5], [0, 1 / fy, 0.5 / fy - 0.5], [0, 0, 1.0]])
    return cv2.warpPerspective(src, M, dsize, flags=cv2.INTER_LINEAR, borderMode=border)


def local_scale(H: np.ndarray, p: np.ndarray) -> float:
    """sqrt(|det J|) of the homography at plane point p: how many output px one input px becomes."""
    eps = 0.5
    q = apply_h(H, np.array([p, p + (eps, 0), p + (0, eps)]))
    J = np.stack([(q[1] - q[0]) / eps, (q[2] - q[0]) / eps], axis=1)
    return math.sqrt(abs(float(np.linalg.det(J))))


def up_angle_deg(motif_to_frame: np.ndarray, motif_corners: np.ndarray) -> float:
    """Direction of a piece's motif-up vector in the image, clockwise from image-up, in [0, 360).

    types.ts defines upAngleDeg as the frame direction of the motif-up vector (motif -y). Under
    perspective that direction varies across the piece, so it is taken at the core centre (mean of
    the 4 motif corners), from the Jacobian of the motif -> frame homography there. An earlier version
    used "bottom-edge midpoint -> top-edge midpoint" of the frame corners; that is only the up vector
    for rectangular cores and was off by up to 8.7 degrees on irregular cuts (eval/check-dataset.ts).
    """
    M = np.asarray(motif_to_frame, np.float64)
    c = np.asarray(motif_corners, np.float64).mean(axis=0)
    X, Y, Wh = M @ np.array([c[0], c[1], 1.0])
    # d(frame point)/d(motif y); motif-up is the negative of it.
    ux = -(M[0, 1] * Wh - X * M[2, 1]) / (Wh * Wh)
    uy = -(M[1, 1] * Wh - Y * M[2, 1]) / (Wh * Wh)
    return float(math.degrees(math.atan2(ux, -uy)) % 360.0)


# ----------------------------------------------------------------------------------------------
# Noise
# ----------------------------------------------------------------------------------------------


def value_noise(rng: np.random.Generator, h: int, w: int, cells_y: float, cells_x: float) -> np.ndarray:
    """Smooth random field (float32, ~unit variance) with about cells_x x cells_y features."""
    gy = max(2, int(round(cells_y)) + 2)
    gx = max(2, int(round(cells_x)) + 2)
    g = rng.standard_normal((gy, gx)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def fbm(rng: np.random.Generator, h: int, w: int, feature_px: float, octaves: int = 5,
        persistence: float = 0.5, aspect: float = 1.0) -> np.ndarray:
    """Fractal noise normalised to zero mean, unit std.

    `feature_px` is the size of the largest features; `aspect` > 1 stretches them horizontally.
    """
    out = np.zeros((h, w), np.float32)
    amp = 1.0
    fx = feature_px * aspect
    fy = feature_px
    for _ in range(octaves):
        if fx < 1.0 or fy < 1.0:  # sub-pixel features add nothing but aliasing
            break
        cx = w / fx
        cy = h / fy
        out += amp * value_noise(rng, h, w, cy, cx)
        amp *= persistence
        fx /= 2.0
        fy /= 2.0
    std = float(out.std())
    return (out - float(out.mean())) / (std if std > 1e-6 else 1.0)


def fbm1d(rng: np.random.Generator, n: int, feature: float, octaves: int = 6, persistence: float = 0.5) -> np.ndarray:
    return fbm(rng, 1, n, feature, octaves, persistence)[0] if n > 0 else np.zeros(0, np.float32)


def smooth_grid(h: int, w: int, factor: int = 8) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Coordinate grids (yy, xx) in full-resolution pixel units, sampled every ~`factor` px.

    Smooth fields (lighting, vignette, glare) are evaluated on this grid and resized up with
    `upsample`, which is ~50x cheaper than evaluating them per pixel and visually identical.
    """
    hl, wl = h // factor + 2, w // factor + 2
    ys = np.linspace(0, h - 1, hl, dtype=np.float32)
    xs = np.linspace(0, w - 1, wl, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return yy, xx, hl, wl


def upsample(field: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a field sampled on smooth_grid(h, w) back to h x w (end points aligned)."""
    hl, wl = field.shape[:2]
    M = np.array([[(wl - 1) / max(w - 1, 1), 0, 0], [0, (hl - 1) / max(h - 1, 1), 0]], np.float32)
    return cv2.warpAffine(field.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----------------------------------------------------------------------------------------------
# Colour helpers
# ----------------------------------------------------------------------------------------------


def hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    """h in [0, 1), s, v in [0, 1] -> float RGB in [0, 1]."""
    h = h % 1.0
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return np.array([r, g, b], np.float32)


def lerp(a, b, t):
    return a + (b - a) * t


def to_u8(img: np.ndarray) -> np.ndarray:
    """Float RGB 0..1 -> uint8 (rounded, clipped)."""
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# Byte-stable writers
# ----------------------------------------------------------------------------------------------


def write_png(path: Path, img_rgb_or_gray: np.ndarray) -> None:
    img = img_rgb_or_gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    assert ok, f"PNG encode failed for {path}"
    path.write_bytes(buf.tobytes())


def write_jpg(path: Path, img_rgb: np.ndarray, quality: int) -> None:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok, f"JPEG encode failed for {path}"
    path.write_bytes(buf.tobytes())


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img


def pts_json(pts: np.ndarray, decimals: int = 2) -> list[list[float]]:
    """Points as [[x, y], ...] with fixed rounding (keeps files byte-stable and small)."""
    r = np.round(np.asarray(pts, np.float64), decimals) + 0.0  # + 0.0 turns -0.0 into 0.0
    return [[float(x), float(y)] for x, y in r]


def write_json(path: Path, obj: object) -> None:
    """Compact-but-readable JSON: one line per top-level list item, deterministic bytes."""
    path.write_text(_dumps(obj), encoding="utf-8", newline="\n")


def _dumps(obj: object) -> str:
    if isinstance(obj, list) and obj and all(isinstance(o, dict) for o in obj):
        return "[\n" + ",\n".join("  " + json.dumps(o, separators=(",", ":")) for o in obj) + "\n]\n"
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, list) and v and all(isinstance(o, dict) for o in v):
                inner = ",\n".join("    " + json.dumps(o, separators=(",", ":")) for o in v)
                lines.append(f"  {json.dumps(k)}: [\n{inner}\n  ]")
            else:
                lines.append(f"  {json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}")
        return "{\n" + ",\n".join(lines) + "\n}\n"
    return json.dumps(obj, separators=(",", ":")) + "\n"
