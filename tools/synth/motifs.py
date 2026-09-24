"""Puzzle motifs: real CC0 artwork when downloaded, otherwise procedural puzzle-like pictures.

Why procedural motifs must be good: until the user approves downloading CC0 art
(fetch_sources.py), the matching engine is tuned only on these. Real puzzle pictures mix two
kinds of pieces, and both must exist here:
- hard, low-texture pieces: sky gradients with soft clouds, calm water, dark painting
  backgrounds, flat illustration fills — the engine can only use colour and faint gradients;
- high-detail pieces: foliage, flowers, rocks, buildings with window grids, many small objects,
  stripes and text-like marks — easy individually, but repetitive enough to confuse.
Palettes vary widely (day, sunset, dusk, pastel, autumn, neon) so no colour prior leaks in.

Real motifs come from tools/synth/sources/motifs/*.jpg|png (see fetch_sources.py and
sources/manifest.json for provenance). All motifs are returned as uint8 RGB at the requested size.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .common import choice_weighted, fbm, fbm1d, hsv_to_rgb, lerp, read_rgb, smoothstep, to_u8

SOURCES_DIR = Path(__file__).resolve().parent / "sources"
MOTIF_DIR = SOURCES_DIR / "motifs"

STYLES = ("landscape", "city", "illustration", "stilllife", "garden")
# Mix of procedural styles. Guessed from what sells as jigsaws: landscapes and cities dominate,
# busy illustrations ("Wasgij"-like) and paintings are common, gardens/flowers are the classic
# "all pieces look alike" hard case.
STYLE_WEIGHTS = (0.30, 0.22, 0.20, 0.14, 0.14)


# ----------------------------------------------------------------------------------------------
# Sources on disk
# ----------------------------------------------------------------------------------------------


def source_motifs() -> list[tuple[Path, str]]:
    """(path, provenance) for downloaded artwork, sorted so the choice is deterministic."""
    if not MOTIF_DIR.is_dir():
        return []
    manifest: dict[str, str] = {}
    mf = SOURCES_DIR / "manifest.json"
    if mf.is_file():
        data = json.loads(mf.read_text(encoding="utf-8"))
        for item in data.get("motifs", []):
            manifest[item["file"]] = item.get("id", "")
    files = sorted(p for p in MOTIF_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return [(p, manifest.get(p.name) or f"file:{p.name}") for p in files]


def load_source_motif(path: Path, long_side: int) -> np.ndarray:
    img = read_rgb(path)
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, size, interpolation=interp)


def motif_size(long_side: int, aspect: float) -> tuple[int, int]:
    """(width, height) for a motif whose long side is `long_side` and aspect is W/H."""
    if aspect >= 1:
        return long_side, max(8, round(long_side / aspect))
    return max(8, round(long_side * aspect)), long_side


def pick_aspect(rng: np.random.Generator) -> float:
    """Box-picture aspect ratios. Guessed from common formats: 4:3, 3:2, 1.4 (70x50 cm), portrait, square."""
    options = [4 / 3, 3 / 2, 1.4, 1.25, 3 / 4, 2 / 3, 1.0, 1.8]
    weights = [0.24, 0.24, 0.18, 0.06, 0.1, 0.08, 0.05, 0.05]
    return float(options[int(rng.choice(len(options), p=np.array(weights) / sum(weights)))])


# ----------------------------------------------------------------------------------------------
# Procedural motifs
# ----------------------------------------------------------------------------------------------


def procedural_motif(rng: np.random.Generator, width: int, height: int, style: str | None = None) -> tuple[np.ndarray, str]:
    """A puzzle-like picture (uint8 RGB, height x width) and the style name used."""
    if style is None:
        style = choice_weighted(rng, STYLES, STYLE_WEIGHTS)
    build = {"landscape": _landscape, "city": _city, "illustration": _illustration,
             "stilllife": _stilllife, "garden": _garden}[style]
    img = build(rng, height, width)
    img = _finish(rng, img)
    return to_u8(img), style


def _finish(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    """Printed-artwork finish: soften synthetic hard edges, add a faint painterly modulation, grade."""
    h, w = img.shape[:2]
    img = cv2.GaussianBlur(img, (0, 0), 0.6)
    stroke = fbm(rng, max(8, h // 4), max(8, w // 4), feature_px=w / 360, octaves=3, persistence=0.6,
                 aspect=rng.uniform(0.5, 2.0))
    mod = cv2.resize(1.0 + 0.02 * stroke, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.multiply(img, cv2.merge([mod, mod, mod]))
    # Global grade: saturation and warmth vary between artworks.
    sat = rng.uniform(0.85, 1.2)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.addWeighted(img, sat, cv2.merge([gray, gray, gray]), 1.0 - sat, 0.0)
    img = img * np.array([rng.uniform(0.96, 1.04), 1.0, rng.uniform(0.96, 1.04)], np.float32)
    return np.clip(img, 0.0, 1.0)


def _up(lo: np.ndarray, h: int, w: int) -> np.ndarray:
    """Upsample a smooth half-resolution layer to the full motif size."""
    return np.clip(cv2.resize(lo, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def _yy_xx(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    return np.mgrid[0:h, 0:w].astype(np.float32)


# ---------- sky ----------

_SKY_MODES = ("day", "sunset", "dusk", "overcast", "pastel", "golden")


def _sky(rng: np.random.Generator, h: int, w: int, horizon: float) -> tuple[np.ndarray, np.ndarray]:
    """Sky over the whole canvas (only the part above the horizon is kept by callers).

    Returns (sky, horizon colour). Clouds are sparse and soft on purpose: large patches of pure
    gradient are the hardest pieces a real puzzle has.
    """
    mode = _SKY_MODES[int(rng.integers(len(_SKY_MODES)))]
    hue = rng.uniform(-0.03, 0.03)
    if mode == "day":
        top = hsv_to_rgb(0.6 + hue, rng.uniform(0.5, 0.8), rng.uniform(0.55, 0.85))
        hor = hsv_to_rgb(0.57 + hue, rng.uniform(0.12, 0.3), rng.uniform(0.88, 0.98))
        sun = np.array([1.0, 0.98, 0.9], np.float32)
    elif mode == "sunset":
        top = hsv_to_rgb(0.68 + hue, rng.uniform(0.4, 0.7), rng.uniform(0.25, 0.5))
        hor = hsv_to_rgb(0.07 + hue, rng.uniform(0.55, 0.85), rng.uniform(0.9, 1.0))
        sun = np.array([1.0, 0.85, 0.55], np.float32)
    elif mode == "dusk":
        top = hsv_to_rgb(0.66 + hue, rng.uniform(0.5, 0.8), rng.uniform(0.12, 0.3))
        hor = hsv_to_rgb(0.8 + hue, rng.uniform(0.25, 0.5), rng.uniform(0.5, 0.75))
        sun = np.array([0.95, 0.7, 0.75], np.float32)
    elif mode == "overcast":
        v = rng.uniform(0.6, 0.85)
        top = hsv_to_rgb(0.6 + hue, rng.uniform(0.05, 0.15), v)
        hor = hsv_to_rgb(0.6 + hue, rng.uniform(0.02, 0.08), min(1.0, v + 0.1))
        sun = np.array([0.95, 0.95, 0.95], np.float32)
    elif mode == "pastel":
        top = hsv_to_rgb(rng.uniform(0.5, 0.9), rng.uniform(0.2, 0.4), rng.uniform(0.8, 0.95))
        hor = hsv_to_rgb(rng.uniform(0.0, 0.15), rng.uniform(0.15, 0.35), rng.uniform(0.92, 1.0))
        sun = np.array([1.0, 0.95, 0.9], np.float32)
    else:  # golden
        top = hsv_to_rgb(0.58 + hue, rng.uniform(0.3, 0.55), rng.uniform(0.6, 0.8))
        hor = hsv_to_rgb(0.11 + hue, rng.uniform(0.35, 0.6), rng.uniform(0.9, 1.0))
        sun = np.array([1.0, 0.92, 0.7], np.float32)

    yy, xx = _yy_xx(h, w)
    t = np.clip(yy / max(horizon, 1.0), 0.0, 1.0) ** rng.uniform(0.9, 2.2)
    sky = lerp(top[None, None, :], hor[None, None, :], t[..., None])
    # Sun glow near the horizon.
    sx, sy = rng.uniform(0.1, 0.9) * w, horizon * rng.uniform(0.55, 1.0)
    d2 = ((xx - sx) ** 2 + ((yy - sy) * 1.4) ** 2) / (w * rng.uniform(0.12, 0.35)) ** 2
    glow = np.exp(-d2) * rng.uniform(0.1, 0.45 if mode in ("sunset", "golden") else 0.25)
    sky = sky + (sun - sky) * glow[..., None]

    # Clouds: distinct cloud bodies over clear gradient sky. A very-low-frequency field decides
    # where the sky is clear, so big cloudless areas (the hardest pieces) always exist.
    kind = rng.choice(["none", "cumulus", "stratus", "cirrus"], p=[0.15, 0.4, 0.3, 0.15])
    if kind != "none":
        aspect = {"cumulus": rng.uniform(1.3, 2.2), "stratus": rng.uniform(2.5, 4.5), "cirrus": rng.uniform(6, 12)}[kind]
        feat = w / {"cumulus": rng.uniform(4, 8), "stratus": rng.uniform(2.5, 5), "cirrus": rng.uniform(3, 6)}[kind]
        n = fbm(rng, h, w, feature_px=feat, octaves=6, persistence=0.5, aspect=aspect)
        clear = fbm(rng, h, w, feature_px=w / 1.2, octaves=2)
        th = rng.uniform(0.4, 1.3) - 0.6 * clear
        soft = {"cumulus": rng.uniform(0.15, 0.35), "stratus": rng.uniform(0.4, 0.8), "cirrus": rng.uniform(0.3, 0.6)}[kind]
        c = smoothstep(-soft, soft, n - th)
        c *= smoothstep(0.0, 0.3, 1.0 - yy / max(horizon, 1.0))  # thinner right above the horizon
        c *= {"cumulus": rng.uniform(0.8, 1.0), "stratus": rng.uniform(0.5, 0.85), "cirrus": rng.uniform(0.35, 0.6)}[kind]
        nb = cv2.GaussianBlur(n, (0, 0), max(1.0, h * 0.006))
        shift = max(2, int(h * 0.012))
        lit = np.clip(0.55 + (nb - np.roll(nb, shift, axis=0)) * 2.5, 0.0, 1.0)
        cloud_lit = np.clip(lerp(hor, np.ones(3, np.float32), 0.75), 0, 1)
        cloud_dark = np.clip(lerp(top, hor, 0.6) * rng.uniform(0.75, 0.95), 0, 1)
        if mode in ("sunset", "golden", "dusk"):
            cloud_lit = np.clip(lerp(cloud_lit, sun, 0.6), 0, 1)
            cloud_dark = np.clip(lerp(cloud_dark, np.array([0.45, 0.3, 0.45], np.float32), 0.4), 0, 1)
        col = lerp(cloud_dark[None, None, :], cloud_lit[None, None, :], lit[..., None])
        sky = sky * (1 - c[..., None]) + col * c[..., None]
    return np.clip(sky, 0, 1).astype(np.float32), hor


# ---------- landscape parts ----------


def _mountains(rng: np.random.Generator, img: np.ndarray, base_y: float, hor: np.ndarray, layers: int) -> np.ndarray:
    h, w = img.shape[:2]
    yy, _ = _yy_xx(h, w)
    rock_hue = rng.uniform(0.05, 0.12) if rng.random() < 0.6 else rng.uniform(0.55, 0.7)
    for i in range(layers):
        far = 1.0 - i / max(layers, 1)
        amp = h * rng.uniform(0.08, 0.22) * (0.6 + 0.6 * far)
        ridge_noise = np.abs(fbm1d(rng, w, feature=w / rng.uniform(1.5, 4.0), octaves=8, persistence=0.52))
        ridge = base_y - amp * (0.3 + 0.7 * (1 - np.clip(ridge_noise / 2.5, 0, 1))) + i * h * 0.03
        mask = np.clip(yy - ridge[None, :] + 0.5, 0.0, 1.0)
        base = hsv_to_rgb(rock_hue, rng.uniform(0.15, 0.4), rng.uniform(0.3, 0.6))
        greenish = hsv_to_rgb(rng.uniform(0.22, 0.35), rng.uniform(0.3, 0.6), rng.uniform(0.25, 0.45))
        tex = fbm(rng, h, w, feature_px=w / 60, octaves=5, persistence=0.6, aspect=0.5)
        depth = np.clip((yy - ridge[None, :]) / (amp + 1), 0, 1)
        col = lerp(base[None, None, :], greenish[None, None, :], smoothstep(0.2, 0.9, depth)[..., None])
        col = col * (1 + 0.12 * tex[..., None])
        if rng.random() < 0.55 and i < 2:
            snow_line = ridge[None, :] + amp * rng.uniform(0.15, 0.4) * (1 + 0.5 * tex)
            snow = np.clip((snow_line - yy) / (h * 0.01), 0, 1)
            col = lerp(col, np.array([0.95, 0.96, 0.98], np.float32), snow[..., None] * 0.9)
        haze = 0.65 * far ** 1.2
        col = lerp(col, hor[None, None, :], haze)
        img = img * (1 - mask[..., None]) + col * mask[..., None]
    return img


def _treeline(rng: np.random.Generator, img: np.ndarray, base_y: float, depth: float, tint: np.ndarray | None = None) -> np.ndarray:
    """A band of conifers and round trees whose tops form a jagged silhouette at base_y."""
    h, w = img.shape[:2]
    top = np.full(w, base_y, np.float32)
    xs = np.arange(w, dtype=np.float32)
    spacing = w / rng.uniform(80, 220)
    n = int(w / spacing * 1.3)
    tree_h = h * rng.uniform(0.03, 0.09)
    for _ in range(n):
        xc = rng.uniform(-spacing, w + spacing)
        hh = tree_h * rng.uniform(0.5, 1.4)
        if rng.random() < 0.55:  # conifer: triangle
            ww = hh * rng.uniform(0.22, 0.35)
            x0, x1 = max(0, int(xc - ww)), min(w, int(xc + ww) + 1)
            if x0 >= x1:
                continue
            prof = 1.0 - np.abs(xs[x0:x1] - xc) / ww
        else:  # broadleaf: half ellipse
            ww = hh * rng.uniform(0.4, 0.7)
            x0, x1 = max(0, int(xc - ww)), min(w, int(xc + ww) + 1)
            if x0 >= x1:
                continue
            prof = np.sqrt(np.clip(1.0 - ((xs[x0:x1] - xc) / ww) ** 2, 0, 1))
        top[x0:x1] = np.minimum(top[x0:x1], base_y - hh * np.clip(prof, 0, 1))
    yy, _ = _yy_xx(h, w)
    mask = np.clip(yy - top[None, :] + 0.5, 0, 1) * np.clip(base_y + depth - yy + 0.5, 0, 1)
    g = hsv_to_rgb(rng.uniform(0.22, 0.4), rng.uniform(0.35, 0.75), rng.uniform(0.12, 0.35))
    if tint is not None:
        g = lerp(g, tint, 0.3)
    leaf = fbm(rng, h, w, feature_px=max(3.0, h * 0.004), octaves=4, persistence=0.7)
    big = fbm(rng, h, w, feature_px=w / 30, octaves=3)
    col = g[None, None, :] * (1 + 0.25 * leaf[..., None] + 0.12 * big[..., None])
    return img * (1 - mask[..., None]) + col * mask[..., None]


def _water(rng: np.random.Generator, img: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """Mirror everything above y0 into [y0, y1) with ripples: calm water with reflections."""
    h, w = img.shape[:2]
    y1 = min(y1, h)
    n = y1 - y0
    if n <= 2:
        return img
    ys = np.arange(y0, y1, dtype=np.float32)
    dist = (ys - y0) / n
    src_y = np.clip(2 * y0 - ys, 0, y0 - 1)
    amp = (1.0 + 14.0 * dist ** 1.5) * rng.uniform(0.5, 1.5) * (w / 3000)
    freq = rng.uniform(0.15, 0.4) * (3000 / w)
    ripple = np.sin(ys * freq + rng.uniform(0, 6.28)) + 0.6 * np.sin(ys * freq * 2.7 + rng.uniform(0, 6.28))
    wobble = fbm(rng, n, w, feature_px=max(4.0, w / 200), octaves=3, aspect=6.0)
    map_x = np.arange(w, dtype=np.float32)[None, :] + (amp * ripple)[:, None] + wobble * amp[:, None] * 0.8
    map_y = np.repeat(src_y[:, None], w, axis=1) + wobble * 1.5
    refl = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)
    refl = cv2.GaussianBlur(refl, (1, 0), sigmaX=0.1, sigmaY=max(1.0, h * 0.002))
    water_col = hsv_to_rgb(rng.uniform(0.5, 0.62), rng.uniform(0.3, 0.7), rng.uniform(0.2, 0.45))
    mix = (0.2 + 0.45 * dist)[:, None, None] * rng.uniform(0.6, 1.2)
    refl = lerp(refl * rng.uniform(0.7, 0.9), water_col[None, None, :], np.clip(mix, 0, 0.9))
    glint = fbm(rng, n, w, feature_px=max(2.0, w / 300), octaves=3, aspect=25.0)
    refl = refl + (0.25 * smoothstep(1.6, 2.6, glint) * (1 - dist[:, None]))[..., None]
    out = img.copy()
    out[y0:y1] = refl
    return out


def _flowers(rng: np.random.Generator, canvas: np.ndarray, y0: int, y1: int, density: float, palette: list[np.ndarray],
             size_px: tuple[float, float]) -> None:
    """Scatter flowers (per 1000 px^2: `density`) on a uint8 canvas, bigger toward the bottom."""
    h, w = canvas.shape[:2]
    n = int(w * max(1, y1 - y0) * density / 1000)
    if n == 0:
        return
    ys = np.sort(rng.uniform(y0, y1, n))
    xs = rng.uniform(0, w, n)
    ts = (ys - y0) / max(1, y1 - y0)
    rs = (size_px[0] + (size_px[1] - size_px[0]) * ts) * rng.uniform(0.6, 1.3, n)
    cols = np.stack([palette[i] for i in rng.integers(0, len(palette), n)]) * rng.uniform(0.8, 1.1, (n, 1))
    _draw_flowers(rng, canvas, xs, ys, rs, cols)


def _draw_flowers(rng: np.random.Generator, canvas: np.ndarray, xs: np.ndarray, ys: np.ndarray, rs: np.ndarray,
                  cols: np.ndarray) -> None:
    """Flowers as rose-curve polygons (petal ring) with a yellow or brown centre; dots when tiny."""
    n = len(xs)
    petals = rng.integers(4, 7, n)
    rot = rng.uniform(0, 2 * np.pi, n)
    centre_yellow = rng.random(n) < 0.6
    theta = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    ang = theta[None, :] + rot[:, None]
    rad = rs[:, None] * (0.45 + 0.55 * np.abs(np.cos(petals[:, None] * theta[None, :] / 2)) ** 0.7)
    poly = np.stack([xs[:, None] + rad * np.cos(ang), ys[:, None] + rad * np.sin(ang)], axis=-1)
    poly = np.round(poly * 4).astype(np.int32)
    colours = np.clip(cols * 255, 0, 255).astype(np.int32)
    for i in range(n):
        colour = (int(colours[i, 0]), int(colours[i, 1]), int(colours[i, 2]))
        cx, cy, r = int(xs[i] * 4), int(ys[i] * 4), float(rs[i])
        if r < 2.5:
            cv2.circle(canvas, (cx, cy), max(1, int(r * 4)), colour, -1, cv2.LINE_AA, shift=2)
            continue
        cv2.fillPoly(canvas, [poly[i]], colour, cv2.LINE_AA, shift=2)
        centre = (230, 190, 40) if centre_yellow[i] else (90, 60, 30)
        cv2.circle(canvas, (cx, cy), max(1, int(r * 0.28 * 4)), centre, -1, cv2.LINE_AA, shift=2)


def _flower(rng: np.random.Generator, canvas: np.ndarray, x: float, y: float, r: float, rgb: np.ndarray) -> None:
    _draw_flowers(rng, canvas, np.array([x]), np.array([y]), np.array([r]), np.asarray(rgb, np.float64)[None, :])


def _rocks(rng: np.random.Generator, canvas: np.ndarray, y0: int, y1: int, count: int) -> None:
    h, w = canvas.shape[:2]
    for _ in range(count):
        cx, cy = rng.uniform(0, w), rng.uniform(y0, y1)
        t = (cy - y0) / max(1, y1 - y0)
        rx = w * rng.uniform(0.01, 0.05) * (0.5 + t)
        ry = rx * rng.uniform(0.4, 0.75)
        base = hsv_to_rgb(rng.uniform(0.05, 0.15), rng.uniform(0.05, 0.25), rng.uniform(0.35, 0.65))
        for k in range(6):  # stacked ellipses fake a lit, rounded rock
            f = 1 - k / 6
            col = tuple(int(v) for v in np.clip(base * (0.65 + 0.12 * k) * 255, 0, 255))
            cv2.ellipse(canvas, (int(cx * 4), int((cy - ry * 0.08 * k) * 4)),
                        (max(1, int(rx * f * 4)), max(1, int(ry * f * 4))), rng.uniform(-15, 15), 0, 360, col, -1,
                        cv2.LINE_AA, shift=2)


def _meadow_base(rng: np.random.Generator, img: np.ndarray, y0: int, autumn: bool) -> np.ndarray:
    """Grass from y0 down: far-to-near colour gradient, vertical grass streaks, patches."""
    h, w = img.shape[:2]
    if y0 >= h - 2:
        return img
    n = h - y0
    hue = rng.uniform(0.08, 0.14) if autumn else rng.uniform(0.18, 0.32)
    far = hsv_to_rgb(hue, rng.uniform(0.3, 0.6), rng.uniform(0.45, 0.7))
    near = hsv_to_rgb(hue + rng.uniform(-0.03, 0.03), rng.uniform(0.5, 0.85), rng.uniform(0.25, 0.45))
    t = (np.arange(n, dtype=np.float32) / n)[:, None, None]
    base = lerp(far[None, None, :], near[None, None, :], t)
    grass = fbm(rng, n, w, feature_px=max(3.0, h * 0.01), octaves=4, persistence=0.65, aspect=0.12)
    patches = fbm(rng, n, w, feature_px=w / 12, octaves=3)
    out = img.copy()
    out[y0:] = base * (1 + 0.18 * grass[..., None] + 0.12 * patches[..., None])
    return out


def _meadow_details(rng: np.random.Generator, canvas: np.ndarray, y0: int) -> None:
    """Flowers and rocks on the grass, drawn at full resolution on a uint8 canvas."""
    h, w = canvas.shape[:2]
    n = h - y0
    if n < 4:
        return
    if rng.random() < 0.75:
        pal = [hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.5, 0.95), rng.uniform(0.75, 1.0))
               for _ in range(int(rng.integers(2, 6)))]
        pal.append(np.array([0.97, 0.97, 0.95], np.float32))
        _flowers(rng, canvas, y0 + int(n * 0.1), h, density=rng.uniform(0.6, 3.0), palette=pal,
                 size_px=(w / 1500, w / 180))
    if rng.random() < 0.5:
        _rocks(rng, canvas, y0, h, int(rng.integers(2, 12)))


def _landscape(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    hl, wl = h // 2, w // 2
    f = hl / h
    horizon = h * rng.uniform(0.32, 0.55)
    lo, hor = _sky(rng, hl, wl, horizon * f)
    if rng.random() < 0.8:
        lo = _mountains(rng, lo, horizon * f, hor, int(rng.integers(1, 4)))
    has_water = rng.random() < 0.65
    shore = int(horizon + h * rng.uniform(0.0, 0.04))
    lo = _treeline(rng, lo, shore * f, depth=hl * rng.uniform(0.01, 0.05))
    autumn = rng.random() < 0.25
    meadow_y: int | None
    if has_water:
        shore_bottom = int(shore + h * rng.uniform(0.01, 0.04))
        water_end = int(h * rng.uniform(0.72, 1.0))
        lo = _water(rng, lo, int(shore_bottom * f), int(water_end * f))
        meadow_y = water_end if water_end < h - 8 else None
    else:
        meadow_y = shore + int(h * 0.02)
    if meadow_y is not None:
        lo = _meadow_base(rng, lo, int(meadow_y * f), autumn)
    canvas = to_u8(_up(lo, h, w))
    if meadow_y is not None:
        _meadow_details(rng, canvas, meadow_y)
    return canvas.astype(np.float32) / 255.0


# ---------- city ----------


def _city(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    ground = int(h * rng.uniform(0.55, 0.72))
    hl, wl = h // 2, w // 2
    lo, hor = _sky(rng, hl, wl, ground * hl / h)
    night = float(np.mean(hor)) < 0.6
    canvas = to_u8(_up(lo, h, w))
    facade_pal = [
        (0.62, 0.3, 0.22), (0.72, 0.45, 0.35), (0.85, 0.78, 0.62), (0.7, 0.7, 0.72), (0.9, 0.9, 0.88),
        (0.35, 0.45, 0.55), (0.45, 0.6, 0.65), (0.88, 0.7, 0.5), (0.75, 0.55, 0.6), (0.55, 0.62, 0.45),
    ]
    rows_back = [(0.55, 0.45), (1.0, 0.0)]  # (height scale, haze) for back and front rows
    for hs, haze in rows_back:
        x = -rng.uniform(0, w * 0.05)
        while x < w:
            bw = w * rng.uniform(0.025, 0.085)
            bh = h * rng.uniform(0.08, 0.42) * hs
            if rng.random() < 0.08:
                bh *= 1.6  # a tower
            x0, x1 = int(x), int(min(w, x + bw))
            y0 = int(max(0, ground - bh))
            fc = np.array(facade_pal[int(rng.integers(len(facade_pal)))], np.float32) * rng.uniform(0.8, 1.1)
            fc = lerp(fc, hor, haze)
            if x1 > max(x0, 0):
                xa = max(x0, 0)
                shade = np.linspace(1.0, rng.uniform(0.75, 0.95), x1 - xa, dtype=np.float32)[None, :, None]
                canvas[y0:ground, xa:x1] = to_u8(np.clip(fc[None, None, :] * shade, 0, 1))
                # Roof
                roof = rng.random()
                rc = tuple(int(v) for v in to_u8(np.clip(fc * 0.7, 0, 1)))
                if roof < 0.25:
                    pts = np.array([[x0, y0], [x1, y0], [(x0 + x1) // 2, y0 - int(bw * rng.uniform(0.2, 0.5))]], np.int32)
                    cv2.fillPoly(canvas, [pts], rc, cv2.LINE_AA)
                elif roof < 0.35:
                    cv2.ellipse(canvas, ((x0 + x1) // 2, y0), (max(1, (x1 - x0) // 3), max(1, (x1 - x0) // 4)), 0, 180, 360, rc, -1, cv2.LINE_AA)
                elif roof < 0.45:
                    cv2.line(canvas, ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y0 - int(bh * 0.2)), rc, max(1, w // 1500), cv2.LINE_AA)
                # Windows
                ww = max(2, int(bw * rng.uniform(0.07, 0.14)))
                wh = max(3, int(ww * rng.uniform(1.1, 1.8)))
                gx = max(1, int(ww * rng.uniform(0.6, 1.4)))
                gy = max(1, int(wh * rng.uniform(0.5, 1.0)))
                p_lit = (0.55 if night else 0.08) * rng.uniform(0.5, 1.5)
                glass = np.clip(lerp(hor, np.array([0.2, 0.25, 0.3], np.float32), rng.uniform(0.3, 0.8)), 0, 1)
                lit_c = np.array([1.0, 0.85, 0.5], np.float32)
                for wy in range(y0 + gy, ground - wh - gy, wh + gy):
                    for wx in range(x0 + gx, x1 - ww - gx + 1, ww + gx):
                        if wx < 0:
                            continue
                        c = lit_c if rng.random() < p_lit else glass * rng.uniform(0.8, 1.1)
                        c = lerp(c, hor, haze)
                        canvas[wy:wy + wh, wx:wx + ww] = to_u8(np.clip(c, 0, 1))
            x += bw + (w * rng.uniform(0.0, 0.01) if rng.random() < 0.3 else 0)
    img = canvas.astype(np.float32) / 255.0
    if rng.random() < 0.6:  # waterfront: the reflection is soft anyway, so compute it at half size
        small = cv2.resize(img, (wl, hl), interpolation=cv2.INTER_AREA)
        g = int(ground * hl / h)
        small = _water(rng, small, g, hl)
        img[ground:] = _up(small, h, w)[ground:]
        return img
    return _street(rng, img, ground)


def _street(rng: np.random.Generator, img: np.ndarray, y0: int) -> np.ndarray:
    h, w = img.shape[:2]
    canvas = to_u8(img)
    road = hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.0, 0.08), rng.uniform(0.3, 0.5))
    walk = hsv_to_rgb(rng.uniform(0.05, 0.12), rng.uniform(0.05, 0.2), rng.uniform(0.6, 0.8))
    wy = y0 + int((h - y0) * rng.uniform(0.1, 0.25))
    canvas[y0:wy] = to_u8(walk)
    canvas[wy:] = to_u8(road)
    tex = fbm(rng, h - y0, w, feature_px=4.0, octaves=3)
    f = canvas[y0:].astype(np.float32) * (1 + 0.06 * tex[..., None])
    canvas[y0:] = np.clip(f, 0, 255).astype(np.uint8)
    lane_y = wy + (h - wy) // 2
    for x in range(0, w, max(8, w // 25)):
        cv2.rectangle(canvas, (x, lane_y), (x + w // 50, lane_y + max(2, h // 200)), (235, 235, 220), -1)
    for _ in range(int(rng.integers(4, 14))):  # cars
        cx = rng.uniform(0, w)
        cy = rng.uniform(wy + (h - wy) * 0.1, h - (h - wy) * 0.1)
        cw_ = w * rng.uniform(0.04, 0.08)
        chh = cw_ * rng.uniform(0.35, 0.5)
        col = tuple(int(v) for v in to_u8(hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.95))))
        cv2.rectangle(canvas, (int(cx), int(cy)), (int(cx + cw_), int(cy + chh)), col, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (int(cx + cw_ * 0.2), int(cy - chh * 0.45)), (int(cx + cw_ * 0.75), int(cy)), col, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (int(cx + cw_ * 0.27), int(cy - chh * 0.35)), (int(cx + cw_ * 0.68), int(cy - 2)), (60, 80, 100), -1)
        for fx in (0.22, 0.78):
            cv2.circle(canvas, (int(cx + cw_ * fx), int(cy + chh)), max(2, int(chh * 0.28)), (25, 25, 25), -1, cv2.LINE_AA)
    return canvas.astype(np.float32) / 255.0


# ---------- illustration ----------


def _bright_palette(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    base = rng.uniform(0, 1)
    scheme = rng.integers(3)
    cols = []
    for i in range(n):
        if scheme == 0:
            hue = base + i / n
        elif scheme == 1:
            hue = base + rng.uniform(-0.12, 0.12) + (0.5 if i % 2 else 0.0)
        else:
            hue = rng.uniform(0, 1)
        cols.append(hsv_to_rgb(hue, rng.uniform(0.35, 0.85), rng.uniform(0.55, 0.97)))
    return cols


def _star(cx: float, cy: float, r: float, rot: float, points: int = 5) -> np.ndarray:
    a = rot + np.arange(points * 2) * math.pi / points
    rr = np.where(np.arange(points * 2) % 2 == 0, r, r * 0.45)
    return np.stack([cx + rr * np.cos(a), cy + rr * np.sin(a)], axis=1)


def _text_marks(rng: np.random.Generator, canvas: np.ndarray, x: float, y: float, w: float, h: float, colour) -> None:
    """Rows of short dashes / letters inside a box, like a sign or a page of text."""
    if rng.random() < 0.5 and h > 14:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        n = max(2, int(w / (h * 0.55)))
        s = "".join(letters[int(i)] for i in rng.integers(0, len(letters), min(n, 14)))
        scale = h / 30.0
        cv2.putText(canvas, s, (int(x), int(y + h * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                    max(1, int(scale * 2)), cv2.LINE_AA)
        return
    line_h = max(2.0, h / rng.uniform(3, 7))
    yy = y
    while yy + line_h * 0.5 < y + h:
        xx = x
        while xx < x + w:
            seg = rng.uniform(0.5, 2.5) * line_h
            x_end = min(x + w, xx + seg)
            cv2.rectangle(canvas, (int(xx), int(yy)), (int(x_end), int(yy + line_h * 0.45)), colour, -1)
            xx = x_end + line_h * 0.5
        yy += line_h


def _illustration(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    pal = _bright_palette(rng, int(rng.integers(6, 11)))
    dark = hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.2, 0.6), rng.uniform(0.05, 0.25))
    # Large flat regions (sky, hills, sea) — the low-texture part of a cartoon picture.
    hl, wl = h // 2, w // 2
    yy, xx = _yy_xx(hl, wl)
    img = np.empty((hl, wl, 3), np.float32)
    img[:] = lerp(pal[0], np.ones(3, np.float32), 0.35)
    n_bands = int(rng.integers(2, 5))
    for i in range(n_bands):
        base = hl * (0.3 + 0.6 * (i + 1) / (n_bands + 1))
        wave = hl * rng.uniform(0.02, 0.1) * np.sin(xx[0] / wl * rng.uniform(3, 12) + rng.uniform(0, 6.28))
        wave = wave + hl * 0.02 * fbm1d(rng, wl, feature=wl / 4, octaves=3)
        mask = np.clip(yy - (base + wave)[None, :] + 0.5, 0, 1)
        c = pal[(i + 1) % len(pal)] * rng.uniform(0.75, 1.0)
        grad = 1.0 + 0.08 * (yy - base) / hl
        img = img * (1 - mask[..., None]) + (c[None, None, :] * grad[..., None]) * mask[..., None]
    canvas = to_u8(_up(img, h, w))
    outline = tuple(int(v) for v in to_u8(dark))
    thick = max(1, w // 900)
    n_obj = int(w * h / 1e6 * rng.uniform(60, 180))
    sizes = np.exp(rng.normal(math.log(w / 70), 0.6, n_obj))
    order = np.argsort(-sizes)  # big objects first, small ones on top
    for i in order:
        s = float(min(sizes[i], w / 8))
        cx, cy = rng.uniform(-0.02, 1.02) * w, rng.uniform(0.02, 1.02) * h
        col = tuple(int(v) for v in to_u8(pal[int(rng.integers(len(pal)))]))
        kind = int(rng.integers(9))
        edge = rng.random() < 0.6
        if kind == 0:
            cv2.circle(canvas, (int(cx * 4), int(cy * 4)), max(1, int(s * 2)), col, -1, cv2.LINE_AA, shift=2)
            if edge:
                cv2.circle(canvas, (int(cx * 4), int(cy * 4)), max(1, int(s * 2)), outline, thick, cv2.LINE_AA, shift=2)
        elif kind == 1:
            pts = _star(cx, cy, s * 0.6, rng.uniform(0, 6.28), int(rng.integers(4, 8)))
            cv2.fillPoly(canvas, [np.round(pts * 4).astype(np.int32)], col, cv2.LINE_AA, shift=2)
            if edge:
                cv2.polylines(canvas, [np.round(pts * 4).astype(np.int32)], True, outline, thick, cv2.LINE_AA, shift=2)
        elif kind == 2:  # house
            bw, bh = s, s * rng.uniform(0.7, 1.2)
            cv2.rectangle(canvas, (int(cx), int(cy)), (int(cx + bw), int(cy + bh)), col, -1)
            roof = tuple(int(v) for v in to_u8(pal[int(rng.integers(len(pal)))]))
            tri = np.array([[cx - bw * 0.1, cy], [cx + bw * 1.1, cy], [cx + bw * 0.5, cy - bh * 0.6]])
            cv2.fillPoly(canvas, [np.round(tri * 4).astype(np.int32)], roof, cv2.LINE_AA, shift=2)
            for k in range(int(rng.integers(1, 4))):
                wx = cx + bw * (0.15 + 0.3 * k)
                cv2.rectangle(canvas, (int(wx), int(cy + bh * 0.25)), (int(wx + bw * 0.18), int(cy + bh * 0.5)), (250, 240, 200), -1)
            if edge:
                cv2.rectangle(canvas, (int(cx), int(cy)), (int(cx + bw), int(cy + bh)), outline, thick)
        elif kind == 3:  # striped awning / flag
            bw, bh = s * 1.4, s * 0.6
            n_str = int(rng.integers(4, 9))
            c2 = tuple(int(v) for v in to_u8(pal[int(rng.integers(len(pal)))]))
            for k in range(n_str):
                x0 = cx + bw * k / n_str
                cv2.rectangle(canvas, (int(x0), int(cy)), (int(x0 + bw / n_str), int(cy + bh)), col if k % 2 else c2, -1)
        elif kind == 4:  # text-like sign
            bw, bh = s * rng.uniform(1.0, 2.2), s * rng.uniform(0.4, 0.9)
            cv2.rectangle(canvas, (int(cx), int(cy)), (int(cx + bw), int(cy + bh)), col, -1)
            ink = outline if np.mean(col) > 110 else (245, 245, 240)
            _text_marks(rng, canvas, cx + bw * 0.08, cy + bh * 0.12, bw * 0.84, bh * 0.76, ink)
        elif kind == 5:  # balloon
            cv2.ellipse(canvas, (int(cx), int(cy)), (max(1, int(s * 0.4)), max(1, int(s * 0.5))), 0, 0, 360, col, -1, cv2.LINE_AA)
            cv2.line(canvas, (int(cx), int(cy + s * 0.5)), (int(cx + s * 0.1), int(cy + s * 1.3)), outline, max(1, thick // 2), cv2.LINE_AA)
            cv2.ellipse(canvas, (int(cx - s * 0.13), int(cy - s * 0.18)), (max(1, int(s * 0.08)), max(1, int(s * 0.12))), 30, 0, 360, (255, 255, 255), -1, cv2.LINE_AA)
        elif kind == 6:  # polka-dot patch
            r = max(2, int(s * 0.08))
            for dy in range(0, int(s), max(3, r * 3)):
                for dx in range(0, int(s * 1.5), max(3, r * 3)):
                    cv2.circle(canvas, (int(cx + dx), int(cy + dy)), r, col, -1, cv2.LINE_AA)
        elif kind == 7:  # flower
            _flower(rng, canvas, cx, cy, s * 0.4, pal[int(rng.integers(len(pal)))])
        else:  # triangle / confetti
            a = rng.uniform(0, 6.28)
            tri = np.array([[cx + s * 0.5 * math.cos(a + k * 2.09), cy + s * 0.5 * math.sin(a + k * 2.09)] for k in range(3)])
            cv2.fillPoly(canvas, [np.round(tri * 4).astype(np.int32)], col, cv2.LINE_AA, shift=2)
            if edge:
                cv2.polylines(canvas, [np.round(tri * 4).astype(np.int32)], True, outline, thick, cv2.LINE_AA, shift=2)
    return canvas.astype(np.float32) / 255.0


# ---------- still life ----------


def _shaded_ellipse(canvas_f: np.ndarray, cx: float, cy: float, rx: float, ry: float, colour: np.ndarray,
                    light: tuple[float, float], gloss: float) -> None:
    """Draw a lit, round object (fruit) into a float canvas: radial shading plus a highlight."""
    h, w = canvas_f.shape[:2]
    x0, x1 = max(0, int(cx - rx - 2)), min(w, int(cx + rx + 3))
    y0, y1 = max(0, int(cy - ry - 2)), min(h, int(cy + ry + 3))
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    u = (xx - cx) / rx
    v = (yy - cy) / ry
    d = u * u + v * v
    mask = np.clip((1.0 - d) * min(rx, ry) * 0.5 + 0.5, 0, 1)
    nz = np.sqrt(np.clip(1 - d, 0, 1))
    lambert = np.clip(-u * light[0] - v * light[1] + nz * 0.6, 0, 1.3)
    col = colour[None, None, :] * (0.25 + 0.75 * lambert[..., None])
    hx, hy = cx - light[0] * rx * 0.45, cy - light[1] * ry * 0.45
    spec = np.exp(-(((xx - hx) / (rx * 0.18)) ** 2 + ((yy - hy) / (ry * 0.14)) ** 2)) * gloss
    col = col + spec[..., None] * (1 - col)
    region = canvas_f[y0:y1, x0:x1]
    canvas_f[y0:y1, x0:x1] = region * (1 - mask[..., None]) + col * mask[..., None]


def _stilllife(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    H, W = h, w
    h, w = H // 2, W // 2  # background, table and vase are smooth: build them at half size
    yy, xx = _yy_xx(h, w)
    dark_bg = rng.random() < 0.65
    if dark_bg:
        bg = hsv_to_rgb(rng.uniform(0.02, 0.4), rng.uniform(0.3, 0.8), rng.uniform(0.05, 0.22))
    else:
        bg = hsv_to_rgb(rng.uniform(0.05, 0.6), rng.uniform(0.1, 0.35), rng.uniform(0.6, 0.85))
    lx, ly = rng.uniform(0.1, 0.5) * w, rng.uniform(0.0, 0.3) * h
    light_fall = np.exp(-(((xx - lx) ** 2 + (yy - ly) ** 2) / (w * 0.8) ** 2))
    tex = fbm(rng, h, w, feature_px=w / 8, octaves=5, persistence=0.55)
    img = bg[None, None, :] * (0.6 + 0.6 * light_fall[..., None]) * (1 + 0.08 * tex[..., None])
    table_y = h * rng.uniform(0.55, 0.75)
    table = hsv_to_rgb(rng.uniform(0.04, 0.1), rng.uniform(0.3, 0.7), rng.uniform(0.25, 0.55)) if rng.random() < 0.6 \
        else hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.0, 0.2), rng.uniform(0.75, 0.95))
    folds = np.sin(xx / w * rng.uniform(8, 20) + 2 * fbm(rng, h, w, feature_px=w / 5, octaves=3)) * 0.08
    tmask = np.clip(yy - table_y + 0.5, 0, 1)
    tcol = table[None, None, :] * (0.8 + 0.3 * (yy - table_y)[..., None] / h + folds[..., None])
    img = img * (1 - tmask[..., None]) + tcol * tmask[..., None]
    light = (rng.uniform(-0.8, -0.2), rng.uniform(-0.8, -0.3))
    # Vase or jug: a revolved profile with cylinder shading.
    has_vase = rng.random() < 0.7
    if has_vase:
        vx = rng.uniform(0.25, 0.75) * w
        vh = h * rng.uniform(0.3, 0.5)
        vb = table_y + h * 0.03
        vw = vh * rng.uniform(0.2, 0.35)
        t = np.clip((vb - yy) / vh, 0, 1)
        prof = vw * (0.55 + 0.45 * np.clip(np.sin(np.pi * np.clip(t * 1.1, 0, 1)), 0, 1) ** 0.8 - 0.25 * smoothstep(0.75, 1.0, t))
        inside = (np.abs(xx - vx) < prof) & (yy <= vb) & (yy >= vb - vh)
        u = np.clip((xx - vx) / np.maximum(prof, 1), -1, 1)
        vcol = hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.3, 0.8), rng.uniform(0.4, 0.8))
        shade = 0.35 + 0.75 * np.clip(np.sqrt(1 - u * u) - 0.4 * u * light[0], 0, 1.2)
        vimg = vcol[None, None, :] * shade[..., None]
        m = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 0.8)
        img = img * (1 - m[..., None]) + vimg * m[..., None]
    img = _up(img, H, W)
    f = H / h
    table_y *= f
    if has_vase:
        vx, vh, vb, vw = vx * f, vh * f, vb * f, vw * f
        # Bouquet above the vase.
        canvas = to_u8(np.clip(img, 0, 1))
        pal = _bright_palette(rng, 5)
        for _ in range(int(rng.integers(20, 60))):
            fx = vx + rng.normal(0, vw * 1.2)
            fy = vb - vh - abs(rng.normal(0, vh * 0.35))
            leaf = tuple(int(v) for v in to_u8(hsv_to_rgb(rng.uniform(0.2, 0.35), 0.6, rng.uniform(0.2, 0.45))))
            cv2.ellipse(canvas, (int(fx), int(fy + vw * 0.3)), (max(1, int(vw * 0.25)), max(1, int(vw * 0.1))), rng.uniform(0, 180), 0, 360, leaf, -1, cv2.LINE_AA)
        for _ in range(int(rng.integers(8, 25))):
            fx = vx + rng.normal(0, vw * 1.1)
            fy = vb - vh - abs(rng.normal(0, vh * 0.3))
            col = tuple(int(v) for v in to_u8(pal[int(rng.integers(len(pal)))]))
            r = vw * rng.uniform(0.12, 0.25)
            for k in range(6):
                a = k * 1.047 + rng.uniform(0, 0.5)
                cv2.circle(canvas, (int((fx + math.cos(a) * r * 0.6) * 4), int((fy + math.sin(a) * r * 0.6) * 4)), max(1, int(r * 0.55 * 4)), col, -1, cv2.LINE_AA, shift=2)
            cv2.circle(canvas, (int(fx * 4), int(fy * 4)), max(1, int(r * 0.3 * 4)), (240, 200, 60), -1, cv2.LINE_AA, shift=2)
        img = canvas.astype(np.float32) / 255.0
    # Fruit on the table.
    fruit_cols = [(0.75, 0.1, 0.08), (0.95, 0.55, 0.1), (0.95, 0.85, 0.2), (0.45, 0.65, 0.15), (0.35, 0.1, 0.35),
                  (0.8, 0.3, 0.1)]
    for _ in range(int(rng.integers(4, 14))):
        r = W * rng.uniform(0.025, 0.06)
        fx = rng.uniform(0.1, 0.9) * W
        fy = table_y + rng.uniform(0.02, 0.2) * H
        c = np.array(fruit_cols[int(rng.integers(len(fruit_cols)))], np.float32) * rng.uniform(0.8, 1.1)
        if rng.random() < 0.2:  # grapes
            for _k in range(int(rng.integers(10, 25))):
                _shaded_ellipse(img, fx + rng.normal(0, r * 0.5), fy + rng.normal(0, r * 0.4), r * 0.28, r * 0.28, c, light, 0.6)
        else:
            _shaded_ellipse(img, fx, fy, r, r * rng.uniform(0.85, 1.0), c, light, rng.uniform(0.2, 0.7))
    return np.clip(img, 0, 1)


# ---------- garden ----------


def _garden(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    hl, wl = h // 2, w // 2
    f = hl / h
    sky_h = h * rng.uniform(0.0, 0.25)
    lo, hor = _sky(rng, hl, wl, max(sky_h * f, 1.0))
    if sky_h > 8:
        lo = _treeline(rng, lo, sky_h * f, depth=hl * rng.uniform(0.06, 0.15))
    start = int(sky_h + h * 0.05)
    n = h - start
    sl = int(start * f)
    leaf = hsv_to_rgb(rng.uniform(0.2, 0.33), rng.uniform(0.4, 0.8), rng.uniform(0.2, 0.4))
    tex = fbm(rng, hl - sl, wl, feature_px=max(3.0, wl / 400), octaves=4, persistence=0.7)
    lo[sl:] = leaf[None, None, :] * (1 + 0.3 * tex[..., None])
    canvas = to_u8(_up(lo, h, w))
    # Beds: soft regions each dominated by one flower colour.
    n_beds = int(rng.integers(3, 8))
    pal = _bright_palette(rng, n_beds)
    bed_h = n / n_beds
    for i in range(n_beds):
        y0 = int(start + i * bed_h)
        y1 = int(min(h, y0 + bed_h * 1.2))
        sub = [pal[i], pal[(i + 1) % n_beds], np.array([0.97, 0.97, 0.95], np.float32)]
        t = (y0 - start) / max(1, n)
        _flowers(rng, canvas, y0, y1, density=rng.uniform(2.0, 5.0), palette=[sub[0], sub[0], sub[1], sub[2]],
                 size_px=(w / 400 * (0.6 + 1.2 * t), w / 160 * (0.6 + 1.2 * t)))
    if rng.random() < 0.5:  # gravel path through the beds
        path = np.zeros((h, w), np.float32)
        x_mid = w * rng.uniform(0.3, 0.7)
        pts = [(x_mid + w * 0.02 * math.sin(k), start + k * n / 8) for k in range(9)]
        for k in range(8):
            wid = w * (0.02 + 0.08 * k / 8)
            quad = np.array([[pts[k][0] - wid, pts[k][1]], [pts[k][0] + wid, pts[k][1]],
                             [pts[k + 1][0] + wid * 1.1, pts[k + 1][1]], [pts[k + 1][0] - wid * 1.1, pts[k + 1][1]]])
            cv2.fillPoly(path, [np.round(quad).astype(np.int32)], 1.0, cv2.LINE_AA)
        gravel = hsv_to_rgb(rng.uniform(0.07, 0.12), rng.uniform(0.1, 0.3), rng.uniform(0.6, 0.8))
        g = fbm(rng, h, w, feature_px=3.0, octaves=2)
        pimg = gravel[None, None, :] * (1 + 0.12 * g[..., None])
        f = canvas.astype(np.float32) / 255.0
        f = f * (1 - path[..., None]) + pimg * path[..., None]
        return np.clip(f, 0, 1)
    return canvas.astype(np.float32) / 255.0
