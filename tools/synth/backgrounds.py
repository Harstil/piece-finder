"""Table surfaces the pieces lie on.

Why: the user must be able to scan pieces on any table, and piece segmentation will be a small
network trained on these frames. So the backgrounds span what real tables look like (plain
surfaces and puzzle mats, wood grain of several species, woven cloth, printed tablecloths,
cluttered desks) and include deliberately hard cases: surfaces coloured like the motif itself and
busy high-contrast patterns that look like piece content.

A background is plain albedo in *table-plane* coordinates; scene.py warps it with the camera
homography and applies lighting, so perspective and illumination match the pieces exactly.
Texture photos in tools/synth/sources/textures (CC0, see fetch_sources.py) are mixed in when
present. Face-down pieces on cluttered tables are not drawn here: scene.py adds them as labelled
distractors so every piece-shaped object in a frame is in the instance mask.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .common import choice_weighted, fbm, hsv_to_rgb, lerp, read_rgb, smoothstep, to_u8
from .motifs import SOURCES_DIR, _draw_flowers

TEXTURE_DIR = SOURCES_DIR / "textures"

KINDS = ("plain", "wood", "cloth", "pattern", "clutter")
# Guessed mix: wood and plain tables dominate real puzzling; cloth/pattern/clutter keep the
# segmenter honest.
KIND_WEIGHTS = (0.22, 0.26, 0.17, 0.2, 0.15)
# Probability that a background borrows the motif's own colours (the hardest case). Guessed.
P_MOTIF_COLOURS = 0.25
# Probability that a downloaded texture replaces a procedural plain/wood/cloth surface. Guessed.
P_TEXTURE_FILE = 0.5

_WOOD_SPECIES = {
    "oak": (0.70, 0.53, 0.34), "walnut": (0.36, 0.24, 0.15), "pine": (0.85, 0.70, 0.46),
    "cherry": (0.60, 0.33, 0.20), "greywash": (0.62, 0.60, 0.56), "whitewash": (0.86, 0.83, 0.77),
    "mahogany": (0.45, 0.21, 0.13), "ebony": (0.18, 0.14, 0.12), "beech": (0.80, 0.62, 0.45),
}
_PLAIN_TABLES = [
    (0.92, 0.92, 0.9), (0.97, 0.97, 0.96), (0.75, 0.75, 0.74), (0.45, 0.45, 0.46), (0.12, 0.12, 0.13),
    (0.82, 0.76, 0.64), (0.18, 0.38, 0.22), (0.16, 0.26, 0.45), (0.55, 0.62, 0.55), (0.72, 0.8, 0.86),
]


def make_background(rng: np.random.Generator, h: int, w: int, motif: np.ndarray | None = None,
                    kind: str | None = None) -> tuple[np.ndarray, str]:
    """(float32 RGB image h x w in 0..1, label) where label is a DATASET.md background name."""
    if kind is None:
        kind = choice_weighted(rng, KINDS, KIND_WEIGHTS)
    palette = _motif_palette(rng, motif) if motif is not None and rng.random() < P_MOTIF_COLOURS else None
    textures = texture_files()
    if kind in ("plain", "wood", "cloth") and textures and rng.random() < P_TEXTURE_FILE:
        path, label = textures[int(rng.integers(len(textures)))]
        return _texture_file(rng, h, w, path), label
    if kind == "plain":
        img = _plain(rng, h, w, palette)
    elif kind == "wood":
        img = _wood(rng, h, w)
    elif kind == "cloth":
        img = _cloth(rng, h, w, palette)
    elif kind == "pattern":
        img = _pattern(rng, h, w, palette)
    else:
        img = _clutter(rng, h, w, palette)
    return np.clip(img, 0, 1).astype(np.float32), kind


def _motif_palette(rng: np.random.Generator, motif: np.ndarray) -> list[np.ndarray]:
    """A few colours sampled from a blurred motif: backgrounds that look like the puzzle itself."""
    small = cv2.resize(motif, (64, 48), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32) / 255.0
    idx = rng.choice(len(small), size=5, replace=False)
    return [small[i] for i in idx]


# ----------------------------------------------------------------------------------------------
# Texture files
# ----------------------------------------------------------------------------------------------


def texture_files() -> list[tuple[Path, str]]:
    """(path, background label) for downloaded textures, sorted for determinism."""
    if not TEXTURE_DIR.is_dir():
        return []
    cats: dict[str, str] = {}
    mf = SOURCES_DIR / "manifest.json"
    if mf.is_file():
        for item in json.loads(mf.read_text(encoding="utf-8")).get("textures", []):
            cats[item["file"]] = item.get("background", "plain")
    files = sorted(p for p in TEXTURE_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return [(p, cats.get(p.name, "plain")) for p in files]


def _texture_file(rng: np.random.Generator, h: int, w: int, path: Path) -> np.ndarray:
    tex = read_rgb(path).astype(np.float32) / 255.0
    th, tw = tex.shape[:2]
    # The texture spans 0.6-1.6x the canvas's longer side (physical scale of the photo unknown).
    s = max(h, w) * rng.uniform(0.6, 1.6) / max(th, tw)
    ang = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((tw / 2, th / 2), ang, s)
    M[:, 2] += (w / 2 - tw / 2, h / 2 - th / 2)
    img = cv2.warpAffine(tex, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gain = np.array(rng.uniform(0.85, 1.15, 3), np.float32) * rng.uniform(0.8, 1.15)
    return img * gain


# ----------------------------------------------------------------------------------------------
# Procedural surfaces
# ----------------------------------------------------------------------------------------------


def _pick(rng: np.random.Generator, palette: list[np.ndarray] | None, fallback: list) -> np.ndarray:
    if palette:
        return np.asarray(palette[int(rng.integers(len(palette)))], np.float32)
    return np.asarray(fallback[int(rng.integers(len(fallback)))], np.float32) * rng.uniform(0.9, 1.08)


def _plain(rng: np.random.Generator, h: int, w: int, palette) -> np.ndarray:
    """Laminate, painted wood, felt puzzle mat: one colour, low-frequency blotches, fine fuzz."""
    col = _pick(rng, palette, _PLAIN_TABLES)
    low = fbm(rng, h, w, feature_px=max(h, w) / rng.uniform(1.5, 4), octaves=3)
    fuzz = fbm(rng, h, w, feature_px=rng.uniform(1.0, 3.0), octaves=2)
    felt = rng.random() < 0.4
    k_fuzz = rng.uniform(0.03, 0.07) if felt else rng.uniform(0.005, 0.02)
    shade = 1.0 + rng.uniform(0.02, 0.06) * low + k_fuzz * fuzz
    img = col[None, None, :] * shade[..., None]
    if rng.random() < 0.3:  # specks and dust
        n = int(h * w / rng.uniform(3000, 20000))
        canvas = to_u8(np.clip(img, 0, 1))
        for _ in range(n):
            c = tuple(int(v) for v in to_u8(np.clip(col * rng.uniform(0.6, 1.3), 0, 1)))
            cv2.circle(canvas, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.integers(1, 3)), c, -1, cv2.LINE_AA)
        img = canvas.astype(np.float32) / 255.0
    return img


def _wood(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Planks with growth rings, fine streaks, pores and seams, at any grain angle."""
    name = list(_WOOD_SPECIES)[int(rng.integers(len(_WOOD_SPECIES)))]
    base = np.array(_WOOD_SPECIES[name], np.float32) * rng.uniform(0.9, 1.1)
    # Build along +x on a half-resolution square that covers the canvas at any rotation.
    d = int(math.ceil(math.hypot(h, w) / 2)) + 4
    yy, xx = np.mgrid[0:d, 0:d].astype(np.float32)
    plank_w = d * rng.uniform(0.1, 0.3)
    n_planks = int(d / plank_w) + 2
    plank_idx = np.floor((yy + rng.uniform(0, plank_w)) / plank_w).astype(np.int32)
    tone = rng.uniform(0.86, 1.14, n_planks + 2).astype(np.float32)[np.clip(plank_idx, 0, n_planks + 1)]
    period = rng.uniform(4.0, 12.0)
    warp = fbm(rng, d, d, feature_px=d / 4, octaves=4, aspect=rng.uniform(6, 14)) * rng.uniform(3, 8) * period
    offs = rng.uniform(0, 1000, n_planks + 2).astype(np.float32)[np.clip(plank_idx, 0, n_planks + 1)]
    phase = (yy + warp + offs) / period
    rings = smoothstep(0.55, 0.95, 0.5 + 0.5 * np.sin(2 * np.pi * phase)) ** rng.uniform(0.7, 2.0)
    streak = fbm(rng, d, d, feature_px=rng.uniform(1.0, 2.5), octaves=3, aspect=rng.uniform(25, 60))
    low = fbm(rng, d, d, feature_px=d / 3, octaves=3, aspect=4.0)
    shade = tone * (1.0 - rng.uniform(0.12, 0.3) * rings + rng.uniform(0.025, 0.06) * streak + 0.05 * low)
    seam_y = (yy + rng.uniform(0, plank_w)) % plank_w
    seam = np.clip(1.2 - np.minimum(seam_y, plank_w - seam_y), 0, 1)
    shade = shade * (1 - rng.uniform(0.3, 0.6) * seam)
    lo = base[None, None, :] * shade[..., None]
    ang = rng.choice([0.0, 90.0]) + rng.uniform(-25, 25) if rng.random() < 0.7 else rng.uniform(0, 180)
    M = cv2.getRotationMatrix2D((d / 2, d / 2), ang, 2.0)
    M[:, 2] += (w / 2 - d / 2, h / 2 - d / 2)
    return cv2.warpAffine(lo, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _weave(rng: np.random.Generator, h: int, w: int, period: float, angle: float) -> np.ndarray:
    """Plain-weave thread relief in 0..1 (over/under checker of rounded threads)."""
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    u = (xs * c + ys * s) / period
    v = (-xs * s + ys * c) / period
    iu, iv = np.floor(u), np.floor(v)
    lut = (np.sin(np.pi * (np.arange(256) + 0.5) / 256) ** 0.6).astype(np.float32)  # rounded thread profile
    across_u = lut[((u - iu) * 255.999).astype(np.int32)]
    across_v = lut[((v - iv) * 255.999).astype(np.int32)]
    over = ((iu + iv) % 2) == 0
    return np.where(over, across_v, across_u)


def _folds(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Soft cloth wrinkles as a shading field around 1.0."""
    f = fbm(rng, h // 4 + 1, w // 4 + 1, feature_px=max(h, w) / 4 / rng.uniform(2, 6), octaves=3,
            aspect=rng.uniform(0.3, 3.0))
    gy, gx = np.gradient(f)
    light = cv2.resize((gx * 0.7 + gy * 0.7).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return 1.0 + rng.uniform(0.5, 2.0) * light


def _cloth(rng: np.random.Generator, h: int, w: int, palette) -> np.ndarray:
    """Woven tablecloth or linen: thread relief, slubs, wrinkles; solid or yarn-dyed."""
    col = _pick(rng, palette, [(0.9, 0.88, 0.82), (0.7, 0.15, 0.15), (0.2, 0.3, 0.55), (0.35, 0.5, 0.3),
                               (0.85, 0.75, 0.55), (0.6, 0.6, 0.62), (0.95, 0.95, 0.95), (0.4, 0.25, 0.35)])
    period = rng.uniform(2.2, 6.0)
    ang = rng.uniform(0, 180)
    weave = _weave(rng, h, w, period, ang)
    slub = fbm(rng, h, w, feature_px=rng.uniform(3, 8), octaves=2, aspect=rng.uniform(4, 10))
    img = col[None, None, :] * (0.82 + 0.22 * weave + 0.05 * slub)[..., None]
    if rng.random() < 0.35:  # gingham / check
        img = img * _check(rng, h, w, ang)[..., None]
    return img * _folds(rng, h, w)[..., None]


def _check(rng: np.random.Generator, h: int, w: int, ang: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    p = rng.uniform(15, 60)
    u = ((xx * c + yy * s) / p) % 1.0 < 0.5
    v = ((-xx * s + yy * c) / p) % 1.0 < 0.5
    return 1.0 - rng.uniform(0.2, 0.5) * (u.astype(np.float32) + v.astype(np.float32)) / 2


def _pattern(rng: np.random.Generator, h: int, w: int, palette) -> np.ndarray:
    """Printed tablecloth / oilcloth: plaid, stripes, florals or geometric, often high contrast."""
    style = rng.choice(["plaid", "stripes", "floral", "geometric", "dots"])
    if palette:
        cols = [np.asarray(c, np.float32) for c in palette]
    else:
        hue = rng.uniform(0, 1)
        cols = [hsv_to_rgb(hue + rng.uniform(-0.5, 0.5), rng.uniform(0.2, 0.9), rng.uniform(0.2, 0.95)) for _ in range(5)]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = math.radians(rng.uniform(0, 180))
    u = xx * math.cos(ang) + yy * math.sin(ang)
    v = -xx * math.sin(ang) + yy * math.cos(ang)
    if style == "plaid":
        img = np.zeros((h, w, 3), np.float32) + cols[0]
        for k in range(1, len(cols)):
            p = rng.uniform(40, 160)
            wd = rng.uniform(0.08, 0.35)
            a = rng.uniform(0.25, 0.6)
            bu = (((u / p + rng.uniform(0, 1)) % 1.0) < wd).astype(np.float32)
            bv = (((v / p + rng.uniform(0, 1)) % 1.0) < wd).astype(np.float32)
            m = np.clip(bu + bv, 0, 1) * a
            img = img * (1 - m[..., None]) + cols[k] * m[..., None]
    elif style == "stripes":
        p = rng.uniform(12, 90)
        idx = np.floor(u / p).astype(np.int64) % len(cols)
        img = np.stack(cols)[idx]
    elif style == "floral":
        img = np.zeros((h, w, 3), np.float32) + cols[0]
        canvas = to_u8(img)
        step = rng.uniform(25, 70)
        gy, gx = np.mgrid[0:int(h / step) + 2, 0:int(w / step) + 2]
        xs = (gx.ravel() + (gy.ravel() % 2) * 0.5) * step + rng.normal(0, step * 0.15, gx.size)
        ys = gy.ravel() * step + rng.normal(0, step * 0.15, gx.size)
        rs = step * rng.uniform(0.2, 0.4, gx.size)
        cc = np.stack([cols[1 + int(i)] for i in rng.integers(0, len(cols) - 1, gx.size)])
        _draw_flowers(rng, canvas, xs, ys, rs, cc)
        img = canvas.astype(np.float32) / 255.0
    elif style == "geometric":
        p = rng.uniform(20, 80)
        kind = rng.integers(3)
        if kind == 0:  # checkerboard
            idx = (np.floor(u / p) + np.floor(v / p)).astype(np.int64) % 2
        elif kind == 1:  # triangles
            fu, fv = (u / p) % 1.0, (v / p) % 1.0
            idx = ((fu > fv).astype(np.int64) + np.floor(u / p).astype(np.int64)) % 3
        else:  # diamonds
            idx = (np.floor((u + v) / p) + np.floor((u - v) / p)).astype(np.int64) % 3
        img = np.stack(cols)[idx % len(cols)]
    else:  # dots
        p = rng.uniform(14, 50)
        fu, fv = (u / p) % 1.0 - 0.5, (v / p) % 1.0 - 0.5
        r = rng.uniform(0.15, 0.4)
        m = np.clip((r - np.sqrt(fu * fu + fv * fv)) * p + 0.5, 0, 1)
        img = cols[0] * (1 - m[..., None]) + cols[1] * m[..., None]
    weave = _weave(rng, h, w, rng.uniform(2.5, 5.0), rng.uniform(0, 180))
    img = img * (0.9 + 0.12 * weave)[..., None]
    if rng.random() < 0.5:
        img = img * _folds(rng, h, w)[..., None]
    return img.astype(np.float32)


def _clutter(rng: np.random.Generator, h: int, w: int, palette) -> np.ndarray:
    """A desk: plain or wood surface with paper, notes, a mug, pens, coins and crumbs."""
    base = _wood(rng, h, w) if rng.random() < 0.5 else _plain(rng, h, w, palette)
    canvas = to_u8(np.clip(base, 0, 1))
    shadow = np.zeros((h, w), np.float32)
    scale = min(h, w)
    n = int(rng.integers(3, 10))
    for _ in range(n):
        kind = rng.choice(["paper", "note", "mug", "pen", "coin", "lid", "phone"])
        cx, cy = rng.uniform(-0.1, 1.1) * w, rng.uniform(-0.1, 1.1) * h
        ang = rng.uniform(0, 180)
        if kind in ("paper", "note", "lid", "phone"):
            if kind == "paper":
                sw, sh = scale * rng.uniform(0.35, 0.7), scale * rng.uniform(0.45, 0.9)
                col = (rng.uniform(0.9, 1.0), rng.uniform(0.9, 1.0), rng.uniform(0.85, 0.97))
            elif kind == "note":
                sw = sh = scale * rng.uniform(0.1, 0.18)
                col = [(1.0, 0.92, 0.45), (1.0, 0.7, 0.75), (0.6, 0.9, 1.0), (0.7, 1.0, 0.6)][int(rng.integers(4))]
            elif kind == "lid":
                sw, sh = scale * rng.uniform(0.5, 0.9), scale * rng.uniform(0.4, 0.7)
                col = tuple(hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.4, 0.9), rng.uniform(0.3, 0.9)))
            else:
                sw, sh = scale * 0.16, scale * 0.33
                col = (0.08, 0.08, 0.1)
            box = cv2.boxPoints(((cx, cy), (sw, sh), ang))
            cv2.fillPoly(shadow, [np.round(box + (scale * 0.01, scale * 0.012)).astype(np.int32)], 1.0, cv2.LINE_AA)
            c8 = tuple(int(v) for v in to_u8(np.clip(np.array(col), 0, 1)))
            cv2.fillPoly(canvas, [np.round(box * 4).astype(np.int32)], c8, cv2.LINE_AA, shift=2)
            if kind == "paper":  # lines of text
                R = cv2.getRotationMatrix2D((0, 0), -ang, 1.0)[:, :2]
                for k in range(int(sh / (scale * 0.03))):
                    y = -sh / 2 + scale * 0.05 + k * scale * 0.03
                    if y > sh / 2 - scale * 0.04:
                        break
                    x1 = sw / 2 - scale * 0.04 - rng.uniform(0, sw * 0.4)
                    p = np.array([[-sw / 2 + scale * 0.04, y], [x1, y]]) @ R.T + (cx, cy)
                    cv2.line(canvas, tuple(int(q) for q in p[0]), tuple(int(q) for q in p[1]), (70, 70, 80),
                             max(1, int(scale * 0.004)), cv2.LINE_AA)
            if kind == "lid":  # printed band on a puzzle box lid
                inner = cv2.boxPoints(((cx, cy), (sw * 0.8, sh * 0.7), ang))
                ic = tuple(int(v) for v in to_u8(hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.2, 0.6), rng.uniform(0.5, 0.95))))
                cv2.fillPoly(canvas, [np.round(inner * 4).astype(np.int32)], ic, cv2.LINE_AA, shift=2)
        elif kind == "mug":
            r = scale * rng.uniform(0.07, 0.12)
            cv2.circle(shadow, (int(cx + r * 0.15), int(cy + r * 0.2)), int(r * 1.05), 1.0, -1, cv2.LINE_AA)
            mc = tuple(int(v) for v in to_u8(hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0, 0.7), rng.uniform(0.4, 0.95))))
            cv2.circle(canvas, (int(cx), int(cy)), int(r), mc, -1, cv2.LINE_AA)
            cv2.circle(canvas, (int(cx), int(cy)), int(r * 0.85), (60, 38, 22), -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (int(cx + r * 1.1), int(cy)), (int(r * 0.35), int(r * 0.15)), ang, 0, 360, mc, -1, cv2.LINE_AA)
        elif kind == "pen":
            L, t = scale * rng.uniform(0.25, 0.4), max(3, int(scale * 0.012))
            dx, dy = math.cos(math.radians(ang)) * L / 2, math.sin(math.radians(ang)) * L / 2
            pc = tuple(int(v) for v in to_u8(hsv_to_rgb(rng.uniform(0, 1), rng.uniform(0.3, 0.9), rng.uniform(0.2, 0.9))))
            cv2.line(shadow, (int(cx - dx + 3), int(cy - dy + 4)), (int(cx + dx + 3), int(cy + dy + 4)), 1.0, t, cv2.LINE_AA)
            cv2.line(canvas, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), pc, t, cv2.LINE_AA)
        else:  # coins
            for _k in range(int(rng.integers(1, 5))):
                r = scale * rng.uniform(0.015, 0.025)
                x, y = cx + rng.normal(0, r * 3), cy + rng.normal(0, r * 3)
                g = rng.uniform(0.55, 0.85)
                cc = tuple(int(v) for v in to_u8(np.array([g, g * rng.uniform(0.75, 0.95), g * rng.uniform(0.4, 0.8)])))
                cv2.circle(canvas, (int(x), int(y)), int(r), cc, -1, cv2.LINE_AA)
                cv2.circle(canvas, (int(x), int(y)), int(r * 0.8), tuple(int(v * 0.85) for v in cc), 1, cv2.LINE_AA)
    img = canvas.astype(np.float32) / 255.0
    shadow = cv2.GaussianBlur(shadow, (0, 0), scale * 0.008)
    img = img * (1 - 0.3 * shadow[..., None])
    n_crumbs = int(rng.integers(0, 60))
    for _ in range(n_crumbs):
        x, y = int(rng.uniform(0, w)), int(rng.uniform(0, h))
        cv2.circle(img, (x, y), int(rng.integers(1, 3)), tuple(float(v) for v in lerp(np.array([0.6, 0.45, 0.3]), np.array([0.9, 0.85, 0.7]), rng.random())), -1)
    return img
