"""Draws a dataset's ground truth over its images, so a human can check it at a glance.

    python -m tools.synth.preview datasets/<name> [--max-scenes 48]

Why: ground truth that is wrong by a rotation, a mirrored corner order or half a cell looks fine
in JSON and ruins every metric computed on it. These overlays make such errors obvious:
- scenes: outline, corners numbered 0-3 in MOTIF order (0 = TL ... 3 = BL of the piece as it
  sits in the puzzle), an arrow along upAngleDeg from the core centre, a "c12 r5" label, plus the
  instance mask in false colour next to the frame;
- references: referenceCorners and the cols x rows grid warped into the photo;
- cuts: every piece outline drawn over motif.png.
Output goes to <dataset>/_preview/: contact sheets (scenes_NN.png, references.png, cuts.png) and
full-resolution overlays in _preview/scenes/ for zooming in.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .common import apply_h, read_gray, read_rgb, write_png

CORNER_COLOURS = [(255, 60, 60), (60, 220, 60), (60, 120, 255), (255, 220, 0)]  # TL, TR, BR, BL


def _colour(k: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(k * 7919 + 13)
    c = rng.integers(60, 256, 3)
    c[int(rng.integers(3))] = 255
    return int(c[0]), int(c[1]), int(c[2])


def draw_scene(img: np.ndarray, scene: dict) -> np.ndarray:
    out = img.copy()
    for p in scene["pieces"]:
        o = np.array(p["outline"])
        c = np.array(p["corners"])
        col = _colour(p["index"]) if p["faceUp"] else (170, 170, 170)
        cv2.polylines(out, [np.round(o * 8).astype(np.int32)], True, col, 2, cv2.LINE_AA, shift=3)
        side = float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))
        centre = c.mean(axis=0)
        for i, q in enumerate(c):
            cv2.circle(out, (int(round(q[0] * 8)), int(round(q[1] * 8))), 5 * 8, CORNER_COLOURS[i], -1, cv2.LINE_AA, shift=3)
            d = centre - q
            t = q + d / (np.linalg.norm(d) + 1e-6) * min(22, side * 0.2)
            cv2.putText(out, str(i), (int(t[0] - 5), int(t[1] + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, str(i), (int(t[0] - 5), int(t[1] + 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORNER_COLOURS[i], 1, cv2.LINE_AA)
        a = math.radians(p["upAngleDeg"])
        tip = centre + np.array([math.sin(a), -math.cos(a)]) * side * 0.4
        cv2.arrowedLine(out, tuple(int(v) for v in centre), tuple(int(v) for v in tip), (0, 0, 0), 4, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(out, tuple(int(v) for v in centre), tuple(int(v) for v in tip), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
        label = f"c{p['col']} r{p['row']}"
        if not p["faceUp"]:
            label += " down"
        if p["visibleFraction"] < 0.999:
            label += f" {p['visibleFraction']:.2f}"
        org = (int(centre[0] - 4 * len(label)), int(centre[1] + side * 0.12 + 14))
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def colour_mask(mask: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 3), np.uint8)
    for k in range(1, 256):
        lut[k] = _colour(k)
    return lut[mask]


def draw_reference(img: np.ndarray, ref: dict) -> np.ndarray:
    out = img.copy()
    W, H = ref["motifSize"]
    src = np.array([[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H - 0.5], [-0.5, H - 0.5]], np.float64)
    dst = np.array(ref["referenceCorners"], np.float64)
    Hm = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
    t = max(1, int(round(max(img.shape[:2]) / 900)))
    for c in range(ref["cols"] + 1):
        x = -0.5 + c * W / ref["cols"]
        p = apply_h(Hm, np.array([[x, -0.5], [x, H - 0.5]]))
        cv2.line(out, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), (0, 255, 255), t, cv2.LINE_AA)
    for r in range(ref["rows"] + 1):
        y = -0.5 + r * H / ref["rows"]
        p = apply_h(Hm, np.array([[-0.5, y], [W - 0.5, y]]))
        cv2.line(out, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), (0, 255, 255), t, cv2.LINE_AA)
    cv2.polylines(out, [np.round(dst).astype(np.int32)], True, (255, 0, 255), 2 * t, cv2.LINE_AA)
    for i, q in enumerate(dst):
        cv2.circle(out, (int(q[0]), int(q[1])), 8 * t, CORNER_COLOURS[i], -1, cv2.LINE_AA)
        cv2.putText(out, ["TL", "TR", "BR", "BL"][i], (int(q[0]) + 10 * t, int(q[1]) - 10 * t), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 * t, CORNER_COLOURS[i], 2 * t, cv2.LINE_AA)
    return out


def draw_cut(motif: np.ndarray, pieces: list[dict]) -> np.ndarray:
    out = motif.copy()
    t = max(1, int(round(max(motif.shape[:2]) / 1500)))
    for p in pieces:
        cv2.polylines(out, [np.round(np.array(p["outline"]) * 8).astype(np.int32)], True, (255, 255, 255), t, cv2.LINE_AA, shift=3)
    return out


def _fit(img: np.ndarray, h: int) -> np.ndarray:
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(round(img.shape[1] * s))), h), interpolation=cv2.INTER_AREA)


def _sheet(tiles: list[np.ndarray], per_row: int, gap: int = 8) -> np.ndarray:
    rows = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        h = max(t.shape[0] for t in row)
        parts = []
        for t in row:
            pad = np.full((h, t.shape[1] + gap, 3), 30, np.uint8)
            pad[:t.shape[0], :t.shape[1]] = t
            parts.append(pad)
        rows.append(np.concatenate(parts, axis=1))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, gap), (0, width - r.shape[1]), (0, 0)), constant_values=30) for r in rows]
    return np.concatenate(rows, axis=0)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.synth.preview")
    ap.add_argument("dataset")
    ap.add_argument("--max-scenes", type=int, default=48)
    args = ap.parse_args(argv)
    root = Path(args.dataset)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    prev = root / "_preview"
    (prev / "scenes").mkdir(parents=True, exist_ok=True)

    tiles = []
    for sid in meta["scenes"][:args.max_scenes]:
        img = read_rgb(root / "scenes" / f"{sid}.jpg")
        scene = json.loads((root / "scenes" / f"{sid}.json").read_text(encoding="utf-8"))
        mask = read_gray(root / "scenes" / f"{sid}_mask.png")
        ov = draw_scene(img, scene)
        cv2.imwrite(str(prev / "scenes" / f"{sid}.jpg"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        pair = np.concatenate([_fit(ov, 720), _fit(colour_mask(mask), 720)], axis=1)
        cv2.putText(pair, f"{sid} {scene['background']} tilt {scene['render']['tiltDeg']:.0f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        tiles.append(pair)
    per_sheet = 6
    for n, i in enumerate(range(0, len(tiles), per_sheet), start=1):
        write_png(prev / f"scenes_{n:02d}.png", _sheet(tiles[i:i + per_sheet], per_row=3))

    ref_tiles, cut_tiles = [], []
    for pid in meta["puzzles"]:
        pdir = root / "puzzles" / pid
        ref = json.loads((pdir / "reference.json").read_text(encoding="utf-8"))
        img = draw_reference(read_rgb(pdir / "reference.jpg"), ref)
        cv2.imwrite(str(prev / f"reference_{pid}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        ref_tiles.append(_fit(img, 700))
        pieces = json.loads((pdir / "pieces.json").read_text(encoding="utf-8"))
        cut = draw_cut(read_rgb(pdir / "motif.png"), pieces)
        cv2.imwrite(str(prev / f"cut_{pid}.jpg"), cv2.cvtColor(cut, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        cut_tiles.append(_fit(cut, 700))
    write_png(prev / "references.png", _sheet(ref_tiles, per_row=3))
    write_png(prev / "cuts.png", _sheet(cut_tiles, per_row=3))
    print(f"wrote previews to {prev}")


if __name__ == "__main__":
    main()
