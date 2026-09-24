"""Invariant tests for the synthetic data generator (plain asserts, no pytest needed).

    python -m tools.synth.test_synth                    # cut tests + a fresh mini dataset, twice
    python -m tools.synth.test_synth datasets/smoke     # ...and also check existing datasets

Why these checks: the dataset is the ruler every engine decision is measured with. A ruler that
is off by a rotation, a mirrored corner order, a half-pixel shift or a swapped mask index would
silently bias every metric. So each contract clause of docs/DATASET.md is checked here, including
photometric ones that compare the rendered pixels with motif.png through the ground truth.

The checkers are themselves tested: each is fed deliberately corrupted data and must fail
(a checker that has never failed has proven nothing).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from .common import read_gray, read_rgb, rng_for, signed_area, up_angle_deg
from .cut import Cut, Piece, choose_grid, make_cut, round_cut
from .make_dataset import main as make_dataset_main

# Thresholds. Measured on generated data unless noted.
TAB_MIN_BULGE = 0.10        # tab/blank must bulge >= 10 % of the chord (tabs are ~0.26)
FLAT_MAX_BULGE = 0.03       # flat sides stay within 3 % (edge waviness is <= ~1.2 %)
MASK_TOP_RECALL_MIN = 0.97  # share of an unoccluded top face labelled with its index (measured >= 0.985;
                            # the check raster includes boundary pixels the renderer covers < 50 %)
WALL_MAX_FRACTION = 0.16    # mask may extend beyond the outline only by the side wall: board 0.12 x core
                            # seen up to ~50 deg off vertical (measured max 0.087 in smoke1000)
ZNCC_ROT_WIN_MIN = 0.9      # share of textured pieces whose true rotation beats the other three
ZNCC_TRUE_MEDIAN_MIN = 0.5  # median ZNCC of frame vs motif through the ground-truth homography
REF_ZNCC_MIN = 0.35         # reference photo vs motif through referenceCorners (badges, glare, crop)


# ----------------------------------------------------------------------------------------------
# Cut checks
# ----------------------------------------------------------------------------------------------


def corner_indices(outline: np.ndarray, corners: np.ndarray, tol: float = 1e-6) -> list[int]:
    idx = []
    for c in corners:
        hit = np.where(np.all(np.abs(outline - c) <= tol, axis=1))[0]
        assert len(hit) == 1, f"corner {c} is not exactly one outline vertex ({len(hit)} hits)"
        idx.append(int(hit[0]))
    return idx


def side_segments(outline: np.ndarray, corners: np.ndarray) -> list[np.ndarray]:
    """The 4 outline stretches corner i -> corner i+1 (both corners included)."""
    idx = corner_indices(outline, corners)
    n = len(outline)
    segs = []
    for i in range(4):
        a, b = idx[i], idx[(i + 1) % 4]
        seg = outline[a:b + 1] if b > a else np.concatenate([outline[a:], outline[:b + 1]])
        segs.append(seg)
    total = sum(len(s) - 1 for s in segs)
    assert total == n, "corners are not in TL, TR, BR, BL order along the outline"
    return segs


def bulge(seg: np.ndarray) -> float:
    """Signed max distance from the chord, outward (> 0) for a clockwise outline, / chord length."""
    a, b = seg[0], seg[-1]
    d = b - a
    L = float(np.linalg.norm(d))
    out = np.array([d[1], -d[0]]) / L
    dist = (seg - a) @ out
    m = dist[np.argmax(np.abs(dist))]
    return float(m) / L


def check_cut(cut: Cut) -> None:
    by_cell = {(p.row, p.col): p for p in cut.pieces}
    assert len(by_cell) == cut.cols * cut.rows
    total = 0.0
    segs: dict[int, list[np.ndarray]] = {}
    for p in cut.pieces:
        o = p.outline
        assert p.id == p.row * cut.cols + p.col == p.cell
        assert o.ndim == 2 and o.shape[1] == 2 and len(o) >= 200, f"piece {p.id}: {len(o)} outline points"
        assert np.linalg.norm(o[0] - o[-1]) > 1e-9, f"piece {p.id}: outline repeats its first point"
        assert np.all(np.linalg.norm(np.diff(o, axis=0), axis=1) > 1e-9), f"piece {p.id}: duplicate vertices"
        a = signed_area(o)
        assert a > 0, f"piece {p.id}: outline is not clockwise"
        total += a
        s = side_segments(o, p.corners)
        segs[p.id] = s
        border = [p.row == 0, p.col == cut.cols - 1, p.row == cut.rows - 1, p.col == 0]
        for i in range(4):
            assert (p.sides[i] == "flat") == border[i], f"piece {p.id} side {i}: {p.sides[i]} vs border {border[i]}"
            bl = bulge(s[i])
            kind = p.sides[i]
            ok = (abs(bl) <= FLAT_MAX_BULGE) if kind == "flat" else (bl >= TAB_MIN_BULGE if kind == "tab" else bl <= -TAB_MIN_BULGE)
            assert ok, f"piece {p.id} side {i} is {kind} but bulges {bl:+.3f}"
    for p in cut.pieces:
        if p.col + 1 < cut.cols:
            q = by_cell[(p.row, p.col + 1)]
            assert {p.sides[1], q.sides[3]} == {"tab", "blank"}, f"pieces {p.id}/{q.id}: {p.sides[1]}/{q.sides[3]}"
            assert np.array_equal(segs[p.id][1], segs[q.id][3][::-1]), f"pieces {p.id}/{q.id}: shared edge differs"
        if p.row + 1 < cut.rows:
            q = by_cell[(p.row + 1, p.col)]
            assert {p.sides[2], q.sides[0]} == {"tab", "blank"}, f"pieces {p.id}/{q.id}: {p.sides[2]}/{q.sides[0]}"
            assert np.array_equal(segs[p.id][2], segs[q.id][0][::-1]), f"pieces {p.id}/{q.id}: shared edge differs"
    assert abs(total - cut.width * cut.height) < 1e-6 * cut.width * cut.height, f"areas sum to {total}"
    _check_tiling(cut)


def _check_tiling(cut: Cut) -> None:
    """Every pixel away from the cut lines is covered by exactly one piece (no gaps, no overlaps)."""
    k = min(1.0, 1500.0 / max(cut.width, cut.height))
    w, h = int(math.ceil(cut.width * k)), int(math.ceil(cut.height * k))
    count = np.zeros((h, w), np.int32)
    lines = np.zeros((h, w), np.uint8)
    shift = 4
    for p in cut.pieces:
        pts = np.round(((p.outline + 0.5) * k - 0.5) * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 1, cv2.LINE_8, shift=shift)
        count += m
        cv2.polylines(lines, [pts], True, 1, 3, cv2.LINE_8, shift=shift)
    bad = (count != 1) & (lines == 0)
    assert not bad.any(), f"{int(bad.sum())} pixels away from cut lines are covered {sorted(set(count[bad].tolist()))} times"


# ----------------------------------------------------------------------------------------------
# Scene checks
# ----------------------------------------------------------------------------------------------


def _homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ph = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def _canonical(img_gray: np.ndarray, quad: np.ndarray, size: int = 48, inner: float = 0.15) -> np.ndarray:
    """Sample the inner part of the core quad (TL, TR, BR, BL) onto a size x size square."""
    lo, hi = inner * size, (1 - inner) * size
    sq = np.array([[lo, lo], [hi, lo], [hi, hi], [lo, hi]])
    full = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float64)
    Hq = _homography(full, quad)
    sub = _homography(np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float64), _apply(Hq, sq))
    return cv2.warpPerspective(img_gray, sub, (size, size), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)


def check_scene(scene: dict, mask: np.ndarray, frame: np.ndarray, pieces: dict[int, dict], motif_gray: np.ndarray,
                stats: dict) -> None:
    H, W = mask.shape
    assert scene["width"] == W and scene["height"] == H and frame.shape[:2] == (H, W)
    assert scene["background"] in ("plain", "wood", "cloth", "pattern", "clutter", "real")
    ps = scene["pieces"]
    indices = [p["index"] for p in ps]
    assert indices == list(range(1, len(ps) + 1)), f"indices {indices} are not 1..n in paint order"
    present = set(np.unique(mask).tolist()) - {0}
    assert present <= set(indices), f"mask has indices {sorted(present - set(indices))} missing from the JSON"
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    for p in ps:
        ref = pieces[p["pieceId"]]
        assert (p["col"], p["row"], p["cell"]) == (ref["col"], ref["row"], ref["cell"]), "col/row/cell mismatch"
        c = np.array(p["corners"], np.float64)
        o = np.array(p["outline"], np.float64)
        vis = p["visibleFraction"]
        assert 0.0 <= vis <= 1.0
        if vis > 0.02:
            assert p["index"] in present, f"piece {p['index']} is {vis:.2f} visible but absent from the mask"
        assert signed_area(o) > 0, f"piece {p['index']}: scene outline is not clockwise"
        quad_area = signed_area(c)
        assert (quad_area > 0) == p["faceUp"], f"piece {p['index']}: corner order does not match faceUp"
        corner_indices(o, c, tol=0.011)
        # Outline and corners come from one motif -> frame map (the ground truth is self-consistent).
        Hmf = _homography(np.array(ref["corners"], np.float64), c)
        up = up_angle_deg(Hmf, np.array(ref["corners"], np.float64))
        d = (up - p["upAngleDeg"] + 180) % 360 - 180
        assert abs(d) < 0.02, f"piece {p['index']}: upAngleDeg {p['upAngleDeg']} vs corners {up:.2f}"
        mo = np.array(ref["outline"], np.float64)
        mapped = _apply(Hmf, mo)
        if not p["faceUp"]:
            mapped = mapped[::-1]
        err = float(np.abs(mapped - o).max())
        assert err < 0.05, f"piece {p['index']}: outline is not the motif outline under the corner homography ({err:.3f} px)"
        if p["faceUp"] and vis >= 0.999:
            x0, y0 = np.floor(o.min(axis=0)).astype(int) - 2
            x1, y1 = np.ceil(o.max(axis=0)).astype(int) + 3
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [np.round((o - (x0, y0)) * 16).astype(np.int32).reshape(-1, 1, 2)], 1, cv2.LINE_8, shift=4)
            mk = mask[y0:y1, x0:x1] == p["index"]
            recall = float((m.astype(bool) & mk).sum()) / max(1, int(m.sum()))
            assert recall >= MASK_TOP_RECALL_MIN, f"piece {p['index']}: only {recall:.3f} of the top face is in the mask"
            side = float(np.mean(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)))
            beyond = cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)[mk]
            far = float(beyond.max()) if beyond.size else 0.0
            assert far <= WALL_MAX_FRACTION * side + 1.5, \
                f"piece {p['index']}: mask reaches {far:.1f} px beyond the outline (side {side:.0f} px)"
            stats.setdefault("wall", []).append(far / side)
            _photometric(frame_gray, motif_gray, c, np.array(ref["corners"], np.float64), stats)


def _photometric(frame_gray: np.ndarray, motif_gray: np.ndarray, frame_corners: np.ndarray, motif_corners: np.ndarray,
                 stats: dict) -> None:
    """Does the frame show the motif cell where the ground truth says, in the stated rotation?"""
    ref = _canonical(motif_gray, motif_corners)
    if float(ref.std()) < 6.0:  # too flat to tell rotations apart (sky, water)
        return
    scores = [_zncc(_canonical(frame_gray, np.roll(frame_corners, -r, axis=0)), ref) for r in range(4)]
    stats.setdefault("true", []).append(scores[0])
    stats.setdefault("wins", []).append(int(np.argmax(scores) == 0))


def check_reference(ref: dict, img: np.ndarray, motif_gray: np.ndarray, stats: dict) -> None:
    for k in ("cols", "rows", "motifSize", "referenceCorners", "cut", "source"):
        assert k in ref, f"reference.json lacks {k}"
    W, H = ref["motifSize"]
    assert motif_gray.shape == (H, W)
    q = np.array(ref["referenceCorners"], np.float64)
    assert q.shape == (4, 2) and signed_area(q) > 0, "referenceCorners are not a clockwise quad"
    h, w = img.shape[:2]
    assert np.all(q > -0.1 * max(w, h)) and np.all(q[:, 0] < 1.1 * w) and np.all(q[:, 1] < 1.1 * h)
    # Rectify the reference with its ground-truth corners and compare with the motif (small scale).
    s = 256.0 / max(W, H)
    size = (int(round(W * s)), int(round(H * s)))
    dst = np.array([[-0.5, -0.5], [size[0] - 0.5, -0.5], [size[0] - 0.5, size[1] - 0.5], [-0.5, size[1] - 0.5]])
    Hr = _homography(q, dst)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.5 / s)
    rect = cv2.warpPerspective(gray, Hr, size, flags=cv2.INTER_LINEAR)
    mot = cv2.resize(motif_gray.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    inner = (slice(size[1] // 10, -size[1] // 10), slice(size[0] // 10, -size[0] // 10))
    z = _zncc(rect[inner], mot[inner])
    stats.setdefault("ref", []).append(z)
    assert z >= REF_ZNCC_MIN, f"reference rectified with referenceCorners matches the motif poorly (ZNCC {z:.2f})"
    # Registration: shifting the rectified reference by a quarter cell must make the match worse.
    cell = size[0] / ref["cols"] / 4
    for dx, dy in ((cell, 0), (-cell, 0), (0, cell), (0, -cell)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        zs = _zncc(cv2.warpAffine(rect, M, size)[inner], mot[inner])
        assert zs < z, f"reference corners are off by about a quarter cell ({zs:.3f} >= {z:.3f})"


def check_dataset(root: Path) -> dict:
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    for k in ("format", "generator", "seed", "split", "puzzles", "scenes"):
        assert k in meta, f"meta.json lacks {k}"
    stats: dict = {}
    pieces_by: dict[str, dict[int, dict]] = {}
    motifs: dict[str, np.ndarray] = {}
    for pid in meta["puzzles"]:
        pdir = root / "puzzles" / pid
        ref = json.loads((pdir / "reference.json").read_text(encoding="utf-8"))
        raw = json.loads((pdir / "pieces.json").read_text(encoding="utf-8"))
        W, H = ref["motifSize"]
        cut = Cut(cols=ref["cols"], rows=ref["rows"], width=W, height=H, kind=ref["cut"], lattice=np.zeros((0, 0, 2)),
                  pieces=[Piece(p["id"], p["col"], p["row"], p["sides"], np.array(p["corners"]), np.array(p["outline"]))
                          for p in raw])
        check_cut(cut)
        pieces_by[pid] = {p["id"]: p for p in raw}
        motifs[pid] = cv2.cvtColor(read_rgb(pdir / "motif.png"), cv2.COLOR_RGB2GRAY).astype(np.float32)
        check_reference(ref, read_rgb(pdir / "reference.jpg"), motifs[pid], stats)
    n_pieces = 0
    for sid in meta["scenes"]:
        scene = json.loads((root / "scenes" / f"{sid}.json").read_text(encoding="utf-8"))
        mask = read_gray(root / "scenes" / f"{sid}_mask.png")
        assert mask.dtype == np.uint8 and mask.ndim == 2, "mask must be 8-bit single channel"
        frame = read_rgb(root / "scenes" / f"{sid}.jpg")
        try:
            check_scene(scene, mask, frame, pieces_by[scene["puzzleId"]], motifs[scene["puzzleId"]], stats)
        except AssertionError as e:
            raise AssertionError(f"{root.name}/{sid}: {e}") from None
        n_pieces += len(scene["pieces"])
    if stats.get("wins"):
        win = float(np.mean(stats["wins"]))
        med = float(np.median(stats["true"]))
        assert win >= ZNCC_ROT_WIN_MIN, f"true rotation wins only {win:.2%} of textured pieces"
        assert med >= ZNCC_TRUE_MEDIAN_MIN, f"median ZNCC frame vs motif {med:.2f}"
    stats["n_pieces"] = n_pieces
    return stats


# ----------------------------------------------------------------------------------------------
# Checker self-tests: corrupted data must be rejected
# ----------------------------------------------------------------------------------------------


def _must_fail(fn, *args, what: str) -> None:
    try:
        fn(*args)
    except AssertionError:
        return
    raise AssertionError(f"checker accepted corrupted data: {what}")


def test_checkers_reject_corruption(root: Path) -> None:
    cut = round_cut(make_cut(rng_for(5, "mut"), 900, 600, 6, 4, "grid"))
    check_cut(cut)
    c1 = copy.deepcopy(cut)
    c1.pieces[7].outline = c1.pieces[7].outline[::-1].copy()
    _must_fail(check_cut, c1, what="counter-clockwise outline")
    c2 = copy.deepcopy(cut)
    c2.pieces[7].corners = np.roll(c2.pieces[7].corners, 1, axis=0)
    _must_fail(check_cut, c2, what="corners not in TL, TR, BR, BL order")
    c3 = copy.deepcopy(cut)
    i = next(k for k, s in enumerate(c3.pieces[7].sides) if s != "flat")
    c3.pieces[7].sides[i] = "tab" if c3.pieces[7].sides[i] == "blank" else "blank"
    _must_fail(check_cut, c3, what="tab/blank label flipped")
    c4 = copy.deepcopy(cut)
    c4.pieces[7].outline = c4.pieces[7].outline + np.array([3.0, 0.0])
    c4.pieces[7].corners = c4.pieces[7].corners + np.array([3.0, 0.0])
    _must_fail(check_cut, c4, what="piece shifted so it overlaps its neighbour")

    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    sid = next(s for s in meta["scenes"]
               if any(p["faceUp"] for p in json.loads((root / "scenes" / f"{s}.json").read_text())["pieces"]))
    scene = json.loads((root / "scenes" / f"{sid}.json").read_text(encoding="utf-8"))
    mask = read_gray(root / "scenes" / f"{sid}_mask.png")
    frame = read_rgb(root / "scenes" / f"{sid}.jpg")
    pdir = root / "puzzles" / scene["puzzleId"]
    pieces = {p["id"]: p for p in json.loads((pdir / "pieces.json").read_text(encoding="utf-8"))}
    motif = cv2.cvtColor(read_rgb(pdir / "motif.png"), cv2.COLOR_RGB2GRAY).astype(np.float32)
    check_scene(scene, mask, frame, pieces, motif, {})
    k = next(j for j, p in enumerate(scene["pieces"]) if p["faceUp"])
    s1 = copy.deepcopy(scene)
    s1["pieces"][k]["corners"] = s1["pieces"][k]["corners"][1:] + s1["pieces"][k]["corners"][:1]
    _must_fail(check_scene, s1, mask, frame, pieces, motif, {}, what="scene corners rotated by one")
    s2 = copy.deepcopy(scene)
    s2["pieces"][k]["upAngleDeg"] = (s2["pieces"][k]["upAngleDeg"] + 90) % 360
    _must_fail(check_scene, s2, mask, frame, pieces, motif, {}, what="upAngleDeg off by 90")
    m2 = mask.copy()
    m2[m2 == scene["pieces"][-1]["index"]] = len(scene["pieces"]) + 1
    _must_fail(check_scene, scene, m2, frame, pieces, motif, {}, what="mask index not in the JSON")
    m3 = mask.copy()
    idx = scene["pieces"][k]["index"]
    m3[m3 == idx] = 0
    m3[np.roll(mask == idx, 25, axis=1)] = idx
    _must_fail(check_scene, scene, m3, frame, pieces, motif, {}, what="mask region 25 px away from its outline")
    s3 = copy.deepcopy(scene)
    s3["pieces"][k]["outline"] = [[x + 2.0, y] for x, y in s3["pieces"][k]["outline"]]
    _must_fail(check_scene, s3, mask, frame, pieces, motif, {}, what="outline shifted by 2 px")


# ----------------------------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------------------------


def _hash_tree(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    t0 = time.perf_counter()
    for n, aspect, kind, seed in [(24, 1.5, "grid", 1), (100, 1.5, "irregular", 2), (500, 4 / 3, "grid", 3),
                                  (500, 0.75, "irregular", 4), (1000, 1.5, "irregular", 5), (2000, 1.4, "grid", 6)]:
        cols, rows = choose_grid(n, aspect)
        W = 3000 if aspect >= 1 else int(3000 * aspect)
        H = int(W / aspect)
        check_cut(round_cut(make_cut(rng_for(seed, "t"), W, H, cols, rows, kind)))
        print(f"ok  cut {kind:9s} {cols}x{rows} on {W}x{H}")
    for seed in range(20):  # many small irregular grids: tab collisions would show up here
        check_cut(make_cut(rng_for(seed, "irr"), 1200, 900, 8, 6, "irregular"))
    print("ok  20 irregular 8x6 cuts tile exactly")

    with tempfile.TemporaryDirectory(prefix="synth_test_") as tmp:
        a, b = Path(tmp) / "a", Path(tmp) / "b"
        args = ["--pieces", "40", "--puzzles", "2", "--scenes", "6", "--seed", "3", "--cut", "mixed"]
        make_dataset_main(["--out", str(a), "--workers", "1", *args])
        make_dataset_main(["--out", str(b), "--workers", "3", *args])
        ha, hb = _hash_tree(a), _hash_tree(b)
        assert ha == hb, f"not deterministic: {sorted(k for k in ha if ha.get(k) != hb.get(k))}"
        print(f"ok  deterministic: {len(ha)} files identical with 1 and 3 workers")
        st = check_dataset(a)
        print(f"ok  mini dataset invariants ({st['n_pieces']} scene pieces)")
        test_checkers_reject_corruption(a)
        print("ok  every checker rejects corrupted data")

    for d in argv:
        st = check_dataset(Path(d))
        wins = f"true rotation wins {np.mean(st['wins']):.1%}, median ZNCC {np.median(st['true']):.2f}" if st.get("wins") else "no textured unoccluded pieces"
        wall = f"side wall <= {max(st['wall']):.3f} x core" if st.get("wall") else ""
        print(f"ok  {d}: {st['n_pieces']} scene pieces; {wins}; {wall}; reference ZNCC {np.round(st['ref'], 2).tolist()}")
    print(f"all synth tests passed in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
