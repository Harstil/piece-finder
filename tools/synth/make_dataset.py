"""CLI: generate a synthetic Piece Finder dataset in the docs/DATASET.md layout.

    python -m tools.synth.make_dataset --out datasets/<name> --pieces 500 --puzzles 4 --scenes 60 \
        --split test --seed 1 [--cut grid|irregular|mixed] [--workers N] [--motifs auto|procedural|sources]

Why a single entry point: frozen test sets are regenerated bit-for-bit from (seed, arguments),
so everything random is keyed off `common.rng_for(seed, ...)` per puzzle and per scene. The output
does not depend on --workers or on generation order, and no file contains a timestamp. Timings go
to stdout only.

Puzzle i uses cut "grid" for even i and "irregular" for odd i under --cut mixed. Scene j shows
puzzle j mod --puzzles. Each worker process loads a puzzle from the files it was written to, so
the scenes are rendered from exactly the numbers in pieces.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import GENERATOR
from .common import pts_json, read_rgb, rng_for, write_jpg, write_json, write_png
from .cut import Cut, Piece, choose_grid, make_cut, round_cut
from .motifs import load_source_motif, motif_size, pick_aspect, procedural_motif, source_motifs
from .reference import make_reference
from .render import apply_print, make_print_style
from .scene import PuzzleForScenes, render_scene, write_scene

# Motif resolution: ~100 px per cell along the long side so pieces are never upsampled much
# in 1000-2000 piece scenes, within the ~3000 px of real box-art scans. Guessed.
MOTIF_LONG_SIDE = (3000, 4200)
MOTIF_PX_PER_CELL = 100


def motif_long_side(n_pieces: int, aspect: float) -> int:
    cells_long = math.sqrt(n_pieces * max(aspect, 1.0 / aspect))
    return int(np.clip(round(cells_long * MOTIF_PX_PER_CELL / 50) * 50, *MOTIF_LONG_SIDE))


# ----------------------------------------------------------------------------------------------
# Puzzles
# ----------------------------------------------------------------------------------------------


def _puzzle_job(job: dict) -> dict:
    t0 = time.perf_counter()
    seed, idx, pieces, cut_kind, motifs_mode = job["seed"], job["index"], job["pieces"], job["cut"], job["motifs"]
    pid = f"p{idx:03d}"
    pdir = Path(job["out"]) / "puzzles" / pid
    pdir.mkdir(parents=True, exist_ok=True)
    rng = rng_for(seed, "puzzle", idx)
    sources = source_motifs() if motifs_mode != "procedural" else []
    if motifs_mode == "sources" and not sources:
        raise SystemExit("--motifs sources: tools/synth/sources/motifs is empty (run fetch_sources.py first)")
    style = "source"
    if sources and (motifs_mode == "sources" or rng.random() < 0.75):
        path, provenance = sources[int(rng.integers(len(sources)))]
        probe = read_rgb(path)
        aspect = probe.shape[1] / probe.shape[0]
        motif = load_source_motif(path, motif_long_side(pieces, aspect))
    else:
        aspect = pick_aspect(rng)
        w, h = motif_size(motif_long_side(pieces, aspect), aspect)
        motif, style = procedural_motif(rng_for(seed, "motif", idx), w, h)
        provenance = f"procedural:{style}:{seed}:{idx}"
    t_motif = time.perf_counter()
    H, W = motif.shape[:2]
    cols, rows = choose_grid(pieces, W / H)
    cut = round_cut(make_cut(rng_for(seed, "cut", idx), W, H, cols, rows, cut_kind))
    write_png(pdir / "motif.png", motif)
    write_json(pdir / "pieces.json", [
        {"id": p.id, "col": p.col, "row": p.row, "cell": p.cell, "sides": p.sides,
         "corners": pts_json(p.corners), "outline": pts_json(p.outline)} for p in cut.pieces])
    t_cut = time.perf_counter()
    ref, corners, info = make_reference(rng_for(seed, "reference", idx), motif, pieces)
    write_jpg(pdir / "reference.jpg", ref, info["jpegQuality"])
    write_json(pdir / "reference.json", {
        "cols": cols, "rows": rows, "motifSize": [W, H], "referenceCorners": pts_json(corners, 2),
        "cut": cut_kind, "source": provenance,
        "referenceKind": info["referenceKind"], "referenceSize": [int(ref.shape[1]), int(ref.shape[0])],
        "piecesNominal": pieces, "motifStyle": style,
    })
    t_ref = time.perf_counter()
    return {"pid": pid, "cols": cols, "rows": rows, "size": [W, H], "style": style, "kind": info["referenceKind"],
            "t_motif": t_motif - t0, "t_cut": t_cut - t_motif, "t_ref": t_ref - t_cut}


def load_puzzle(pdir: Path) -> tuple[Cut, np.ndarray, dict]:
    """(cut as written to pieces.json, motif uint8 RGB, reference.json) for an existing puzzle."""
    ref = json.loads((pdir / "reference.json").read_text(encoding="utf-8"))
    motif = read_rgb(pdir / "motif.png")
    raw = json.loads((pdir / "pieces.json").read_text(encoding="utf-8"))
    pieces = [Piece(id=p["id"], col=p["col"], row=p["row"], sides=list(p["sides"]),
                    corners=np.array(p["corners"], np.float64), outline=np.array(p["outline"], np.float64))
              for p in raw]
    W, H = ref["motifSize"]
    cut = Cut(cols=ref["cols"], rows=ref["rows"], width=W, height=H, kind=ref["cut"], pieces=pieces,
              lattice=np.zeros((0, 0, 2)))
    return cut, motif, ref


# ----------------------------------------------------------------------------------------------
# Scenes
# ----------------------------------------------------------------------------------------------

_CACHE: dict[str, PuzzleForScenes] = {}


def _scene_job(job: dict) -> dict:
    t0 = time.perf_counter()
    out, seed, i, p = Path(job["out"]), job["seed"], job["index"], job["puzzle"]
    pid = f"p{p:03d}"
    if pid not in _CACHE:
        _CACHE.clear()  # one puzzle at a time per worker keeps memory flat
        cut, motif, ref = load_puzzle(out / "puzzles" / pid)
        style = make_print_style(rng_for(seed, "print", p))
        _CACHE[pid] = PuzzleForScenes(pid, cut, apply_print(motif, style), style, ref["piecesNominal"])
    t_load = time.perf_counter()
    frame, labels, scene, quality = render_scene(rng_for(seed, "scene", i), _CACHE[pid])
    sid = f"s{i:05d}"
    write_scene(frame, labels, scene, quality, out / "scenes", sid)
    return {"sid": sid, "pieces": len(scene["pieces"]), "t_load": t_load - t0, "t_render": time.perf_counter() - t_load}


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def _prepare_out(out: Path) -> None:
    """Start from a clean folder, but only ever delete what this tool writes."""
    if out.exists():
        known = {"meta.json", "puzzles", "scenes", "_preview"}
        present = {p.name for p in out.iterdir()}
        if present - known:
            raise SystemExit(f"{out} contains files this tool did not write: {sorted(present - known)}")
        for name in present:
            target = out / name
            shutil.rmtree(target) if target.is_dir() else target.unlink()
    (out / "puzzles").mkdir(parents=True, exist_ok=True)
    (out / "scenes").mkdir(parents=True, exist_ok=True)


def _map(fn, jobs: list[dict], workers: int):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, jobs, chunksize=max(1, len(jobs) // (workers * 4))))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.synth.make_dataset", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output folder, e.g. datasets/test500")
    ap.add_argument("--pieces", type=int, default=500, help="nominal piece count per puzzle")
    ap.add_argument("--puzzles", type=int, default=4)
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cut", default="mixed", choices=["grid", "irregular", "mixed"])
    ap.add_argument("--motifs", default="auto", choices=["auto", "procedural", "sources"],
                    help="auto: downloaded art when present (75%%), else procedural")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    args = ap.parse_args(argv)
    if args.pieces < 4 or args.puzzles < 1 or args.scenes < 0:
        raise SystemExit("need --pieces >= 4, --puzzles >= 1, --scenes >= 0")

    out = Path(args.out)
    t0 = time.perf_counter()
    _prepare_out(out)
    pjobs = [{"out": str(out), "seed": args.seed, "index": i, "pieces": args.pieces, "motifs": args.motifs,
              "cut": args.cut if args.cut != "mixed" else ("grid" if i % 2 == 0 else "irregular")}
             for i in range(args.puzzles)]
    presults = _map(_puzzle_job, pjobs, min(args.workers, args.puzzles))
    t1 = time.perf_counter()
    for r in presults:
        print(f"  {r['pid']}: {r['cols']}x{r['rows']} motif {r['size'][0]}x{r['size'][1]} ({r['style']}), "
              f"reference {r['kind']} | motif {r['t_motif']:.1f}s cut {r['t_cut']:.1f}s reference {r['t_ref']:.1f}s")
    print(f"puzzles: {args.puzzles} in {t1 - t0:.1f}s")
    sjobs = [{"out": str(out), "seed": args.seed, "index": i, "puzzle": i % args.puzzles} for i in range(args.scenes)]
    sresults = _map(_scene_job, sjobs, args.workers)
    t2 = time.perf_counter()
    if sresults:
        rend = [r["t_render"] for r in sresults]
        n_pieces = sum(r["pieces"] for r in sresults)
        print(f"scenes: {len(sresults)} ({n_pieces} pieces) in {t2 - t1:.1f}s wall, "
              f"render {np.mean(rend):.2f}s mean / {np.max(rend):.2f}s max per scene, {args.workers} workers")
    write_json(out / "meta.json", {
        "format": 1, "generator": GENERATOR, "seed": args.seed, "split": args.split,
        "puzzles": [f"p{i:03d}" for i in range(args.puzzles)],
        "scenes": [f"s{i:05d}" for i in range(args.scenes)],
    })
    print(f"total: {time.perf_counter() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    sys.exit(main())
