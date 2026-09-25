"""The frozen evaluation sets: the exact commands (piece counts, sizes, seeds) behind every number in eval/.

Why a script and not a README line: the engine is tuned on `val` and judged on `test`, and those
numbers are only comparable across weeks if the sets are regenerated bit-for-bit from the same
arguments. Change a seed here and every historic report stops being comparable, so bump SET_VERSION
(it is part of each set's folder name) when you do.

    python -m tools.synth.make_eval_sets            # all val + test sets
    python -m tools.synth.make_eval_sets --only test500 val1000

Artwork is held out per split (motifs.SPLIT_BUCKETS), so test puzzles never reuse val/train art.
"""

from __future__ import annotations

import argparse
import time

from . import make_dataset

SET_VERSION = 1
PIECE_COUNTS = (100, 300, 500, 1000, 2000)
# Scene counts chosen so each set has a few hundred matchable pieces (big 100-piece pieces fill the
# frame, so fewer fit per scene). Guessed, then checked against the generator's piece totals.
SCENES = {100: 60, 300: 45, 500: 40, 1000: 40, 2000: 40}
SPLITS = {"val": {"puzzles": 3, "seed_base": 200}, "test": {"puzzles": 4, "seed_base": 100}}


def eval_sets() -> list[dict]:
    sets = []
    for split, cfg in SPLITS.items():
        for i, n in enumerate(PIECE_COUNTS):
            sets.append({
                "name": f"{split}{n}",
                "out": f"datasets/v{SET_VERSION}/{split}{n}",
                "argv": ["--pieces", str(n), "--puzzles", str(cfg["puzzles"]), "--scenes", str(SCENES[n]),
                         "--split", split, "--seed", str(cfg["seed_base"] + i + 1), "--cut", "mixed"],
            })
    return sets


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m tools.synth.make_eval_sets", description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", help="set names, e.g. test500 val1000")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    for s in eval_sets():
        if args.only and s["name"] not in args.only:
            continue
        print(f"== {s['name']} -> {s['out']}", flush=True)
        make_dataset.main(["--out", s["out"], *s["argv"]])
    print(f"all sets: {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
