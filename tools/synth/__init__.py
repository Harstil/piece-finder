"""Synthetic jigsaw dataset generator for Piece Finder.

Every recognition-engine decision is measured on the data this package writes, and the
segmentation network is trained on it, so the package is built around two promises:
exact ground truth (see docs/DATASET.md and src/engine/types.ts) and bit-for-bit
reproducibility from a seed. See tools/synth/README.md for the commands.
"""

__version__ = "0.1.0"
GENERATOR = f"tools/synth {__version__}"
