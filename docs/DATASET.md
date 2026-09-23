# Dataset format (contract)

The Python generator in `tools/synth/` writes datasets in this format. The Node evaluation harness in `eval/`
and the segmentation trainer in `tools/train/` read them. Real-world sets captured from the user's own puzzle
use the same format, so every metric can be computed on synthetic and real data alike.

All conventions (coordinates, grid indexing, corner/side order, angles) are the ones in `src/engine/types.ts`.

```
datasets/<datasetName>/
  meta.json
  puzzles/<puzzleId>/
    motif.png            the true printed image (the whole puzzle picture, as printed on the pieces)
    reference.jpg        the reference the app would get: a simulated (or real) phone photo of the box
    reference.json
    pieces.json
  scenes/<sceneId>.jpg        a camera frame with pieces lying on a table
  scenes/<sceneId>.json
  scenes/<sceneId>_mask.png   8-bit instance map: 0 = background, k = scene piece index k (1-based)
```

`datasets/` is git-ignored. Frozen test sets are regenerated bit-for-bit from their seed.

## meta.json
```json
{
  "format": 1,
  "generator": "tools/synth 0.1.0",
  "seed": 1234,
  "split": "test",                      // "train" | "val" | "test" | "real"
  "puzzles": ["p000", "p001"],
  "scenes": ["s00000", "s00001"]
}
```

## puzzles/<id>/reference.json
```json
{
  "cols": 25, "rows": 20,
  "motifSize": [2000, 1600],            // size of motif.png in px
  "referenceCorners": [[x,y],[x,y],[x,y],[x,y]],  // TL,TR,BR,BL of the motif area inside reference.jpg
  "cut": "grid",                        // "grid" (ribbon cut) | "irregular" (jittered, still 4-connected)
  "source": "aic:12345"                 // provenance of the artwork
}
```
The reference corners are the ground truth for the auto-quad detector; the app itself gets them from the user.

## puzzles/<id>/pieces.json
```json
[
  { "id": 0, "col": 0, "row": 0, "cell": 0,
    "sides": ["flat","tab","blank","flat"],      // top,right,bottom,left in motif orientation
    "corners": [[x,y],[x,y],[x,y],[x,y]],        // TL,TR,BR,BL core corners in motif.png coordinates
    "outline": [[x,y], ...]                      // full piece outline incl. tabs, motif coords, clockwise
  }
]
```

## scenes/<id>.json
```json
{
  "puzzleId": "p000",
  "width": 1920, "height": 1080,
  "background": "wood",                 // "plain" | "wood" | "cloth" | "pattern" | "clutter" | "real"
  "pieces": [
    { "index": 1,                       // value in the _mask.png instance map
      "pieceId": 137, "col": 12, "row": 5, "cell": 137,
      "upAngleDeg": 212.4,              // where the piece's motif-up points in the frame (clockwise from up)
      "corners": [[x,y],[x,y],[x,y],[x,y]],  // frame coords, in MOTIF order TL,TR,BR,BL (not image order)
      "outline": [[x,y], ...],          // frame coords
      "visibleFraction": 1.0,           // < 1 when occluded by another piece or the frame edge
      "faceUp": true
    }
  ]
}
```
Pieces that are face-down, or less than 60 % visible, are still in the mask but are excluded from matching
metrics (`faceUp: false` or `visibleFraction < 0.6`).

## Metrics computed on this format (by `eval/`)
- matching: top-1 / top-5 cell accuracy, rotation accuracy (|Δ upAngleDeg| < 45°), by piece count and piece type;
- shape: corner error (px, relative to core side length), side-kind accuracy;
- segmentation: instance recall / precision at IoU ≥ 0.8, mean IoU, boundary F-score;
- latency per stage.
