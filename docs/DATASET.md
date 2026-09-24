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

## Precise definitions
These pin down what the examples above leave open. `node eval/check-dataset.ts datasets/<name>` checks them
numerically through the engine's own primitives.

- **Motif area.** Pixel centres are integers, so the motif area is the pixel-*edge* rectangle
  `[-0.5, W-0.5] × [-0.5, H-0.5]`. `referenceCorners` are its four outer corners, border pieces' corners lie
  on it, and lattice point (c, r) of a grid cut is `(-0.5 + c·W/cols, -0.5 + r·H/rows)`. This matches
  `rectToQuadHomography` in `src/engine/geom/warp.ts`, which maps a quad to the output's outer pixel corners.
- **Clockwise** means clockwise on screen (y down): a positive shoelace sum. Outlines never repeat their first
  point. The 4 core corners are vertices of the outline.
- **`upAngleDeg`** is the frame direction of the motif-up vector (motif −y) at the core centre, i.e. at the
  image of the mean of the 4 motif-space corners, from the motif → frame homography. It is *not* the
  bottom-edge → top-edge midpoint direction, which differs by up to ~9° on irregular cuts.
- **Outline vs mask.** `outline` and `corners` describe the printed **top face**. The instance mask covers
  the top face **plus the visible side wall** (pieces are ~0.1 × core thick; parallax shows the wall on the side
  facing the camera's nadir). So for a fully visible piece the whole top face is in the mask, but mask/outline
  IoU is typically 0.86–0.99, not ~1. Segmentation metrics compare against the mask; shape metrics (corner
  error, side kinds) against `corners` / `outline`.
- **Face-down pieces** are mirrored: their motif-order corners run counter-clockwise in the image (their
  `outline` is still clockwise). `upAngleDeg` is still defined, but is meaningless to the user.
- **`visibleFraction`** = top-face pixels labelled with this piece ÷ top-face pixels of the whole piece
  (occlusion by later pieces and by the frame edge both count). Mask indices are the 1-based paint order.
- **Extra fields.** Readers must ignore unknown keys. The generator adds a scene `render` block (corePx,
  tiltDeg, focalPx, jpegQuality, glare, phoneShadow, exposure, blurSigma, motionBlurPx, noise) and, in
  reference.json, `referenceKind` (photo | digital | video), `referenceSize`, `piecesNominal`, `motifStyle`.

## Metrics computed on this format (by `eval/`)
- matching: top-1 / top-5 cell accuracy, rotation accuracy (|Δ upAngleDeg| < 45°), by piece count and piece type;
- shape: corner error (px, relative to core side length), side-kind accuracy;
- segmentation: instance recall / precision at IoU ≥ 0.8, mean IoU, boundary F-score;
- latency per stage.
