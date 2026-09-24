# tools/synth — synthetic jigsaw data

Generates the datasets every Piece Finder engine decision is measured on, and that the segmentation
network is trained on: puzzle motifs, their jigsaw cuts, simulated photos of the box lid, and phone-camera
frames of loose pieces on a table, all with exact ground truth in the format of
[`docs/DATASET.md`](../../docs/DATASET.md) and the conventions of [`src/engine/types.ts`](../../src/engine/types.ts).

Python 3.13 with numpy, opencv-python, scipy and pillow; nothing else. Run everything from the repo root.

## Commands

```sh
# a dataset (writes datasets/<name>/ in the DATASET.md layout; datasets/ is git-ignored)
python -m tools.synth.make_dataset --out datasets/test500 --pieces 500 --puzzles 4 --scenes 60 --split test --seed 1
#   --cut grid|irregular|mixed   (mixed = grid for even puzzle indices, irregular for odd; default mixed)
#   --motifs auto|procedural|sources   (auto = downloaded art for ~75 % of puzzles when present)
#   --workers N                  (default min(8, cpus-2); output is identical for any N)

# ground-truth overlays and contact sheets -> datasets/<name>/_preview/
python -m tools.synth.preview datasets/test500 [--max-scenes 48]

# invariant tests (cut geometry, determinism, a fresh mini dataset, checker self-tests),
# optionally also checking existing datasets
python -m tools.synth.test_synth [datasets/smoke ...]

# download CC0 art + table textures into tools/synth/sources/ (needs the user's OK; not run yet)
python -m tools.synth.fetch_sources [--motifs 60] [--textures 25] [--contact you@example.com]
```

The same seed and arguments give **bit-for-bit identical files** (checked by `test_synth`, which generates
twice with 1 and 3 workers and compares SHA-256 of every file). Every random draw comes from
`common.rng_for(seed, <name>, <index>)`, a stream per puzzle / scene / sub-step, so adding scenes never
changes existing ones. No file contains a timestamp. The guarantee holds for the same OpenCV and numpy
versions and the same `sources/` folder; a library upgrade may change bytes (not geometry).

Measured on the dev laptop (Core 7 150U, 8 workers):

| dataset | puzzles | scenes | wall time |
|---|---|---|---|
| `smoke`: 100 pieces, `--seed 7 --cut mixed` | 2 in 10 s | 12 in 13 s | 24 s |
| `smoke1000`: 1000 pieces, `--seed 7` | 1 in 18 s | 6 in 6 s | 24 s |

Per puzzle: procedural motif 1–7 s, cut < 1 s (+ writing motif.png), box photo 6–12 s (12 MP). Per scene
~1.5–3 s single-threaded.

## Modules

| module | what it does |
|---|---|
| `cut.py` | Grid choice (`choose_grid(n, aspect)`) and the cut: bezier tabs (shoulder, fillet, neck, head), random tab/blank direction, jittered tab position/size/neck, edge waviness; `irregular` jitters interior lattice points ±12 % of a cell. |
| `motifs.py` | Loads `sources/motifs/*` or builds a procedural motif: landscape (sky gradients, soft clouds, mountains, tree lines, calm water with reflections, meadows with flowers and rocks), city (window grids, waterfront or street), busy illustration (many small objects, stripes, text-like marks), still life (dark painting backgrounds, shaded fruit, vase, bouquet), garden (dense flower beds). |
| `backgrounds.py` | Tables: plain / felt, wood (9 species, planks, rings, streaks), woven cloth, printed patterns (plaid, stripes, floral, geometric, dots), clutter (paper, notes, mug, pens, coins, lid, phone), texture files. 25 % borrow the motif's colours (hard case). |
| `render.py` | Printed-cardboard look: print colour shift (levels, saturation, gamma, ink balance), paper micro-noise, optional linen emboss, light cut-edge rim, side wall from parallax, drop + contact shadow, gloss glare field, face-down backs. |
| `camera.py` | Lighting map, phone shadow, white balance / exposure, defocus (growing across a tilted table), motion blur, vignette, sensor noise, phone denoise + sharpening halo. |
| `scene.py` | Camera frames (75 % portrait 1080×1920, else 1920×1080), placement (apart / touching / overlapping / cut by the frame edge), compositing in paint order, ground truth. |
| `reference.py` | Box lid (coloured frame, logo block, "N PIECES" badge, title, ±2 % crop/extension) as a phone photo (3000–4000 px, ≤ 30° tilt, lid walls, shadow, glare streak), a clean digital image, or a 1920 px video grab. |
| `make_dataset.py` | CLI; writes the DATASET.md layout, logs timings. |
| `preview.py` | Overlays: outlines, corners numbered 0–3 in motif order, an arrow along `upAngleDeg`, `c12 r5` labels, false-colour masks; references with the grid warped in; cuts over the motif. |
| `test_synth.py` | Invariants (below). |
| `fetch_sources.py` | AIC (CC0 public-domain artworks via IIIF at 3000 px) and Poly Haven (CC0 2k diffuse textures), rate-limited, resumable, `sources/manifest.json` with provenance and licence. |
| `common.py` | Random streams, anti-aliased polygon rasterisation, homographies, fBm noise, byte-stable writers. |

## Realism knobs

All are module-level constants with a comment saying whether the value is measured or guessed.

| where | knob | default |
|---|---|---|
| `scene.py` | `P_PORTRAIT`, `FOCAL_PX`, `MAX_TILT_DEG`, `TILT_SCALE_DEG` | 0.75, 1250–1550 px, 25°, \|N(0, 9°)\| |
| | `CORE_PX_LIMITS`, `CORE_PX_K` (core side ≈ K/√pieces, ±30 %) | 60–280 px, 2600 (100 → 182–280, 500 → 81–151, 1000 → 60–107 px) |
| | `MAX_PIECES`, `P_TOUCH`, `P_OVERLAP`, `OVERLAP_FRACTION`, `P_FRAME_EDGE`, `FREE_GAP_PX` | 25, 0.18, 0.08, 3–25 %, 0.12, 3 px |
| | `FACE_DOWN_MEAN` (+1–4 on clutter), `P_GLARE`, `P_PHONE_SHADOW`, `LIGHT_ELEVATION_DEG`, `SHADOW_STRENGTH`, `JPEG_QUALITY` | 0.5, 0.4, 0.2, 35–80°, 0.2–0.5, 60–95 |
| `render.py` | `THICKNESS` (board / core, measured), `RIM_FRACTION`, `RIM_PX`; per-puzzle ranges in `make_print_style` | 0.085–0.12, 0.013, 0.9–3 px |
| `camera.py` | `DEFOCUS_SIGMA`, `DEFOCUS_TILT_EXTRA`, `P_MOTION`, `MOTION_LEN`, `VIGNETTE`, `READ_NOISE`, `SHOT_NOISE`, `WB_SPREAD`, `EXPOSURE` | see file |
| `backgrounds.py` | `KIND_WEIGHTS`, `P_MOTIF_COLOURS`, `P_TEXTURE_FILE` | 0.22/0.26/0.17/0.2/0.15, 0.25, 0.5 |
| `reference.py` | `KIND_WEIGHTS` (photo/digital/video), `MISMATCH`, `MAX_PHOTO_TILT_DEG`, `PHOTO_LONG_SIDE`, `BOX_HEIGHT` | 0.6/0.2/0.2, 2 %, 30°, 3000–4000 px, 0.07–0.14 |
| `cut.py` | `IRREGULAR_JITTER`, `EDGE_POINTS`, `MAX_CELL_ASPECT`; tab shape ranges in `_tab_params` | 0.12, 64, 1.25 |
| `motifs.py` / `make_dataset.py` | `STYLE_WEIGHTS`, `MOTIF_LONG_SIDE`, `MOTIF_PX_PER_CELL` | 3000–4200 px, ~100 px per cell |

## Contract interpretations

Where DATASET.md leaves room, the generator does this (chosen to match `types.ts`):

1. **Pixel centres.** Pixel (i, j) is centred on the integer point (i, j). The motif area is the pixel-*edge*
   rectangle `[-0.5, W-0.5] × [-0.5, H-0.5]`; lattice point (c, r) of the cut starts at
   `(-0.5 + c·W/cols, -0.5 + r·H/rows)`. So border pieces have corners at -0.5 / W-0.5, and
   `referenceCorners` are the four outer corners of that rectangle mapped into reference.jpg.
2. **Clockwise** = clockwise as seen on screen (y down), i.e. a positive shoelace sum
   `Σ x_i·y_{i+1} − x_{i+1}·y_i`. Outlines never repeat their first point, and the 4 core corners are exact
   outline vertices, in TL, TR, BR, BL order along it.
3. **Rounding.** pieces.json coordinates are rounded to 0.01 px, and scenes are rendered from exactly those
   rounded numbers (workers re-read pieces.json), so written ground truth and pixels agree.
4. **Scene outline = the printed top face.** Pieces are ~0.1 × core thick; the side wall shows where
   parallax (and tilt) exposes it. The *mask* covers top face + visible side wall (what a segmenter must cut
   out); `outline`/`corners` describe the top face (what the matcher canonicalises).
5. **upAngleDeg** is the image direction of the motif-up vector (motif −y) at the core centre (mean of the four
   motif corners), from the Jacobian of the motif → frame homography, clockwise from image-up, in [0, 360).
   (Not the bottom-edge → top-edge midpoint direction: for jittered irregular cores that differs by up to ~9°.)
6. **visibleFraction** = top-face pixels that end up labelled with this piece ÷ top-face pixels of the whole
   piece rasterised without clipping (so both occlusion by later pieces and the frame edge count). A fully
   visible piece gets exactly 1.0.
7. **Mask indices** are the 1-based paint order; later pieces cover earlier ones. A piece that ends up
   completely hidden stays in the JSON with visibleFraction 0 and has no mask pixels.
8. **Face-down pieces** are other pieces of the same puzzle, flipped: their outline is mirrored, so their
   motif-order corners run *counter-clockwise* in the image (a checkable signal), while `outline` is still
   reported clockwise. `faceUp: false`; they are never shown face-up in the same frame.
9. The **clutter** background's "other face-down pieces" are added by scene.py as labelled distractors, not
   painted into the background, so every piece-shaped object is in the mask.
10. **Grid vs. piece count.** `cols × rows` comes from `choose_grid` and may differ from the nominal count
    (100 → 12×8 = 96 at 3:2); the box badge shows the nominal count (as real boxes do), recorded as
    `piecesNominal`.
11. **Extra fields** (readers must ignore unknown keys): scene `render` block (corePx, tiltDeg, focalPx,
    jpegQuality, glare, phoneShadow, exposure, blurSigma, motionBlurPx, noise) for per-condition metrics;
    reference.json `referenceKind` (photo | digital | video), `referenceSize`, `piecesNominal`, `motifStyle`.
12. **source** is `procedural:<style>:<seed>:<puzzle index>`, `aic:<artwork id>`, or `file:<name>` for a motif
    file without a manifest entry. Texture files get their background label from the manifest category
    (wood → `wood`, fabric → `cloth`, else `plain`).

## What test_synth checks

- Cuts (grid and irregular, 24 to 2000 pieces, plus 20 random irregular grids): ≥ 200 outline points,
  clockwise, no closing duplicate, corners are outline vertices in order, flat exactly on the border, each
  side's bulge matches its label (tab ≥ +10 % of the chord, blank ≤ −10 %, flat within 3 %), neighbours share
  the identical curve with tab ↔ blank, piece areas sum to W·H, and a raster of all pieces covers every pixel
  away from the cut lines exactly once.
- Scenes: indices 1..n in paint order, every mask value is in the JSON, visible pieces are in the mask,
  col/row/cell match pieces.json, outlines clockwise, corner winding matches `faceUp`, corners on the outline,
  `upAngleDeg` matches the corners, the outline is the pieces.json outline under the homography defined by the
  corners (< 0.05 px), for unoccluded pieces ≥ 97 % of the top face is in the mask and the mask reaches beyond the
  outline only by a side-wall band (≤ 0.16 × core + 1.5 px), and a **photometric** check: warping
  the frame into the motif cell through the ground-truth corners matches motif.png (median ZNCC ≥ 0.5) and the
  true rotation beats the other three for ≥ 90 % of textured pieces.
- References: rectifying reference.jpg with `referenceCorners` matches motif.png (ZNCC ≥ 0.35 despite badges,
  glare and crop), and shifting that rectification by a quarter cell always makes it worse.
- Determinism across worker counts, and a self-test that feeds each checker corrupted data (reversed outline,
  rotated corners, flipped tab label, overlapping piece, corners rotated in a scene, upAngle off by 90°, a
  stray mask index, a mask moved 25 px, a 2 px outline shift) and requires it to fail.

## Known limits

- Procedural motifs are stylised; real CC0 art (fetch_sources.py) should replace most of them once approved.
- Pieces lie flat (overlapping pieces are not tilted), the side wall is a single shaded colour, and glare is a
  smooth field rather than a reflection of an actual lamp shape.
- The box photo is a flat lid with walls; there is no hand, no curved lid, and no second object on top of it.
