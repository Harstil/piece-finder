/**
 * The reference model: the box picture, straightened and cut into the puzzle's grid, at every matching
 * resolution. Built once per puzzle (in the worker when the user confirms the picture's corners).
 *
 * How:
 *   1. The user marks the motif's 4 corners on the box photo (TL, TR, BR, BL, pixel-edge convention as
 *      in geom/warp.ts). The photo is cropped to them and, when it is much sharper than needed,
 *      area-downsampled first so the perspective warp cannot alias.
 *   2. One perspective warp straightens it to cols·W × rows·W px (W ≤ 64 px per cell). Cells are made
 *      square here even when the real cells are not (Ravensburger "1000" = 36 × 28 with 19.4 × 17.8 mm
 *      cells): the piece is warped corner-to-corner onto the same square, so both sides agree.
 *   3. Each level (levels.ts: 8, 16, 32 px per cell) is an area-downsample of that, padded by the level
 *      margin on every side (valid = 0 there) so windows around border cells stay in bounds, and stores
 *      Lab planes plus Sobel gradients of L computed before padding (so the motif's own border does not
 *      look like an edge).
 *
 * Window convention: at a level with cell size s and margin m, the window of cell (col, row) — the
 * cell's core plus a margin on every side, exactly the canonical piece's layout — starts at padded
 * pixel (col·s, row·s) and is s + 2m px square.
 */

import { homographyFromQuad, warpRGBA } from './geom/index.ts'
import { clipRect, cropRGBA, fitSize, resizeAreaRGBA, rgbaToLab, sobel } from './image/index.ts'
import { FINEST, LEVEL_SIZES, levelMargin } from './levels.ts'
import type { GridSpec, Point, Quad, RGBAImage, ReferenceSummary } from './types.ts'

/** Working resolution per cell before downsampling to the levels. 2 × FINEST: the finest level is a clean 2:1 area average. */
const WORK_CELL_PX = 2 * FINEST
/** Cap on the straightened working image (px). Keeps a 2000-piece build under ~40 MB on a phone. Guessed. */
const MAX_WORK_PIXELS = 8_000_000
/** Long side of the straightened preview the UI draws the grid and highlights on. */
const PREVIEW_LONG_SIDE = 1400

export interface ReferenceLevel {
  /** Cell core size in px. */
  size: number
  /** Padding around the motif (and the window margin) in px. */
  margin: number
  /** Padded level size: cols·size + 2·margin by rows·size + 2·margin. */
  width: number
  height: number
  L: Float32Array
  A: Float32Array
  B: Float32Array
  GX: Float32Array
  GY: Float32Array
  /** 1 inside the motif, 0 in the padding. */
  valid: Uint8Array
}

export interface ReferenceModel {
  readonly grid: GridSpec
  /** Straightened motif size at the working resolution, px. */
  readonly width: number
  readonly height: number
  /** Working px per cell (x and y are equal by construction; kept for the contract). */
  readonly cellW: number
  readonly cellH: number
  /** Photo px per cell along the shorter cell side: below ~48 the picture is too small to be reliable. */
  readonly nativeCellPx: number
  /** Coarsest first, same order as LEVEL_SIZES. */
  readonly levels: ReferenceLevel[]
  /** Straightened motif for display (RGBA, long side ≤ PREVIEW_LONG_SIDE). */
  readonly preview: RGBAImage
}

export interface BuildReferenceOptions {
  previewLongSide?: number
}

function dist(a: Point, b: Point): number {
  return Math.hypot(b.x - a.x, b.y - a.y)
}

/** Photo px per cell along the shorter cell side, from the motif corners. */
export function photoCellPx(corners: Quad, grid: GridSpec): number {
  const w = Math.min(dist(corners[0], corners[1]), dist(corners[3], corners[2])) / grid.cols
  const h = Math.min(dist(corners[0], corners[3]), dist(corners[1], corners[2])) / grid.rows
  return Math.min(w, h)
}

export function buildReference(photo: RGBAImage, corners: Quad, grid: GridSpec, opts: BuildReferenceOptions = {}): ReferenceModel {
  const { cols, rows } = grid
  if (!(cols >= 2 && rows >= 2)) throw new Error(`grid must be at least 2 × 2, got ${cols} × ${rows}`)
  const native = photoCellPx(corners, grid)
  const work = Math.max(
    LEVEL_SIZES[0],
    Math.min(WORK_CELL_PX, Math.floor(Math.sqrt(MAX_WORK_PIXELS / (cols * rows)))),
  )

  // 1. Crop to the motif, and pre-shrink when the photo has more than twice the needed resolution.
  let xs = corners.map((c) => c.x)
  let ys = corners.map((c) => c.y)
  const rect = clipRect(Math.min(...xs) - 2, Math.min(...ys) - 2, Math.max(...xs) + 2, Math.max(...ys) + 2, photo.width, photo.height)
  if (rect.width < 4 || rect.height < 4) throw new Error('the picture corners enclose (almost) nothing')
  let src = cropRGBA(photo, rect)
  xs = xs.map((x) => x - rect.x)
  ys = ys.map((y) => y - rect.y)
  const shrink = Math.floor(native / work)
  if (shrink >= 2) {
    const w = Math.max(1, Math.round(src.width / shrink))
    const h = Math.max(1, Math.round(src.height / shrink))
    const sx = w / src.width
    const sy = h / src.height
    src = resizeAreaRGBA(src, w, h)
    xs = xs.map((x) => (x + 0.5) * sx - 0.5)
    ys = ys.map((y) => (y + 0.5) * sy - 0.5)
  }

  // 2. Straighten: every cell becomes work × work px.
  const W = cols * work
  const H = rows * work
  const outRect: Point[] = [
    { x: -0.5, y: -0.5 },
    { x: W - 0.5, y: -0.5 },
    { x: W - 0.5, y: H - 0.5 },
    { x: -0.5, y: H - 0.5 },
  ]
  const quad: Point[] = xs.map((x, i) => ({ x, y: ys[i] }))
  const toSrc = homographyFromQuad(outRect, quad)
  if (toSrc === null) throw new Error('the picture corners do not form a usable quadrilateral')
  const straight = warpRGBA(src, toSrc, W, H)

  // 3. Levels.
  const levels = LEVEL_SIZES.map((s) => buildLevel(straight, cols, rows, s))
  const pv = fitSize(W, H, opts.previewLongSide ?? PREVIEW_LONG_SIDE)
  const preview = pv.width < W ? resizeAreaRGBA(straight, pv.width, pv.height) : straight

  return { grid: { cols, rows }, width: W, height: H, cellW: work, cellH: work, nativeCellPx: native, levels, preview }
}

function buildLevel(straight: RGBAImage, cols: number, rows: number, size: number): ReferenceLevel {
  const m = levelMargin(size)
  const w = cols * size
  const h = rows * size
  const small = w === straight.width && h === straight.height ? straight : resizeAreaRGBA(straight, w, h)
  const lab = rgbaToLab(small)
  const grad = sobel({ width: w, height: h, data: lab.L })
  const pw = w + 2 * m
  const ph = h + 2 * m
  const n = pw * ph
  const level: ReferenceLevel = {
    size,
    margin: m,
    width: pw,
    height: ph,
    L: new Float32Array(n),
    A: new Float32Array(n),
    B: new Float32Array(n),
    GX: new Float32Array(n),
    GY: new Float32Array(n),
    valid: new Uint8Array(n),
  }
  for (let y = 0; y < h; y++) {
    const from = y * w
    const to = (y + m) * pw + m
    level.L.set(lab.L.subarray(from, from + w), to)
    level.A.set(lab.a.subarray(from, from + w), to)
    level.B.set(lab.b.subarray(from, from + w), to)
    level.GX.set(grad.dx.data.subarray(from, from + w), to)
    level.GY.set(grad.dy.data.subarray(from, from + w), to)
    level.valid.fill(1, to, to + w)
  }
  return level
}

export function referenceSummary(ref: ReferenceModel): ReferenceSummary {
  return { grid: ref.grid, cellSizePx: ref.nativeCellPx, width: ref.width, height: ref.height }
}
