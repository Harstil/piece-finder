/**
 * The canonical piece: a piece from a camera frame warped corner-to-corner onto the same square layout
 * as a reference cell window (levels.ts), at every matching resolution.
 *
 * One homography — the piece's 4 core corners to the canonical core square — removes the piece's
 * position, size, in-plane rotation and the camera's perspective at once. What remains between the
 * canonical piece and its true cell window is a quarter-turn rotation (the matcher tries all 4), print
 * versus photo colour drift, and small corner errors (the matcher's shift search absorbs those).
 *
 * The mask is the piece outline mapped into the canonical frame: tabs included (they reach into the
 * margin and add evidence), blanks excluded, and eroded by RIM_ERODE_PX so the light cut-edge rim and
 * any segmented side wall do not pollute the colour statistics.
 *
 * Aliasing: pieces are 60–280 px in the frame and the work size is 64 px, so large pieces are area-
 * downsampled (crop first) before the warp; each level is then an exact area-downsample of the work image.
 */

import { homographyFromQuad, invertHomography, rasterizePolygon, transformPoints, warpRGBA } from './geom/index.ts'
import {
  clipRect,
  cropRGBA,
  erode,
  rectKernel,
  resizeAreaGray,
  resizeAreaLab,
  resizeAreaRGBA,
  rgbaToLab,
} from './image/index.ts'
import { LEVEL_SIZES, WORK_SIZE, levelMargin, levelSpan } from './levels.ts'
import type { CanonicalLevel, CanonicalPiece, Mask, PieceShape, Point, RGBAImage } from './types.ts'

/** Erosion of the piece mask at work size (64 px core): ~5 % of the core. Guessed from the rim (1–3 px at ~100 px cores) plus wall bleed. */
const RIM_ERODE_PX = 3
/** A level pixel counts as "piece" when at least this share of its work pixels is. Guessed: keeps mixed edge pixels out. */
const LEVEL_MASK_MIN = 0.9
/** Crop margin around the piece outline, as a fraction of the core side. Tabs reach ~0.3; blur needs a little more. */
const CROP_MARGIN = 0.6

export interface CanonicalOptions {
  /** Skip the work-size pre-shrink (tests). */
  noPrefilter?: boolean
}

export function canonicalize(frame: RGBAImage, contour: Point[], shape: PieceShape, opts: CanonicalOptions = {}): CanonicalPiece {
  const c = shape.corners
  const core = 0.25 * (dist(c[0], c[1]) + dist(c[1], c[2]) + dist(c[2], c[3]) + dist(c[3], c[0]))

  // Crop around the outline, pre-shrink big pieces so the warp does not alias.
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const p of contour.length > 0 ? contour : c) {
    if (p.x < x0) x0 = p.x
    if (p.y < y0) y0 = p.y
    if (p.x > x1) x1 = p.x
    if (p.y > y1) y1 = p.y
  }
  const pad = CROP_MARGIN * core
  const rect = clipRect(x0 - pad, y0 - pad, x1 + pad, y1 + pad, frame.width, frame.height)
  let src = cropRGBA(frame, rect)
  let sx = 1
  let sy = 1
  const shrink = opts.noPrefilter === true ? 1 : Math.floor(core / WORK_SIZE)
  if (shrink >= 2) {
    const w = Math.max(1, Math.round(src.width / shrink))
    const h = Math.max(1, Math.round(src.height / shrink))
    sx = w / src.width
    sy = h / src.height
    src = resizeAreaRGBA(src, w, h)
  }
  const toLocal = (p: Point): Point => ({ x: (p.x - rect.x + 0.5) * sx - 0.5, y: (p.y - rect.y + 0.5) * sy - 0.5 })

  // Warp at work size.
  const S = WORK_SIZE
  const m = levelMargin(S)
  const N = levelSpan(S)
  const coreSquare: Point[] = [
    { x: m - 0.5, y: m - 0.5 },
    { x: m + S - 0.5, y: m - 0.5 },
    { x: m + S - 0.5, y: m + S - 0.5 },
    { x: m - 0.5, y: m + S - 0.5 },
  ]
  const toSrc = homographyFromQuad(coreSquare, c.map(toLocal))
  const toCanon = toSrc === null ? null : invertHomography(toSrc)
  if (toSrc === null || toCanon === null) throw new Error('degenerate piece corners')
  const valid: Mask = { width: N, height: N, data: new Uint8Array(N * N) }
  const rgba = warpRGBA(src, toSrc, N, N, { valid })

  // Mask: outline in canonical coordinates, clipped to what the warp could sample, minus the rim.
  const outline = transformPoints(toCanon, (contour.length > 0 ? contour : c).map(toLocal))
  const raw = rasterizePolygon(outline, N, N)
  for (let i = 0; i < N * N; i++) raw.data[i] &= valid.data[i]
  const mask = erode(raw, rectKernel(2 * RIM_ERODE_PX + 1))
  const lab = rgbaToLab(rgba)

  const maskF = { width: N, height: N, data: new Float32Array(N * N) }
  for (let i = 0; i < N * N; i++) maskF.data[i] = mask.data[i]
  const levels: CanonicalLevel[] = LEVEL_SIZES.map((s) => {
    const n = levelSpan(s)
    const levelMask: Mask = { width: n, height: n, data: new Uint8Array(n * n) }
    const cover = resizeAreaGray(maskF, n, n)
    for (let i = 0; i < n * n; i++) levelMask.data[i] = cover.data[i] >= LEVEL_MASK_MIN ? 1 : 0
    return { size: s, margin: levelMargin(s), lab: resizeAreaLab(lab, n, n), mask: levelMask }
  })
  const finest = levels[levels.length - 1]
  return { size: finest.size, margin: finest.margin, lab: finest.lab, mask: finest.mask, shape, levels }
}

function dist(a: Point, b: Point): number {
  return Math.hypot(b.x - a.x, b.y - a.y)
}
