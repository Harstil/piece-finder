/**
 * Outer-boundary tracing of binary masks: one closed contour per 8-connected component.
 *
 * A piece's outline is the input to everything shape-related (corner finding, side classification, the
 * canonical warp), so it must be exactly reproducible. This follows the border-following rule of
 * Suzuki & Abe (1985) as OpenCV implements it, so for every component the traced pixels, and their
 * order, equal cv2.findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_NONE) (golden-tested), except:
 *
 * - Orientation: contours here run CLOCKWISE on screen (image coordinates, y down), matching the engine
 *   contract in types.ts. OpenCV returns the same points counter-clockwise. Both start at the
 *   component's first pixel in raster order; the sequence here is OpenCV's reversed after that pixel:
 *   [p0, p(n−1), …, p1]. Clockwise means a positive shoelace area in image coordinates (polygon.ts).
 * - Order of contours: raster order of their first pixels (top to bottom), not OpenCV's.
 *
 * Point convention: points are pixel centres (integers), no repeated closing point. Thin parts are
 * traversed on both sides, so 1-pixel lines and spurs appear twice (out and back), as in OpenCV; a
 * single isolated pixel yields a one-point contour.
 *
 * `externalOnly` (default true) reproduces RETR_EXTERNAL: a component lying inside a hole of another is
 * skipped. With it off, every component gets its outer contour (what OpenCV's RETR_LIST would give for
 * outer borders).
 */

import { labelComponents, markOuterBackground, type ComponentStats, type Components } from '../image/components.ts'
import { scratchI32, scratchU8 } from '../image/create.ts'
import type { Mask, Point } from '../types.ts'

/** Chain-code steps as OpenCV numbers them: 0 = +x, then counter-clockwise on screen (y down). */
const DX = [1, 1, 0, -1, -1, -1, 0, 1]
const DY = [0, -1, -1, -1, 0, 1, 1, 1]

export interface ContourOptions {
  /** Skip components enclosed in another component's hole (cv2.RETR_EXTERNAL). Default true. */
  externalOnly?: boolean
  /** Skip components with fewer pixels than this. Default 1 (keep everything). */
  minArea?: number
}

/**
 * Above this many components the pairwise bounding-box test below costs more than just running the
 * outer-background flood fill (guessed from the measured costs: ~0.5 ms for 512² box tests vs a few
 * ms for the fill on a 1080×1920 mask).
 */
const NESTING_CHECK_LIMIT = 512

/**
 * False when no component's bounding box strictly contains another's. A component inside another's
 * hole always has such a container, so then nothing can be nested and the flood fill is skipped —
 * the common case for a table of separate pieces.
 */
function mayBeNested(stats: readonly ComponentStats[]): boolean {
  if (stats.length > NESTING_CHECK_LIMIT) return true
  for (const inner of stats) {
    const b = inner.bbox
    for (const outer of stats) {
      const o = outer.bbox
      if (o.x < b.x && o.y < b.y && o.x + o.width > b.x + b.width && o.y + o.height > b.y + b.height) return true
    }
  }
  return false
}

let coords = new Int32Array(4096)

/**
 * Traces the outer boundary of component `label` in a label map, starting from its first pixel in
 * raster order (`start`, a row-major index). Returns the contour clockwise (see the file header).
 */
export function traceOuterBoundary(labels: Int32Array, width: number, height: number, label: number, start: number): Point[] {
  const x0 = start % width
  const y0 = (start - x0) / width
  const fg = (x: number, y: number): boolean =>
    x >= 0 && y >= 0 && x < width && y < height && labels[y * width + x] === label

  // First neighbour: search clockwise starting just past the left neighbour (always background here).
  let s = 4
  let found = false
  for (let k = 0; k < 8; k++) {
    s = (s - 1) & 7
    if (fg(x0 + DX[s], y0 + DY[s])) {
      found = true
      break
    }
  }
  if (!found) return [{ x: x0, y: y0 }]
  const x1 = x0 + DX[s]
  const y1 = y0 + DY[s]

  let n = 0
  let x3 = x0
  let y3 = y0
  for (;;) {
    // Next boundary pixel: counter-clockwise from the direction just past the previous pixel.
    let nx = x3
    let ny = y3
    let d = s
    for (let k = 1; k <= 8; k++) {
      d = (s + k) & 7
      nx = x3 + DX[d]
      ny = y3 + DY[d]
      if (fg(nx, ny)) break
    }
    if (n * 2 + 2 > coords.length) {
      const grown = new Int32Array(coords.length * 2)
      grown.set(coords)
      coords = grown
    }
    coords[n * 2] = x3
    coords[n * 2 + 1] = y3
    n++
    if (nx === x0 && ny === y0 && x3 === x1 && y3 === y1) break
    x3 = nx
    y3 = ny
    s = (d + 4) & 7
  }

  // OpenCV's order is counter-clockwise on screen; keep the start pixel and reverse the rest.
  const out: Point[] = new Array(n)
  out[0] = { x: coords[0], y: coords[1] }
  for (let i = 1; i < n; i++) out[i] = { x: coords[(n - i) * 2], y: coords[(n - i) * 2 + 1] }
  return out
}

/** The clockwise outer contour of one component of a labelling (see labelComponents). */
export function traceComponent(components: Components, stat: ComponentStats): Point[] {
  return traceOuterBoundary(components.labels, components.width, components.height, stat.label, stat.start)
}

/**
 * Outer contours of the 8-connected components of the nonzero pixels of `mask`, clockwise, in raster
 * order of each component's first pixel.
 */
export function findContours(mask: Mask, opts: ContourOptions = {}): Point[][] {
  const externalOnly = opts.externalOnly ?? true
  const minArea = opts.minArea ?? 1
  const { width: w, height: h } = mask
  const n = w * h
  const comps = labelComponents(mask, scratchI32('contour.labels', n).subarray(0, n))
  const outer =
    externalOnly && mayBeNested(comps.stats)
      ? markOuterBackground(mask, scratchU8('contour.outer', n).subarray(0, n))
      : null
  const contours: Point[][] = []
  for (const stat of comps.stats) {
    if (stat.area < minArea) continue
    // The pixel left of a component's first pixel is background on its outer side; if that background
    // is enclosed (not connected to the frame), the component sits inside another one's hole.
    if (outer !== null && stat.start % w !== 0 && outer[stat.start - 1] === 0) continue
    contours.push(traceComponent(comps, stat))
  }
  return contours
}
