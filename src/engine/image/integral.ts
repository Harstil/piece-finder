/**
 * Integral images (summed-area tables) with O(1) window sum, mean and variance.
 *
 * The matcher normalises patches (ZNCC needs local mean and variance) and the classical segmenter
 * measures local texture energy; both need statistics over many overlapping windows, which an integral
 * image answers in four lookups each regardless of window size.
 *
 * Layout: (width + 1) × (height + 1), row-major, with a zero first row and column, so
 * sum[y · (width + 1) + x] = Σ src over [0, x) × [0, y). Float64 because squared sums of a 1920×1080
 * 8-bit frame reach ~1.3e11, far past float32's 24-bit mantissa.
 *
 * Windows are half-open rectangles (x, y, width, height) clipped to the image; statistics are over the
 * clipped area, so windows hanging off the edge still give meaningful values near borders.
 */

import type { Rect } from '../types.ts'

export interface IntegralImage {
  width: number
  height: number
  /** (width + 1) × (height + 1) running sums. */
  sum: Float64Array
  /** Running sums of squares, or null when built without them. */
  sqsum: Float64Array | null
}

/** Anything row-major with numeric samples: GrayImage, Mask, or a label plane. */
interface Plane {
  width: number
  height: number
  data: ArrayLike<number>
}

export function integralImage(src: Plane, withSquares = true, out?: IntegralImage): IntegralImage {
  const { width: w, height: h, data } = src
  const stride = w + 1
  const size = stride * (h + 1)
  let dst = out
  if (dst === undefined) {
    dst = { width: w, height: h, sum: new Float64Array(size), sqsum: withSquares ? new Float64Array(size) : null }
  } else if (dst.width !== w || dst.height !== h || dst.sum.length !== size || (withSquares && dst.sqsum === null)) {
    throw new Error(`integral output does not fit a ${w}x${h} image`)
  }
  const sum = dst.sum
  const sq = withSquares ? dst.sqsum : null
  sum.fill(0, 0, stride)
  if (sq !== null) sq.fill(0, 0, stride)
  for (let y = 0; y < h; y++) {
    const above = y * stride
    const cur = above + stride
    sum[cur] = 0
    let rowSum = 0
    if (sq === null) {
      for (let x = 0; x < w; x++) {
        rowSum += data[y * w + x]
        sum[cur + x + 1] = sum[above + x + 1] + rowSum
      }
    } else {
      sq[cur] = 0
      let rowSq = 0
      for (let x = 0; x < w; x++) {
        const v = data[y * w + x]
        rowSum += v
        rowSq += v * v
        sum[cur + x + 1] = sum[above + x + 1] + rowSum
        sq[cur + x + 1] = sq[above + x + 1] + rowSq
      }
    }
  }
  return dst
}

/** Clips a window to the image; returns false when nothing is left. Writes the corners into `c`. */
function clip(ii: IntegralImage, win: Rect, c: Int32Array): boolean {
  const x0 = Math.max(0, Math.floor(win.x))
  const y0 = Math.max(0, Math.floor(win.y))
  const x1 = Math.min(ii.width, Math.floor(win.x + win.width))
  const y1 = Math.min(ii.height, Math.floor(win.y + win.height))
  if (x1 <= x0 || y1 <= y0) return false
  c[0] = x0
  c[1] = y0
  c[2] = x1
  c[3] = y1
  return true
}

const corners = new Int32Array(4)

function boxSum(table: Float64Array, stride: number, c: Int32Array): number {
  return table[c[3] * stride + c[2]] - table[c[1] * stride + c[2]] - table[c[3] * stride + c[0]] + table[c[1] * stride + c[0]]
}

/** Number of pixels of the window that lie inside the image. */
export function windowArea(ii: IntegralImage, win: Rect): number {
  if (!clip(ii, win, corners)) return 0
  return (corners[2] - corners[0]) * (corners[3] - corners[1])
}

export function windowSum(ii: IntegralImage, win: Rect): number {
  if (!clip(ii, win, corners)) return 0
  return boxSum(ii.sum, ii.width + 1, corners)
}

/** Mean over the clipped window; 0 for a window entirely outside the image. */
export function windowMean(ii: IntegralImage, win: Rect): number {
  if (!clip(ii, win, corners)) return 0
  const n = (corners[2] - corners[0]) * (corners[3] - corners[1])
  return boxSum(ii.sum, ii.width + 1, corners) / n
}

/** Population variance over the clipped window (never negative); needs an integral built with squares. */
export function windowVariance(ii: IntegralImage, win: Rect): number {
  if (ii.sqsum === null) throw new Error('windowVariance needs an integral image built with squares')
  if (!clip(ii, win, corners)) return 0
  const n = (corners[2] - corners[0]) * (corners[3] - corners[1])
  const stride = ii.width + 1
  const mean = boxSum(ii.sum, stride, corners) / n
  const v = boxSum(ii.sqsum, stride, corners) / n - mean * mean
  return v > 0 ? v : 0
}
