/**
 * Image resizing: area (box-filter) resampling, bilinear resampling and nearest-neighbour masks.
 *
 * Area resampling is what every downscale in the engine uses (camera frame → 512 px segmentation input,
 * reference cells → 16/32/64 px descriptor pyramid): each output pixel is the exact average of the source
 * area it covers, so fine texture averages out instead of aliasing. It matches cv2.resize INTER_AREA for
 * downscales of any ratio, including non-integer and per-axis different ratios (golden-tested). For
 * upscales the same box-overlap rule is applied, which is not what OpenCV does (it switches to a
 * bilinear variant there) — use the bilinear functions for upscaling.
 *
 * Bilinear resampling follows cv2.resize INTER_LINEAR on float images: pixel centres are aligned
 * (src = (dst + 0.5) · scale − 0.5) and coordinates are clamped at the image edge.
 *
 * Both are separable: a horizontal pass produces resampled source rows, and each output row is a
 * weighted sum of a few of them. Horizontally resampled rows live in a small ring buffer (only the rows
 * the current output row needs), so no full-size intermediate image is allocated.
 *
 * Masks use nearest neighbour with centre alignment (src = floor((dst + 0.5) · scale)), i.e. OpenCV's
 * INTER_NEAREST_EXACT, because averaging labels is meaningless.
 */

import type { GrayImage, LabImage, Mask, RGBAImage } from '../types.ts'
import { ensureGray, ensureLab, ensureMask, ensureRGBA, scratchF32, scratchI32 } from './create.ts'

/** Sparse 1-D resampling operator: output i reads index[start[i] .. start[i+1]) with matching weights. */
interface Taps {
  start: Int32Array
  index: Int32Array
  weight: Float32Array
  /** Largest number of taps of any output sample. */
  maxTaps: number
}

/**
 * Box overlaps below this fraction of a source pixel are dropped: they only arise from floating-point
 * rounding of the box edges (a real overlap is a multiple of 1/dstN px). Guessed; OpenCV uses 1e-3.
 */
const MIN_OVERLAP = 1e-9

function areaTaps(srcN: number, dstN: number): Taps {
  const scale = srcN / dstN
  let bound = 0
  for (let i = 0; i < dstN; i++) bound += Math.min(srcN, Math.ceil((i + 1) * scale)) - Math.floor(i * scale)
  const start = new Int32Array(dstN + 1)
  const index = new Int32Array(bound)
  const weight = new Float32Array(bound)
  let maxTaps = 0
  let t = 0
  for (let i = 0; i < dstN; i++) {
    const a = i * scale
    const b = Math.min((i + 1) * scale, srcN)
    start[i] = t
    for (let k = Math.floor(a); k < Math.ceil(b); k++) {
      const overlap = Math.min(b, k + 1) - Math.max(a, k)
      if (overlap <= MIN_OVERLAP) continue
      index[t] = k
      weight[t] = overlap / (b - a)
      t++
    }
    maxTaps = Math.max(maxTaps, t - start[i])
  }
  start[dstN] = t
  return { start, index, weight, maxTaps }
}

function linearTaps(srcN: number, dstN: number): Taps {
  const scale = srcN / dstN
  const start = new Int32Array(dstN + 1)
  const index = new Int32Array(dstN * 2)
  const weight = new Float32Array(dstN * 2)
  for (let i = 0; i < dstN; i++) {
    const s = (i + 0.5) * scale - 0.5
    let i0 = Math.floor(s)
    let f = s - i0
    if (i0 < 0) {
      i0 = 0
      f = 0
    } else if (i0 >= srcN - 1) {
      i0 = srcN - 1
      f = 0
    }
    start[i] = i * 2
    index[i * 2] = i0
    index[i * 2 + 1] = Math.min(i0 + 1, srcN - 1)
    weight[i * 2] = 1 - f
    weight[i * 2 + 1] = f
  }
  start[dstN] = dstN * 2
  return { start, index, weight, maxTaps: 2 }
}

/**
 * Separable resampling of a single plane (channels = 1) or interleaved RGBA (channels = 4). `dst`
 * receives the result; a Uint8ClampedArray destination rounds and clamps on assignment.
 */
function resample(
  src: ArrayLike<number>,
  sw: number,
  channels: 1 | 4,
  xt: Taps,
  yt: Taps,
  dst: Float32Array | Uint8ClampedArray,
  dw: number,
  dh: number,
): void {
  const rowLen = dw * channels
  const ring = yt.maxTaps + 1
  const rows = scratchF32('resize.rows', ring * rowLen)
  const tags = scratchI32('resize.tags', ring)
  tags.fill(-1, 0, ring)
  const acc = scratchF32('resize.acc', rowLen)
  const xs = xt.start
  const xi = xt.index
  const xw = xt.weight

  for (let dy = 0; dy < dh; dy++) {
    acc.fill(0, 0, rowLen)
    for (let t = yt.start[dy]; t < yt.start[dy + 1]; t++) {
      const sy = yt.index[t]
      const wy = yt.weight[t]
      const slot = sy % ring
      const rowOff = slot * rowLen
      if (tags[slot] !== sy) {
        // Horizontal pass for source row sy, accumulating in locals.
        tags[slot] = sy
        const srcRow = sy * sw * channels
        if (channels === 1) {
          for (let dx = 0; dx < dw; dx++) {
            let a = 0
            for (let k = xs[dx]; k < xs[dx + 1]; k++) a += src[srcRow + xi[k]] * xw[k]
            rows[rowOff + dx] = a
          }
        } else {
          for (let dx = 0; dx < dw; dx++) {
            let a0 = 0
            let a1 = 0
            let a2 = 0
            let a3 = 0
            for (let k = xs[dx]; k < xs[dx + 1]; k++) {
              const w = xw[k]
              const s = srcRow + xi[k] * 4
              a0 += src[s] * w
              a1 += src[s + 1] * w
              a2 += src[s + 2] * w
              a3 += src[s + 3] * w
            }
            const o = rowOff + dx * 4
            rows[o] = a0
            rows[o + 1] = a1
            rows[o + 2] = a2
            rows[o + 3] = a3
          }
        }
      }
      for (let i = 0; i < rowLen; i++) acc[i] += rows[rowOff + i] * wy
    }
    const out = dy * rowLen
    for (let i = 0; i < rowLen; i++) dst[out + i] = acc[i]
  }
}

function checkTarget(width: number, height: number): void {
  if (!(width >= 1 && height >= 1) || !Number.isInteger(width) || !Number.isInteger(height)) {
    throw new Error(`resize target must be a positive integer size, got ${width}x${height}`)
  }
}

/** Largest size with the same aspect ratio whose longer side is `maxSide` (rounded, at least 1 px). */
export function fitSize(width: number, height: number, maxSide: number): { width: number; height: number } {
  const s = maxSide / Math.max(width, height)
  return { width: Math.max(1, Math.round(width * s)), height: Math.max(1, Math.round(height * s)) }
}

export function resizeAreaGray(src: GrayImage, width: number, height: number, out?: GrayImage): GrayImage {
  checkTarget(width, height)
  const dst = ensureGray(out, width, height)
  const xt = areaTaps(src.width, width)
  const yt = areaTaps(src.height, height)
  resample(src.data, src.width, 1, xt, yt, dst.data, width, height)
  return dst
}

export function resizeAreaLab(src: LabImage, width: number, height: number, out?: LabImage): LabImage {
  checkTarget(width, height)
  const dst = ensureLab(out, width, height)
  const xt = areaTaps(src.width, width)
  const yt = areaTaps(src.height, height)
  resample(src.L, src.width, 1, xt, yt, dst.L, width, height)
  resample(src.a, src.width, 1, xt, yt, dst.a, width, height)
  resample(src.b, src.width, 1, xt, yt, dst.b, width, height)
  return dst
}

/** All four channels are averaged, alpha included; results are rounded to 8 bits. */
export function resizeAreaRGBA(src: RGBAImage, width: number, height: number, out?: RGBAImage): RGBAImage {
  checkTarget(width, height)
  const dst = ensureRGBA(out, width, height)
  const xt = areaTaps(src.width, width)
  const yt = areaTaps(src.height, height)
  resample(src.data, src.width, 4, xt, yt, dst.data, width, height)
  return dst
}

export function resizeBilinearGray(src: GrayImage, width: number, height: number, out?: GrayImage): GrayImage {
  checkTarget(width, height)
  const dst = ensureGray(out, width, height)
  const xt = linearTaps(src.width, width)
  const yt = linearTaps(src.height, height)
  resample(src.data, src.width, 1, xt, yt, dst.data, width, height)
  return dst
}

export function resizeBilinearLab(src: LabImage, width: number, height: number, out?: LabImage): LabImage {
  checkTarget(width, height)
  const dst = ensureLab(out, width, height)
  const xt = linearTaps(src.width, width)
  const yt = linearTaps(src.height, height)
  resample(src.L, src.width, 1, xt, yt, dst.L, width, height)
  resample(src.a, src.width, 1, xt, yt, dst.a, width, height)
  resample(src.b, src.width, 1, xt, yt, dst.b, width, height)
  return dst
}

export function resizeBilinearRGBA(src: RGBAImage, width: number, height: number, out?: RGBAImage): RGBAImage {
  checkTarget(width, height)
  const dst = ensureRGBA(out, width, height)
  const xt = linearTaps(src.width, width)
  const yt = linearTaps(src.height, height)
  resample(src.data, src.width, 4, xt, yt, dst.data, width, height)
  return dst
}

/** Nearest-neighbour mask resize with centre alignment; label values are copied unchanged. */
export function resizeMaskNearest(src: Mask, width: number, height: number, out?: Mask): Mask {
  checkTarget(width, height)
  const dst = ensureMask(out, width, height)
  const sx = scratchI32('resize.nearestX', width)
  const scaleX = src.width / width
  const scaleY = src.height / height
  for (let x = 0; x < width; x++) sx[x] = Math.min(src.width - 1, Math.floor((x + 0.5) * scaleX))
  const s = src.data
  const d = dst.data
  for (let y = 0; y < height; y++) {
    const srcRow = Math.min(src.height - 1, Math.floor((y + 0.5) * scaleY)) * src.width
    const dstRow = y * width
    for (let x = 0; x < width; x++) d[dstRow + x] = s[srcRow + sx[x]]
  }
  return dst
}
