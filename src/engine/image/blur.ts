/**
 * Separable Gaussian blur on single-channel float images.
 *
 * Used to pre-smooth before gradients (the structure cue) and to suppress sensor noise and paper texture
 * before comparing a piece against a cell. The kernel is sampled from exp(−x²/2σ²) and normalised to sum
 * to 1; borders are mirrored without repeating the edge pixel (OpenCV's default BORDER_REFLECT_101), so
 * results equal cv2.GaussianBlur with the same kernel radius (golden-tested).
 *
 * Default radius = ceil(3σ): it keeps 99.7 % of the kernel mass and is ~25 % cheaper than OpenCV's float
 * default of round(4σ) (the tail beyond 3σ changes a result by < 0.3 % of the local contrast — guessed
 * negligible for matching). Pass `radius` explicitly to reproduce OpenCV exactly.
 *
 * Two passes: horizontal into a scratch image (each row padded once so the inner loop has no branches),
 * then vertical, which walks whole rows so memory access stays sequential.
 */

import type { GrayImage } from '../types.ts'
import { ensureGray, reflect101, scratchF32 } from './create.ts'

/** Normalised 1-D Gaussian of length 2·radius + 1. */
export function gaussianKernel(sigma: number, radius = Math.ceil(3 * sigma)): Float32Array {
  if (!(sigma > 0)) throw new Error(`Gaussian sigma must be positive, got ${sigma}`)
  const r = Math.max(0, Math.floor(radius))
  const k = new Float32Array(2 * r + 1)
  const denom = 2 * sigma * sigma
  let sum = 0
  for (let i = -r; i <= r; i++) {
    const v = Math.exp(-(i * i) / denom)
    k[i + r] = v
    sum += v
  }
  for (let i = 0; i < k.length; i++) k[i] /= sum
  return k
}

export function gaussianBlur(src: GrayImage, sigma: number, out?: GrayImage, radius?: number): GrayImage {
  const { width: w, height: h } = src
  const kernel = gaussianKernel(sigma, radius)
  const r = (kernel.length - 1) >> 1
  const dst = ensureGray(out, w, h)
  const s = src.data
  const d = dst.data
  const tmp = scratchF32('blur.tmp', w * h)
  const padded = scratchF32('blur.row', w + 2 * r)
  const rowIdx = new Int32Array(2 * r + 1)

  // Horizontal pass.
  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let i = -r; i < w + r; i++) padded[i + r] = s[row + (i >= 0 && i < w ? i : reflect101(i, w))]
    for (let x = 0; x < w; x++) {
      let acc = 0
      for (let k = 0; k <= 2 * r; k++) acc += padded[x + k] * kernel[k]
      tmp[row + x] = acc
    }
  }

  // Vertical pass, row by row.
  for (let y = 0; y < h; y++) {
    for (let k = -r; k <= r; k++) rowIdx[k + r] = reflect101(y + k, h) * w
    const out0 = y * w
    const w0 = kernel[0]
    const r0 = rowIdx[0]
    for (let x = 0; x < w; x++) d[out0 + x] = tmp[r0 + x] * w0
    for (let k = 1; k <= 2 * r; k++) {
      const wk = kernel[k]
      const rk = rowIdx[k]
      for (let x = 0; x < w; x++) d[out0 + x] += tmp[rk + x] * wk
    }
  }
  return dst
}
