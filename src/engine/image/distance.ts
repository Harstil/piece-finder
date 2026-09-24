/**
 * Exact Euclidean distance transform (Felzenszwalb & Huttenlocher, "Distance Transforms of Sampled
 * Functions", 2012).
 *
 * For every foreground (nonzero) pixel: the distance to the nearest background (zero) pixel, measured
 * between pixel centres; background pixels get 0. Segmentation post-processing uses it to find piece
 * interiors (seeds for splitting touching pieces) and to measure how far a pixel sits from an outline.
 *
 * Two separable passes of the 1-D squared-distance transform (lower envelope of parabolas), first down
 * the columns, then along the rows: O(width × height), exact. Equals cv2.distanceTransform(DIST_L2,
 * DIST_MASK_PRECISE) (golden-tested). Like OpenCV, the area outside the image is not background; a mask
 * without any background pixel yields +Infinity everywhere (OpenCV returns 2^64 there).
 */

import type { GrayImage, Mask } from '../types.ts'
import { ensureGray, scratchF32, scratchF64, scratchI32 } from './create.ts'

/**
 * 1-D squared distance transform of f[0..n) into d[0..n). Sites with f = Infinity are ignored; when
 * there are none, d is Infinity everywhere.
 */
function edt1d(f: Float32Array, d: Float32Array, n: number, v: Int32Array, z: Float64Array): void {
  let k = -1
  for (let q = 0; q < n; q++) {
    const fq = f[q]
    if (fq === Infinity) continue
    if (k < 0) {
      k = 0
      v[0] = q
      z[0] = -Infinity
      z[1] = Infinity
      continue
    }
    const hq = fq + q * q
    let s = (hq - (f[v[k]] + v[k] * v[k])) / (2 * (q - v[k]))
    while (s <= z[k]) {
      k--
      s = (hq - (f[v[k]] + v[k] * v[k])) / (2 * (q - v[k]))
    }
    k++
    v[k] = q
    z[k] = s
    z[k + 1] = Infinity
  }
  if (k < 0) {
    d.fill(Infinity, 0, n)
    return
  }
  let j = 0
  for (let q = 0; q < n; q++) {
    while (z[j + 1] < q) j++
    const dq = q - v[j]
    d[q] = dq * dq + f[v[j]]
  }
}

export function distanceTransform(mask: Mask, out?: GrayImage): GrayImage {
  const { width: w, height: h, data: m } = mask
  const dst = ensureGray(out, w, h)
  const o = dst.data
  const n = Math.max(w, h)
  const f = scratchF32('distance.f', n)
  const d = scratchF32('distance.d', n)
  const v = scratchI32('distance.v', n)
  const z = scratchF64('distance.z', n + 1)
  const cols = scratchF32('distance.cols', w * h)

  // Columns: squared distance to the nearest background pixel in the same column.
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = m[y * w + x] === 0 ? 0 : Infinity
    edt1d(f, d, h, v, z)
    for (let y = 0; y < h; y++) cols[y * w + x] = d[y]
  }
  // Rows: combine the column results.
  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) f[x] = cols[row + x]
    edt1d(f, d, w, v, z)
    for (let x = 0; x < w; x++) o[row + x] = Math.sqrt(d[x])
  }
  return dst
}
