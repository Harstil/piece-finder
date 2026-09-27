/**
 * Rectangular crops of RGBA images, clipped to the image.
 *
 * The engine works on the part of a large image it needs: the lid area of a 12 MP box photo, or the
 * neighbourhood of one piece in a camera frame. Cropping before any resampling keeps those steps
 * proportional to the object, not the photo.
 */

import type { RGBAImage, Rect } from '../types.ts'

/** Integer crop rectangle covering [x0, x1] × [y0, y1] (inclusive, float), clipped to width × height. */
export function clipRect(x0: number, y0: number, x1: number, y1: number, width: number, height: number): Rect {
  const x = Math.max(0, Math.floor(x0))
  const y = Math.max(0, Math.floor(y0))
  const right = Math.min(width, Math.ceil(x1) + 1)
  const bottom = Math.min(height, Math.ceil(y1) + 1)
  return { x, y, width: Math.max(0, right - x), height: Math.max(0, bottom - y) }
}

/** A copy of `rect` of `src` (rect must already be inside the image, see clipRect). */
export function cropRGBA(src: RGBAImage, rect: Rect): RGBAImage {
  const { x, y, width, height } = rect
  const data = new Uint8ClampedArray(width * height * 4)
  for (let row = 0; row < height; row++) {
    const from = ((y + row) * src.width + x) * 4
    data.set(src.data.subarray(from, from + width * 4), row * width * 4)
  }
  return { width, height, data }
}
