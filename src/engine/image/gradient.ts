/**
 * Image gradients: 3×3 Sobel derivatives, gradient magnitude and orientation.
 *
 * The structure cue compares gradient-orientation maps of a piece and a cell because orientations survive
 * the lighting and white-balance differences between the box photo and the camera frame far better than
 * raw intensities do.
 *
 * Sobel matches cv2.Sobel(src, CV_32F, 1, 0 / 0, 1, ksize=3) with OpenCV's default BORDER_REFLECT_101
 * (golden-tested): dx = [−1 0 1] horizontally ⊗ [1 2 1] vertically, unscaled, so a unit-per-pixel ramp
 * gives 8. Orientation is atan2(dy, dx) in radians in (−π, π], measured in image coordinates (y down):
 * 0 points right (+x), +π/2 points down (+y) — i.e. positive angles turn clockwise on screen.
 */

import type { GrayImage } from '../types.ts'
import { ensureGray, reflect101, scratchI32 } from './create.ts'

export interface Gradients {
  dx: GrayImage
  dy: GrayImage
}

export function sobel(src: GrayImage, outDx?: GrayImage, outDy?: GrayImage): Gradients {
  const { width: w, height: h } = src
  const dx = ensureGray(outDx, w, h)
  const dy = ensureGray(outDy, w, h)
  const s = src.data
  const gx = dx.data
  const gy = dy.data
  // Neighbour column indices with the border already reflected, so the pixel loop has no branches.
  const xm = scratchI32('gradient.xm', w)
  const xp = scratchI32('gradient.xp', w)
  for (let x = 0; x < w; x++) {
    xm[x] = reflect101(x - 1, w)
    xp[x] = reflect101(x + 1, w)
  }
  for (let y = 0; y < h; y++) {
    const rm = reflect101(y - 1, h) * w
    const r0 = y * w
    const rp = reflect101(y + 1, h) * w
    for (let x = 0; x < w; x++) {
      const a = xm[x]
      const c = xp[x]
      const tl = s[rm + a]
      const tc = s[rm + x]
      const tr = s[rm + c]
      const bl = s[rp + a]
      const bc = s[rp + x]
      const br = s[rp + c]
      gx[r0 + x] = tr + 2 * s[r0 + c] + br - (tl + 2 * s[r0 + a] + bl)
      gy[r0 + x] = bl + 2 * bc + br - (tl + 2 * tc + tr)
    }
  }
  return { dx, dy }
}

/** sqrt(dx² + dy²) per pixel. */
export function gradientMagnitude(g: Gradients, out?: GrayImage): GrayImage {
  const { width, height } = g.dx
  const dst = ensureGray(out, width, height)
  const x = g.dx.data
  const y = g.dy.data
  const d = dst.data
  for (let i = 0; i < d.length; i++) d[i] = Math.sqrt(x[i] * x[i] + y[i] * y[i])
  return dst
}

/** atan2(dy, dx) per pixel, radians in (−π, π], clockwise-positive in image coordinates (see header). */
export function gradientOrientation(g: Gradients, out?: GrayImage): GrayImage {
  const { width, height } = g.dx
  const dst = ensureGray(out, width, height)
  const x = g.dx.data
  const y = g.dy.data
  const d = dst.data
  for (let i = 0; i < d.length; i++) d[i] = Math.atan2(y[i], x[i])
  return dst
}
