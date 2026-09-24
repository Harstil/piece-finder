/**
 * Gaussian blur and Sobel against cv2.GaussianBlur / cv2.Sobel on the same float image (golden), plus
 * kernel and orientation invariants. Tolerance 2e-3 grey levels: float32 accumulation order differs
 * from OpenCV's and the fixture is rounded to 5 decimals.
 */

import { describe, expect, it } from 'vitest'
import { golden, grayFrom } from '../__golden__/fixtures.ts'
import { gaussianBlur, gaussianKernel } from './blur.ts'
import { createGray } from './create.ts'
import { gradientMagnitude, gradientOrientation, sobel } from './gradient.ts'

const TOLERANCE = 2e-3

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let worst = 0
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]))
  return worst
}

const { image } = golden.filters
const src = grayFrom(image.width, image.height, image.data)

describe('gaussianBlur', () => {
  it.each(golden.filters.blur.map((c) => [c.sigma, c] as const))('matches cv2.GaussianBlur, sigma %s', (_, c) => {
    const out = gaussianBlur(src, c.sigma, undefined, c.radius)
    expect(maxAbsDiff(out.data, c.data)).toBeLessThan(TOLERANCE)
  })

  it('works in place and keeps constants constant', () => {
    const img = createGray(9, 5)
    img.data.fill(7)
    expect(gaussianBlur(img, 2.5, img)).toBe(img)
    for (const v of img.data) expect(v).toBeCloseTo(7, 5)
  })

  it('builds a normalised, symmetric kernel of radius ceil(3 sigma) by default', () => {
    const k = gaussianKernel(1.5)
    expect(k.length).toBe(11)
    expect(k.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 6)
    expect(k[0]).toBeCloseTo(k[10], 7)
    expect(() => gaussianKernel(0)).toThrow()
  })
})

describe('sobel', () => {
  it('matches cv2.Sobel dx and dy', () => {
    const { dx, dy } = sobel(src)
    expect(maxAbsDiff(dx.data, golden.filters.sobelDx)).toBeLessThan(TOLERANCE)
    expect(maxAbsDiff(dy.data, golden.filters.sobelDy)).toBeLessThan(TOLERANCE)
  })

  it('gives 8 per unit slope and orientation clockwise from +x in image coordinates', () => {
    const w = 6
    const h = 5
    const down = grayFrom(w, h, Array.from({ length: w * h }, (_, i) => Math.floor(i / w)))
    const g = sobel(down)
    const mag = gradientMagnitude(g)
    const ori = gradientOrientation(g)
    const centre = 2 * w + 3
    expect(g.dx.data[centre]).toBe(0)
    expect(g.dy.data[centre]).toBe(8)
    expect(mag.data[centre]).toBe(8)
    expect(ori.data[centre]).toBeCloseTo(Math.PI / 2, 6) // intensity grows downwards → +π/2
  })
})
