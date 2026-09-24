/**
 * Integral image tests: window statistics must equal brute-force sums over the same (clipped) windows.
 */

import { describe, expect, it } from 'vitest'
import type { GrayImage } from '../types.ts'
import { integralImage, windowArea, windowMean, windowSum, windowVariance } from './integral.ts'

function randomGray(w: number, h: number, seed: number): GrayImage {
  let s = seed
  const data = new Float32Array(w * h)
  for (let i = 0; i < data.length; i++) {
    s = (s * 1103515245 + 12345) >>> 0
    data[i] = (s >>> 8) % 256
  }
  return { width: w, height: h, data }
}

function brute(img: GrayImage, x: number, y: number, w: number, h: number) {
  let n = 0
  let sum = 0
  let sq = 0
  for (let yy = Math.max(0, y); yy < Math.min(img.height, y + h); yy++) {
    for (let xx = Math.max(0, x); xx < Math.min(img.width, x + w); xx++) {
      const v = img.data[yy * img.width + xx]
      n++
      sum += v
      sq += v * v
    }
  }
  return { n, sum, mean: n ? sum / n : 0, variance: n ? Math.max(0, sq / n - (sum / n) ** 2) : 0 }
}

describe('integral image', () => {
  const img = randomGray(23, 17, 7)
  const ii = integralImage(img)

  it('matches brute force on inside, edge-hanging and outside windows', () => {
    const windows = [
      [0, 0, 23, 17], [3, 4, 5, 6], [-2, -3, 6, 7], [20, 14, 10, 10], [22, 16, 1, 1], [30, 0, 4, 4], [5, 5, 0, 3],
    ]
    for (const [x, y, w, h] of windows) {
      const win = { x, y, width: w, height: h }
      const b = brute(img, x, y, w, h)
      expect(windowArea(ii, win)).toBe(b.n)
      expect(windowSum(ii, win)).toBeCloseTo(b.sum, 6)
      expect(windowMean(ii, win)).toBeCloseTo(b.mean, 6)
      expect(windowVariance(ii, win)).toBeCloseTo(b.variance, 5)
    }
  })

  it('can skip the squared table and reuse an output', () => {
    const plain = integralImage(img, false)
    expect(plain.sqsum).toBeNull()
    expect(() => windowVariance(plain, { x: 0, y: 0, width: 2, height: 2 })).toThrow()
    expect(integralImage(img, true, ii)).toBe(ii)
    expect(() => integralImage({ width: 3, height: 3, data: new Float32Array(9) }, true, ii)).toThrow()
  })
})
