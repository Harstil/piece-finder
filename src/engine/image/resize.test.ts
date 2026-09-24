/**
 * Resize tests: area resampling against cv2.resize INTER_AREA (float gray and 8-bit RGB), bilinear
 * against INTER_LINEAR, plus invariants (constant images stay constant, identity sizes are exact).
 * Tolerances: float area 1e-3 (measured worst 3e-5, i.e. the 5-decimal rounding of the fixture);
 * 8-bit area 1 level (rounding of the 8-bit output); float bilinear 1e-3.
 */

import { describe, expect, it } from 'vitest'
import { golden, grayFrom, rgbaFromRgb } from '../__golden__/fixtures.ts'
import type { LabImage, Mask } from '../types.ts'
import { createGray } from './create.ts'
import {
  fitSize,
  resizeAreaGray,
  resizeAreaLab,
  resizeAreaRGBA,
  resizeBilinearGray,
  resizeBilinearLab,
  resizeBilinearRGBA,
  resizeMaskNearest,
} from './resize.ts'

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length)
  let worst = 0
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]))
  return worst
}

describe('resizeArea', () => {
  const { gray, area, rgb, areaRgb } = golden.resize
  const src = grayFrom(gray.width, gray.height, gray.data)

  it.each(area.map((c) => [`${c.width}x${c.height}`, c] as const))('matches INTER_AREA float %s', (_, c) => {
    const out = resizeAreaGray(src, c.width, c.height)
    expect(maxAbsDiff(out.data, c.data)).toBeLessThan(1e-3)
  })

  it.each(areaRgb.map((c) => [`${c.width}x${c.height}`, c] as const))('matches INTER_AREA uint8 RGB %s', (_, c) => {
    const out = resizeAreaRGBA(rgbaFromRgb(rgb.width, rgb.height, rgb.data), c.width, c.height)
    let worst = 0
    for (let i = 0; i < c.width * c.height; i++) {
      for (let ch = 0; ch < 3; ch++) worst = Math.max(worst, Math.abs(out.data[i * 4 + ch] - c.data[i * 3 + ch]))
      expect(out.data[i * 4 + 3]).toBe(255)
    }
    expect(worst).toBeLessThanOrEqual(1)
  })

  it('keeps a constant image constant for any ratio', () => {
    const img = createGray(97, 61)
    img.data.fill(42.5)
    for (const [w, h] of [[10, 7], [33, 60], [97, 61], [150, 100]]) {
      const out = resizeAreaGray(img, w, h)
      for (const v of out.data) expect(v).toBeCloseTo(42.5, 4)
    }
  })

  it('preserves the mean on an exact integer-ratio downscale', () => {
    const img = grayFrom(6, 4, Array.from({ length: 24 }, (_, i) => i * 3.5))
    const out = resizeAreaGray(img, 3, 2)
    expect(out.data[0]).toBeCloseTo((0 + 1 + 6 + 7) * 3.5 / 4, 5)
    const mean = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v, 0) / a.length
    expect(mean(out.data)).toBeCloseTo(mean(img.data), 4)
  })

  it('resamples all three Lab planes with the same taps', () => {
    const plane = Float32Array.from(src.data)
    const lab: LabImage = { width: src.width, height: src.height, L: plane, a: plane.map((v) => -v), b: plane.map((v) => v * 2) }
    const out = resizeAreaLab(lab, 12, 9)
    const ref = resizeAreaGray(src, 12, 9)
    for (let i = 0; i < ref.data.length; i++) {
      expect(out.L[i]).toBeCloseTo(ref.data[i], 4)
      expect(out.a[i]).toBeCloseTo(-ref.data[i], 4)
      expect(out.b[i]).toBeCloseTo(ref.data[i] * 2, 3)
    }
  })
})

describe('resizeBilinear', () => {
  const { gray, linear } = golden.resize
  const src = grayFrom(gray.width, gray.height, gray.data)

  it.each(linear.map((c) => [`${c.width}x${c.height}`, c] as const))('matches INTER_LINEAR float %s', (_, c) => {
    const out = resizeBilinearGray(src, c.width, c.height)
    expect(maxAbsDiff(out.data, c.data)).toBeLessThan(1e-3)
  })

  it('is the identity at the same size for Gray, Lab and RGBA', () => {
    expect(maxAbsDiff(resizeBilinearGray(src, src.width, src.height).data, src.data)).toBe(0)
    const lab: LabImage = { width: src.width, height: src.height, L: src.data, a: src.data, b: src.data }
    expect(maxAbsDiff(resizeBilinearLab(lab, src.width, src.height).b, src.data)).toBe(0)
    const rgba = rgbaFromRgb(4, 2, [1, 2, 3, 10, 20, 30, 40, 50, 60, 7, 8, 9, 0, 0, 0, 255, 255, 255, 5, 5, 5, 9, 9, 9])
    expect(Array.from(resizeBilinearRGBA(rgba, 4, 2).data)).toEqual(Array.from(rgba.data))
  })
})

describe('resizeMaskNearest', () => {
  it('copies labels with centre alignment', () => {
    const m: Mask = { width: 4, height: 2, data: Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]) }
    expect(Array.from(resizeMaskNearest(m, 2, 1).data)).toEqual([6, 8])
    expect(Array.from(resizeMaskNearest(m, 8, 2).data)).toEqual([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8])
  })
})

describe('fitSize', () => {
  it('fits the longer side and keeps the aspect ratio', () => {
    expect(fitSize(1920, 1080, 512)).toEqual({ width: 512, height: 288 })
    expect(fitSize(1080, 1920, 512)).toEqual({ width: 288, height: 512 })
  })

  it('rejects impossible targets', () => {
    expect(() => resizeAreaGray(createGray(4, 4), 0, 3)).toThrow()
  })
})
