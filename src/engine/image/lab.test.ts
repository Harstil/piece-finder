/**
 * Lab / luma conversion against OpenCV's float Lab (cv2.cvtColor on float32 RGB/255) and RGB2GRAY.
 * Tolerances: Lab within 0.5 per channel of OpenCV — OpenCV itself interpolates a 33³ table and is up to
 * 0.44 off the exact formula (measured, see lab.ts) — and within 5e-3 of the exact formula, which is what
 * the per-pixel table interpolation here must preserve. Luma within 0.5 of OpenCV's rounded 8-bit grey.
 */

import { describe, expect, it } from 'vitest'
import { golden, rgbaFromRgb } from '../__golden__/fixtures.ts'
import { rgbaToGray, rgbaToLab, srgbToLab } from './lab.ts'

const LAB_TOLERANCE = 0.5

describe('rgbaToLab', () => {
  it('matches OpenCV float Lab on cube corners, a grey ramp, dark colours and random colours', () => {
    const { rgb, lab } = golden.lab
    const n = rgb.length / 3
    const img = rgbaFromRgb(n, 1, rgb)
    const out = rgbaToLab(img)
    let worst = 0
    for (let i = 0; i < n; i++) {
      worst = Math.max(
        worst,
        Math.abs(out.L[i] - lab[i * 3]),
        Math.abs(out.a[i] - lab[i * 3 + 1]),
        Math.abs(out.b[i] - lab[i * 3 + 2]),
      )
    }
    expect(worst).toBeLessThan(LAB_TOLERANCE)
  })

  it('agrees with the exact per-colour conversion to within the table error', () => {
    // A 17³ lattice over the RGB cube (steps of 16, clamped to 255), dark corner included.
    const rgb: number[] = []
    for (let r = 0; r <= 256; r += 16)
      for (let g = 0; g <= 256; g += 16)
        for (let b = 0; b <= 256; b += 16) rgb.push(Math.min(r, 255), Math.min(g, 255), Math.min(b, 255))
    const n = rgb.length / 3
    const out = rgbaToLab(rgbaFromRgb(n, 1, rgb))
    let worstL = 0
    let worstAB = 0
    for (let i = 0; i < n; i++) {
      const [L, a, b] = srgbToLab(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2])
      worstL = Math.max(worstL, Math.abs(out.L[i] - L))
      worstAB = Math.max(worstAB, Math.abs(out.a[i] - a), Math.abs(out.b[i] - b))
    }
    expect(worstL).toBeLessThan(1e-3)
    expect(worstAB).toBeLessThan(5e-3)
  })

  it('maps white to L=100 and black to L=0 with neutral chroma', () => {
    const [Lw, aw, bw] = srgbToLab(255, 255, 255)
    expect(Lw).toBeCloseTo(100, 3)
    expect(Math.abs(aw)).toBeLessThan(1e-3)
    expect(Math.abs(bw)).toBeLessThan(1e-3)
    expect(srgbToLab(0, 0, 0)[0]).toBeCloseTo(0, 6)
  })

  it('writes into a provided output and rejects a wrongly sized one', () => {
    const img = rgbaFromRgb(2, 1, [255, 0, 0, 0, 0, 255])
    const out = { width: 2, height: 1, L: new Float32Array(2), a: new Float32Array(2), b: new Float32Array(2) }
    expect(rgbaToLab(img, out)).toBe(out)
    expect(out.a[0]).toBeGreaterThan(70) // red is strongly +a
    expect(out.b[1]).toBeLessThan(-100) // blue is strongly -b
    expect(() => rgbaToLab(img, { ...out, width: 3 })).toThrow()
  })
})

describe('rgbaToGray', () => {
  it('matches cv2.COLOR_RGB2GRAY before rounding', () => {
    const { rgb, gray8 } = golden.lab
    const n = rgb.length / 3
    const out = rgbaToGray(rgbaFromRgb(n, 1, rgb))
    for (let i = 0; i < n; i++) expect(Math.abs(out.data[i] - gray8[i])).toBeLessThanOrEqual(0.5 + 1e-4)
  })
})
