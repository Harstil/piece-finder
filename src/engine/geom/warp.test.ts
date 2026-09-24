/**
 * Warp tests against cv2.warpPerspective(INTER_LINEAR | WARP_INVERSE_MAP, BORDER_CONSTANT 0) on a smooth
 * 8-bit image. Compared only where the sample lies at least one pixel inside the source (OpenCV blends
 * border colour closer to the edge, see warp.ts): within 1 grey level of OpenCV's 8-bit output and
 * 0.5 of its float output (OpenCV quantises sample positions to 1/32 px). Plus the validity mask, the
 * quad → rectangle convention, and agreement between the Gray, Lab, RGBA and mask variants.
 */

import { describe, expect, it } from 'vitest'
import { golden } from '../__golden__/fixtures.ts'
import { createMask } from '../image/create.ts'
import type { GrayImage, LabImage, Mask, RGBAImage } from '../types.ts'
import { applyHomography, identityHomography } from './homography.ts'
import { rectToQuadHomography, warpGray, warpLab, warpMask, warpRGBA } from './warp.ts'

const { image, warps } = golden.warp
const src: GrayImage = { width: image.width, height: image.height, data: Float32Array.from(image.data) }

/** 0 = sample outside the source, 1 = within a pixel of its edge, 2 = well inside. */
function sampleZone(H: Float64Array, x: number, y: number): number {
  const p = applyHomography(H, { x, y })
  const inside = p.x >= 0 && p.y >= 0 && p.x <= src.width - 1 && p.y <= src.height - 1
  if (!inside) return 0
  return p.x >= 1 && p.y >= 1 && p.x <= src.width - 2 && p.y <= src.height - 2 ? 2 : 1
}

describe('warpGray vs OpenCV', () => {
  it.each(warps.map((w, i) => [i, w] as const))('case %i', (_, c) => {
    const H = Float64Array.from(c.dstToSrc)
    const valid = createMask(c.width, c.height)
    const out = warpGray(src, H, c.width, c.height, { valid })
    let worst8 = 0
    let worstF = 0
    let compared = 0
    for (let y = 0; y < c.height; y++) {
      for (let x = 0; x < c.width; x++) {
        const i = y * c.width + x
        const zone = sampleZone(H, x, y)
        expect(valid.data[i]).toBe(zone === 0 ? 0 : 1)
        if (zone === 0) {
          expect(out.data[i]).toBe(0)
          continue
        }
        if (zone !== 2) continue
        compared++
        worst8 = Math.max(worst8, Math.abs(out.data[i] - c.out8[i]))
        if (c.outF !== null) worstF = Math.max(worstF, Math.abs(out.data[i] - c.outF[i]))
      }
    }
    expect(compared).toBeGreaterThan(c.width * c.height * 0.3)
    expect(worst8).toBeLessThanOrEqual(1)
    expect(worstF).toBeLessThanOrEqual(0.5)
  })

  it('rebuilds the fixture homographies from their quads (outer-corner convention)', () => {
    for (const c of warps.slice(0, 4)) {
      const H = Float64Array.from(c.dstToSrc)
      const quad = [
        applyHomography(H, { x: -0.5, y: -0.5 }),
        applyHomography(H, { x: c.width - 0.5, y: -0.5 }),
        applyHomography(H, { x: c.width - 0.5, y: c.height - 0.5 }),
        applyHomography(H, { x: -0.5, y: c.height - 0.5 }),
      ]
      const rebuilt = rectToQuadHomography(quad, c.width, c.height)!
      for (let k = 0; k < 9; k++) expect(rebuilt[k]).toBeCloseTo(H[k], 6)
    }
  })
})

describe('warp variants', () => {
  it('is exact for the identity and marks every pixel valid', () => {
    const valid = createMask(src.width, src.height)
    const out = warpGray(src, identityHomography(), src.width, src.height, { valid })
    expect(Array.from(out.data)).toEqual(Array.from(src.data))
    expect(valid.data.every((v) => v === 1)).toBe(true)
  })

  it('warps Lab planes and RGBA channels exactly like the Gray version', () => {
    const H = Float64Array.from(warps[1].dstToSrc)
    const ref = warpGray(src, H, 24, 24)
    const lab: LabImage = { width: src.width, height: src.height, L: src.data, a: src.data.map((v) => -v), b: src.data }
    const outLab = warpLab(lab, H, 24, 24)
    const rgba: RGBAImage = { width: src.width, height: src.height, data: new Uint8ClampedArray(src.width * src.height * 4) }
    for (let i = 0; i < src.data.length; i++) rgba.data.fill(src.data[i], i * 4, i * 4 + 4)
    const outRgba = warpRGBA(rgba, H, 24, 24)
    for (let i = 0; i < ref.data.length; i++) {
      expect(outLab.L[i]).toBeCloseTo(ref.data[i], 4)
      expect(outLab.a[i]).toBeCloseTo(-ref.data[i], 4)
      expect(Math.abs(outRgba.data[i * 4 + 1] - ref.data[i])).toBeLessThanOrEqual(0.5 + 1e-6)
    }
  })

  it('warps masks by thresholding bilinear coverage at 0.5', () => {
    const m: Mask = { width: 4, height: 1, data: Uint8Array.from([0, 0, 9, 9]) }
    // Samples at x = 0.4, 1.4, 2.4 → coverage 0, 0.4, 1; at 0.6, 1.6, 2.6 → 0, 0.6, 1.
    const shift = Float64Array.of(1, 0, 0.4, 0, 1, 0, 0, 0, 1)
    expect(Array.from(warpMask(m, shift, 3, 1).data)).toEqual([0, 0, 1])
    const further = Float64Array.of(1, 0, 0.6, 0, 1, 0, 0, 0, 1)
    expect(Array.from(warpMask(m, further, 3, 1).data)).toEqual([0, 1, 1])
  })

  it('treats points behind the projection centre as invalid', () => {
    const flip = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, -1)
    const valid = createMask(3, 3)
    warpGray(src, flip, 3, 3, { valid })
    expect(valid.data.every((v) => v === 0)).toBe(true)
  })
})
