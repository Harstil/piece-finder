/**
 * Distance transform against cv2.distanceTransform(DIST_L2, DIST_MASK_PRECISE) and a brute-force
 * nearest-background search. Tolerance 1e-4 (the fixture is rounded to 5 decimals).
 */

import { describe, expect, it } from 'vitest'
import { MASK_NAMES, golden, goldenMask, maskFromRows } from '../__golden__/fixtures.ts'
import type { Mask } from '../types.ts'
import { distanceTransform } from './distance.ts'

function bruteForce(mask: Mask): Float64Array {
  const { width: w, height: h, data } = mask
  const out = new Float64Array(w * h)
  for (let i = 0; i < w * h; i++) {
    if (data[i] === 0) continue
    let best = Infinity
    for (let j = 0; j < w * h; j++) {
      if (data[j] !== 0) continue
      const dx = (i % w) - (j % w)
      const dy = Math.floor(i / w) - Math.floor(j / w)
      best = Math.min(best, dx * dx + dy * dy)
    }
    out[i] = Math.sqrt(best)
  }
  return out
}

describe('distanceTransform', () => {
  it.each(MASK_NAMES.filter((n) => golden.distance[n] !== undefined))('matches OpenCV on the %s mask', (name) => {
    const expected = golden.distance[name]!
    const out = distanceTransform(goldenMask(name))
    let worst = 0
    for (let i = 0; i < expected.length; i++) worst = Math.max(worst, Math.abs(out.data[i] - expected[i]))
    expect(worst).toBeLessThan(1e-4)
  })

  it('equals brute force on a random mask', () => {
    let s = 99
    const data = new Uint8Array(31 * 23)
    for (let i = 0; i < data.length; i++) {
      s = (s * 1103515245 + 12345) >>> 0
      data[i] = (s >>> 16) % 13 === 0 ? 0 : 1
    }
    const mask = { width: 31, height: 23, data }
    const out = distanceTransform(mask)
    const ref = bruteForce(mask)
    for (let i = 0; i < ref.length; i++) expect(out.data[i]).toBeCloseTo(ref[i], 5)
  })

  it('returns Infinity when there is no background, and 0 on background', () => {
    const out = distanceTransform(maskFromRows(['111', '111']))
    expect(Array.from(out.data).every((v) => v === Infinity)).toBe(true)
    const one = distanceTransform(maskFromRows(['011']))
    expect(Array.from(one.data)).toEqual([0, 1, 2])
  })
})
