/**
 * Contour tracing against cv2.findContours(CHAIN_APPROX_NONE): after reversing OpenCV's orientation
 * (documented in contour.ts) every contour must be the identical point sequence — same start pixel,
 * same order — for RETR_EXTERNAL, and for all outer borders (RETR_TREE even depths) with
 * externalOnly off. Masks cover holes, a nested blob, touching blobs, 1-pixel lines, isolated pixels,
 * frame contact and random noise.
 */

import { describe, expect, it } from 'vitest'
import { MASK_NAMES, golden, goldenMask, maskFromRows, pointsFrom } from '../__golden__/fixtures.ts'
import type { Point } from '../types.ts'
import { findContours } from './contour.ts'
import { signedArea } from './polygon.ts'

/** OpenCV's counter-clockwise sequence → the clockwise convention: keep p0, reverse the rest. */
function toClockwise(flat: number[]): Point[] {
  const pts = pointsFrom(flat)
  return pts.length <= 1 ? pts : [pts[0], ...pts.slice(1).reverse()]
}

/** Contours keyed by start pixel, since the two libraries list them in different orders. */
function byStart(contours: Point[][]): Map<string, Point[]> {
  return new Map(contours.map((c) => [`${c[0].x},${c[0].y}`, c]))
}

describe('findContours vs OpenCV', () => {
  it.each(MASK_NAMES)('RETR_EXTERNAL on the %s mask', (name) => {
    const ours = findContours(goldenMask(name))
    const theirs = golden.contours[name].external.map(toClockwise)
    expect(ours.length).toBe(theirs.length)
    const map = byStart(ours)
    for (const expected of theirs) expect(map.get(`${expected[0].x},${expected[0].y}`)).toEqual(expected)
  })

  it.each(MASK_NAMES)('every outer border on the %s mask with externalOnly off', (name) => {
    const ours = findContours(goldenMask(name), { externalOnly: false })
    const theirs = golden.contours[name].allOuter.map(toClockwise)
    expect(ours.length).toBe(theirs.length)
    const map = byStart(ours)
    for (const expected of theirs) expect(map.get(`${expected[0].x},${expected[0].y}`)).toEqual(expected)
  })

  it('flips OpenCV orientation: OpenCV areas are negative, ours positive (clockwise)', () => {
    for (const name of MASK_NAMES) {
      const areas = golden.contours[name].externalOrientedArea
      const ours = findContours(goldenMask(name))
      const oursByStart = byStart(ours)
      golden.contours[name].external.forEach((flat, i) => {
        const c = oursByStart.get(`${flat[0]},${flat[1]}`)!
        expect(signedArea(c)).toBeCloseTo(-areas[i], 6)
        if (Math.abs(areas[i]) > 0) expect(signedArea(c)).toBeGreaterThan(0)
      })
    }
  })
})

describe('findContours details', () => {
  it('traces a rectangle clockwise from its top-left pixel', () => {
    const c = findContours(maskFromRows(['00000', '01110', '01110', '00000']))
    expect(c).toEqual([[{ x: 1, y: 1 }, { x: 2, y: 1 }, { x: 3, y: 1 }, { x: 3, y: 2 }, { x: 2, y: 2 }, { x: 1, y: 2 }]])
  })

  it('returns single pixels as one-point contours and lists contours top to bottom', () => {
    const c = findContours(maskFromRows(['0001', '0000', '1000']))
    expect(c).toEqual([[{ x: 3, y: 0 }], [{ x: 0, y: 2 }]])
  })

  it('filters by minimum area', () => {
    const c = findContours(maskFromRows(['1001', '0001', '0001']), { minArea: 2 })
    expect(c.length).toBe(1)
    expect(c[0][0]).toEqual({ x: 3, y: 0 })
  })
})
