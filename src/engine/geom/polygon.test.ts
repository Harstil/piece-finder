/**
 * Polygon utilities: approxPolyDP must equal cv2.approxPolyDP(closed=True) point for point on OpenCV's
 * own contours (golden); the rest is checked against closed-form answers (areas, circles, squares) and
 * brute force (rasterisation vs point-in-polygon).
 */

import { describe, expect, it } from 'vitest'
import { golden, pointsFrom } from '../__golden__/fixtures.ts'
import type { Point } from '../types.ts'
import {
  approxPolyDP,
  convexHull,
  isClockwise,
  orientClockwise,
  perimeter,
  pointInPolygon,
  pointSegmentDistance,
  rasterizePolygon,
  resampleClosed,
  signedArea,
  turningAngles,
} from './polygon.ts'

/** Clockwise on screen (y down): top-left → top-right → bottom-right → bottom-left. */
const square: Point[] = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }]

function circle(n: number, r: number, cx = 0, cy = 0): Point[] {
  // Increasing angle with y down runs clockwise on screen.
  return Array.from({ length: n }, (_, i) => ({ x: cx + r * Math.cos((2 * Math.PI * i) / n), y: cy + r * Math.sin((2 * Math.PI * i) / n) }))
}

describe('approxPolyDP vs OpenCV', () => {
  const cases = golden.approx.cases.map((c, i) => [`${i} ${c.mask} eps=${c.epsilon}`, c] as const)
  it.each(cases)('case %s', (_, c) => {
    expect(approxPolyDP(pointsFrom(c.input), c.epsilon)).toEqual(pointsFrom(c.output))
  })

  it('keeps every dropped point within epsilon of the result', () => {
    const pts = circle(200, 50, 60, 60)
    const approx = approxPolyDP(pts, 1.5)
    expect(approx.length).toBeLessThan(40)
    for (const p of pts) {
      let best = Infinity
      for (let i = 0; i < approx.length; i++) best = Math.min(best, pointSegmentDistance(p, approx[i], approx[(i + 1) % approx.length]))
      expect(best).toBeLessThanOrEqual(1.5 + 1e-9)
    }
  })
})

describe('area, perimeter, orientation', () => {
  it('uses positive area for clockwise-on-screen polygons', () => {
    expect(signedArea(square)).toBe(100)
    expect(isClockwise(square)).toBe(true)
    const ccw = [square[0], square[3], square[2], square[1]]
    expect(signedArea(ccw)).toBe(-100)
    expect(orientClockwise(ccw)).toEqual(square)
    expect(orientClockwise(square)).toBe(square)
  })

  it('measures closed and open perimeters', () => {
    expect(perimeter(square)).toBe(40)
    expect(perimeter(square, false)).toBe(30)
  })

  it('measures point-to-segment distances, including beyond the ends', () => {
    const a = { x: 0, y: 0 }
    const b = { x: 10, y: 0 }
    expect(pointSegmentDistance({ x: 5, y: 3 }, a, b)).toBe(3)
    expect(pointSegmentDistance({ x: -3, y: 4 }, a, b)).toBe(5)
    expect(pointSegmentDistance({ x: 1, y: 1 }, a, a)).toBeCloseTo(Math.SQRT2, 12)
  })
})

describe('convexHull', () => {
  it('returns the clockwise hull without interior or collinear points', () => {
    const pts: Point[] = [...square, { x: 5, y: 5 }, { x: 5, y: 0 }, { x: 2, y: 3 }, { x: 10, y: 10 }]
    const hull = convexHull(pts)
    expect(hull).toEqual([{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }])
    expect(isClockwise(hull)).toBe(true)
    expect(convexHull([{ x: 1, y: 1 }, { x: 1, y: 1 }])).toEqual([{ x: 1, y: 1 }])
  })
})

describe('pointInPolygon and rasterizePolygon', () => {
  const star: Point[] = [
    { x: 10, y: 1 }, { x: 13, y: 8 }, { x: 20, y: 8 }, { x: 14.5, y: 12.5 }, { x: 17, y: 19.5 },
    { x: 10, y: 15 }, { x: 3, y: 19.5 }, { x: 5.5, y: 12.5 }, { x: 0.2, y: 8 }, { x: 7, y: 8 },
  ]

  it('agrees with point-in-polygon at every pixel centre away from edges', () => {
    const m = rasterizePolygon(star, 22, 22)
    for (let y = 0; y < 22; y++) {
      for (let x = 0; x < 22; x++) {
        const p = { x, y }
        let nearEdge = false
        for (let i = 0; i < star.length; i++) if (pointSegmentDistance(p, star[i], star[(i + 1) % star.length]) < 1e-9) nearEdge = true
        if (!nearEdge) expect(m.data[y * 22 + x]).toBe(pointInPolygon(p, star) ? 1 : 0)
      }
    }
  })

  it('covers a square exactly once with the half-open rule and uses even-odd for self-overlap', () => {
    const m = rasterizePolygon([{ x: 1, y: 1 }, { x: 4, y: 1 }, { x: 4, y: 3 }, { x: 1, y: 3 }], 6, 5)
    // Centres with 1 ≤ x < 4 and 1 ≤ y < 3.
    expect(m.data.reduce((s, v) => s + v, 0)).toBe(6)
    expect(m.data[1 * 6 + 1]).toBe(1)
    expect(m.data[1 * 6 + 4]).toBe(0)
    const clipped = rasterizePolygon([{ x: -5, y: -5 }, { x: 50, y: -5 }, { x: 50, y: 50 }, { x: -5, y: 50 }], 4, 3)
    expect(clipped.data.every((v) => v === 1)).toBe(true)
    expect(pointInPolygon({ x: 5, y: 5 }, square)).toBe(true)
    expect(pointInPolygon({ x: 15, y: 5 }, square)).toBe(false)
  })
})

describe('resampleClosed and turningAngles', () => {
  it('spaces points evenly by arc length starting at the first vertex', () => {
    const pts = resampleClosed(square, 8)
    expect(pts[0]).toEqual({ x: 0, y: 0 })
    expect(pts[1]).toEqual({ x: 5, y: 0 })
    expect(pts[3]).toEqual({ x: 10, y: 5 })
    expect(pts[7]).toEqual({ x: 0, y: 5 })
  })

  it('gives angle ≈ scale / R on a circle, positive for a clockwise contour', () => {
    const r = 40
    const pts = circle(400, r)
    const angles = turningAngles(pts, 5)
    for (const a of angles) expect(a).toBeCloseTo(5 / r, 2)
    const total = turningAngles(pts, 0.5).reduce((s, a) => s + a, 0)
    expect(total).toBeCloseTo(2 * Math.PI, 3)
  })

  it('finds right-angle corners of a square and straight sides in between', () => {
    const pts = resampleClosed(square, 40)
    const angles = turningAngles(pts, 2)
    expect(angles[0]).toBeCloseTo(Math.PI / 2, 6) // corner (0, 0)
    expect(angles[10]).toBeCloseTo(Math.PI / 2, 6) // corner (10, 0)
    expect(angles[5]).toBeCloseTo(0, 9) // mid-side
    // Counter-clockwise traversal flips the sign.
    const ccw = orientClockwise(pts).slice().reverse()
    expect(turningAngles(ccw, 2)[ccw.length - 1]).toBeCloseTo(-Math.PI / 2, 6)
  })
})
