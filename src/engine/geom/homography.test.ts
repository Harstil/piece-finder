/**
 * Homography and affine fits against OpenCV / numpy oracles:
 * - 4-point solve vs cv2.getPerspectiveTransform: identical matrices (relative 1e-6 after h8 = 1);
 * - N-point fit vs cv2.findHomography(method=0): exact points → same matrix; noisy points (σ = 0.3 px)
 *   → mapped points agree within 0.05 px (OpenCV also refines the reprojection error; measured 0.024);
 * - affine / similarity vs numpy least squares and Umeyama: relative 1e-5.
 * Plus inversion, composition and degenerate inputs.
 */

import { describe, expect, it } from 'vitest'
import { golden, pointsFrom } from '../__golden__/fixtures.ts'
import type { Point } from '../types.ts'
import {
  affineToHomography,
  applyAffine,
  applyHomography,
  composeHomography,
  estimateAffine,
  estimateSimilarity,
  homographyFromPoints,
  homographyFromQuad,
  identityHomography,
  invertHomography,
  transformPoints,
} from './homography.ts'

function expectMatrixClose(actual: ArrayLike<number>, expected: ArrayLike<number>, rel: number): void {
  let scale = 0
  for (let i = 0; i < expected.length; i++) scale = Math.max(scale, Math.abs(expected[i]))
  for (let i = 0; i < expected.length; i++) expect(Math.abs(actual[i] - expected[i])).toBeLessThan(rel * scale + 1e-12)
}

describe('homographyFromQuad', () => {
  it.each(golden.homography.quads.map((q, i) => [i, q] as const))('matches getPerspectiveTransform, case %i', (_, q) => {
    const H = homographyFromQuad(pointsFrom(q.src), pointsFrom(q.dst))
    expect(H).not.toBeNull()
    expectMatrixClose(H!, q.H, 1e-6)
  })

  it('maps the corners exactly and rejects collinear corners', () => {
    const src: Point[] = [{ x: 10, y: 20 }, { x: 300, y: 35 }, { x: 280, y: 250 }, { x: 5, y: 230 }]
    const dst: Point[] = [{ x: 0, y: 0 }, { x: 95, y: 0 }, { x: 95, y: 95 }, { x: 0, y: 95 }]
    const H = homographyFromQuad(src, dst)!
    for (let i = 0; i < 4; i++) {
      const p = applyHomography(H, src[i])
      expect(p.x).toBeCloseTo(dst[i].x, 9)
      expect(p.y).toBeCloseTo(dst[i].y, 9)
    }
    const line: Point[] = [{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }, { x: 0, y: 5 }]
    expect(homographyFromQuad(line, dst)).toBeNull()
    expect(() => homographyFromQuad(src.slice(0, 3), dst.slice(0, 3))).toThrow()
  })
})

describe('homographyFromPoints', () => {
  const [exact, noisy] = golden.homography.pointSets

  it('recovers the exact homography from noise-free points', () => {
    const H = homographyFromPoints(pointsFrom(exact.src), pointsFrom(exact.dst))!
    expectMatrixClose(H, exact.H, 1e-5)
  })

  it('agrees with OpenCV on noisy points within 0.05 px', () => {
    const src = pointsFrom(noisy.src)
    const H = homographyFromPoints(src, pointsFrom(noisy.dst))!
    const ref = Float64Array.from(noisy.H)
    for (const p of src) {
      const a = applyHomography(H, p)
      const b = applyHomography(ref, p)
      expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeLessThan(0.05)
    }
  })

  it('returns null for too few or collinear points', () => {
    const pts: Point[] = [0, 1, 2, 3, 4].map((i) => ({ x: i, y: 2 * i }))
    expect(homographyFromPoints(pts.slice(0, 3), pts.slice(0, 3))).toBeNull()
    expect(homographyFromPoints(pts, pts)).toBeNull()
  })
})

describe('homography algebra', () => {
  const H = Float64Array.from(golden.homography.quads[0].H)

  it('inverts and composes', () => {
    const inv = invertHomography(H)!
    const I = composeHomography(H, inv)
    expectMatrixClose(I, identityHomography(), 1e-9)
    const p = { x: 123.4, y: 56.7 }
    const q = applyHomography(inv, applyHomography(H, p))
    expect(q.x).toBeCloseTo(p.x, 8)
    expect(q.y).toBeCloseTo(p.y, 8)
    expect(invertHomography(new Float64Array(9))).toBeNull()
  })

  it('composes as "apply the right-hand side first"', () => {
    const shift = Float64Array.of(1, 0, 10, 0, 1, 0, 0, 0, 1)
    const double = Float64Array.of(2, 0, 0, 0, 2, 0, 0, 0, 1)
    const p = applyHomography(composeHomography(double, shift), { x: 1, y: 1 })
    expect(p).toEqual({ x: 22, y: 2 })
    expect(transformPoints(shift, [{ x: 0, y: 0 }])).toEqual([{ x: 10, y: 0 }])
  })
})

describe('affine and similarity fits', () => {
  const { fit } = golden.homography
  const src = pointsFrom(fit.src)
  const dst = pointsFrom(fit.dst)

  it('estimateAffine matches numpy least squares', () => {
    expectMatrixClose(estimateAffine(src, dst)!, fit.affine, 1e-5)
  })

  it('estimateSimilarity matches Umeyama', () => {
    const s = estimateSimilarity(src, dst)!
    expectMatrixClose(s.matrix, fit.similarity, 1e-5)
    expect(s.scale).toBeCloseTo(1.3, 1)
    expect(s.angle).toBeCloseTo(0.35, 1)
  })

  it('recovers an exact similarity; positive angles turn clockwise on screen', () => {
    const pts: Point[] = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 3, y: 7 }]
    // 90° clockwise on screen: +x → +y.
    const rotated = pts.map((p) => ({ x: -p.y * 2 + 5, y: p.x * 2 - 1 }))
    const s = estimateSimilarity(pts, rotated)!
    expect(s.angle).toBeCloseTo(Math.PI / 2, 9)
    expect(s.scale).toBeCloseTo(2, 9)
    const q = applyAffine(s.matrix, { x: 10, y: 0 })
    expect(q.x).toBeCloseTo(5, 9)
    expect(q.y).toBeCloseTo(19, 9)
    const asH = affineToHomography(s.matrix)
    expect(applyHomography(asH, { x: 10, y: 0 }).y).toBeCloseTo(19, 9)
  })

  it('rejects degenerate inputs', () => {
    const line: Point[] = [{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }]
    expect(estimateAffine(line, line)).toBeNull()
    expect(estimateSimilarity([{ x: 1, y: 1 }, { x: 1, y: 1 }], line.slice(0, 2))).toBeNull()
  })
})
