/**
 * Quarter-turn rotation on an asymmetric image: the direction must be clockwise on screen, four turns
 * must be the identity, turns must compose, and every variant must agree with rotatePoint.
 */

import { describe, expect, it } from 'vitest'
import { maskFromRows, maskToRows } from '../__golden__/fixtures.ts'
import type { GrayImage, LabImage } from '../types.ts'
import { rotateGray, rotateLab, rotateMask, rotatePoint, rotatedSize } from './rotate.ts'

// An "L" with a foot: no symmetry, so every wrong mapping shows.
const L = maskFromRows(['100', '100', '111', '000'])

describe('rotateMask', () => {
  it('turns clockwise on screen for r = 1', () => {
    // The vertical stroke on the left becomes a horizontal stroke on top, running right to left.
    expect(maskToRows(rotateMask(L, 1))).toEqual(['0111', '0100', '0100'])
  })

  it('handles r = 2, r = 3 and negative / large r', () => {
    expect(maskToRows(rotateMask(L, 2))).toEqual(['000', '111', '001', '001'])
    expect(maskToRows(rotateMask(L, 3))).toEqual(['0010', '0010', '1110'])
    expect(maskToRows(rotateMask(L, -1))).toEqual(maskToRows(rotateMask(L, 3)))
    expect(maskToRows(rotateMask(L, 6))).toEqual(maskToRows(rotateMask(L, 2)))
  })

  it('composes and returns to the start after four turns', () => {
    let m = L
    for (let i = 0; i < 4; i++) m = rotateMask(m, 1)
    expect(maskToRows(m)).toEqual(maskToRows(L))
    expect(maskToRows(rotateMask(rotateMask(L, 1), 2))).toEqual(maskToRows(rotateMask(L, 3)))
  })

  it('refuses to rotate into its own buffer', () => {
    const sq = maskFromRows(['10', '00'])
    expect(() => rotateMask(sq, 1, sq)).toThrow()
  })
})

describe('rotateGray / rotateLab / rotatePoint', () => {
  const W = 5
  const H = 3
  const img: GrayImage = { width: W, height: H, data: Float32Array.from({ length: W * H }, (_, i) => i) }

  it.each([0, 1, 2, 3])('moves every pixel where rotatePoint says, r = %i', (r) => {
    const out = rotateGray(img, r)
    expect({ width: out.width, height: out.height }).toEqual(rotatedSize(W, H, r))
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const p = rotatePoint({ x, y }, r, W, H)
        expect(out.data[p.y * out.width + p.x]).toBe(img.data[y * W + x])
      }
    }
  })

  it('rotates the three Lab planes together', () => {
    const lab: LabImage = { width: W, height: H, L: img.data, a: img.data.map((v) => -v), b: img.data.map((v) => v * 10) }
    const out = rotateLab(lab, 1)
    const ref = rotateGray(img, 1)
    expect(Array.from(out.L)).toEqual(Array.from(ref.data))
    expect(Array.from(out.a)).toEqual(Array.from(ref.data).map((v) => -v))
    expect(Array.from(out.b)).toEqual(Array.from(ref.data).map((v) => v * 10))
  })

  it('maps the top-left corner to the top-right for one clockwise turn', () => {
    expect(rotatePoint({ x: 0, y: 0 }, 1, W, H)).toEqual({ x: H - 1, y: 0 })
    expect(rotatePoint({ x: 0, y: 0 }, 3, W, H)).toEqual({ x: 0, y: W - 1 })
  })
})
