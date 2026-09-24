/**
 * Morphology against cv2.erode / cv2.dilate / cv2.morphologyEx (default anchor and border) for
 * rectangular kernels (odd, even, 1-wide) and elliptical kernels, on masks with holes, lines, noise and
 * frame contact. Binary outputs must be identical; ellipse runs must reproduce getStructuringElement.
 */

import { describe, expect, it } from 'vitest'
import { golden, goldenMask, maskFromRows, maskToRows } from '../__golden__/fixtures.ts'
import { close, dilate, ellipseKernel, ellipseRuns, erode, fillHoles, open, rectKernel } from './morph.ts'

function kernelRows(width: number, height: number): string[] {
  const runs = ellipseRuns(width, height)
  const rows: string[] = []
  for (let i = 0; i < height; i++) {
    let row = ''
    for (let x = 0; x < width; x++) row += x >= runs[i * 2] && x < runs[i * 2 + 1] ? '1' : '0'
    rows.push(row)
  }
  return rows
}

describe('binary morphology vs OpenCV', () => {
  const cases = golden.morph.cases.map((c) => [`${c.mask} ${c.shape} ${c.width}x${c.height}`, c] as const)

  it.each(cases)('%s', (_, c) => {
    const mask = goldenMask(c.mask)
    const se = c.shape === 'rect' ? rectKernel(c.width, c.height) : ellipseKernel(c.width, c.height)
    if (c.shape === 'ellipse') expect(kernelRows(c.width, c.height)).toEqual(c.kernel)
    expect(maskToRows(erode(mask, se))).toEqual(c.erode)
    expect(maskToRows(dilate(mask, se))).toEqual(c.dilate)
    expect(maskToRows(open(mask, se))).toEqual(c.open)
    expect(maskToRows(close(mask, se))).toEqual(c.close)
  })
})

describe('morphology details', () => {
  it('works in place', () => {
    const mask = goldenMask('holes')
    const expected = maskToRows(open(mask, ellipseKernel(5)))
    expect(maskToRows(open(mask, ellipseKernel(5), mask))).toEqual(expected)
  })

  it('treats any nonzero value as foreground and outputs 0/1', () => {
    const m = { width: 3, height: 1, data: Uint8Array.from([0, 7, 200]) }
    expect(Array.from(dilate(m, rectKernel(1)).data)).toEqual([0, 1, 1])
  })

  it('rejects invalid kernels', () => {
    expect(() => erode(goldenMask('blobs'), rectKernel(0))).toThrow()
  })
})

describe('fillHoles', () => {
  it('fills enclosed background and leaves background open to the frame', () => {
    const m = maskFromRows(['0000000', '0111110', '0100010', '0101010', '0100010', '0111110', '0000001'])
    expect(maskToRows(fillHoles(m))).toEqual(['0000000', '0111110', '0111110', '0111110', '0111110', '0111110', '0000001'])
  })

  it('does not fill a region that leaks through a 4-connected gap', () => {
    const m = maskFromRows(['11111', '10001', '10000', '11111'])
    expect(maskToRows(fillHoles(m))).toEqual(['11111', '10001', '10000', '11111'])
  })
})
