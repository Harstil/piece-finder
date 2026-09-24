/**
 * Connected components against cv2.connectedComponentsWithStats(connectivity=8): the label partition,
 * areas, bounding boxes and centroids must be identical (label numbers are matched through the pixels,
 * since only the partition is part of the contract), plus the documented raster numbering order and the
 * outer-background fill.
 */

import { describe, expect, it } from 'vitest'
import { MASK_NAMES, golden, goldenMask, maskFromRows } from '../__golden__/fixtures.ts'
import { componentMask, labelComponents, markOuterBackground } from './components.ts'

describe('labelComponents', () => {
  it.each(MASK_NAMES)('matches OpenCV on the %s mask', (name) => {
    const mask = goldenMask(name)
    const expected = golden.components[name]
    const comps = labelComponents(mask)
    expect(comps.count).toBe(expected.count)

    // Our label k ↔ OpenCV label map[k], must be a bijection consistent on every pixel.
    const map = new Map<number, number>()
    for (let i = 0; i < comps.labels.length; i++) {
      const ours = comps.labels[i]
      const theirs = expected.labels[i]
      expect(ours === 0).toBe(theirs === 0)
      if (ours === 0) continue
      const prev = map.get(ours)
      if (prev === undefined) map.set(ours, theirs)
      else expect(prev).toBe(theirs)
    }
    expect(new Set(map.values()).size).toBe(comps.count)

    for (const s of comps.stats) {
      const t = map.get(s.label)!
      const [x, y, w, h, area] = expected.stats[t - 1]
      expect(s.bbox).toEqual({ x, y, width: w, height: h })
      expect(s.area).toBe(area)
      expect(s.centroid.x).toBeCloseTo(expected.centroids[t - 1][0], 4)
      expect(s.centroid.y).toBeCloseTo(expected.centroids[t - 1][1], 4)
      expect(comps.labels[s.start]).toBe(s.label)
    }
  })

  it('numbers components in raster order of their first pixel', () => {
    const mask = maskFromRows(['00001', '11000', '00000', '01110', '10001'])
    const comps = labelComponents(mask)
    expect(comps.count).toBe(3)
    expect(comps.stats.map((s) => s.start)).toEqual([4, 5, 16])
    // The U-shaped bottom component joins through two provisional labels.
    expect(comps.stats[2].area).toBe(5)
  })

  it('handles empty, full and single-pixel masks', () => {
    expect(labelComponents(maskFromRows(['000', '000'])).count).toBe(0)
    const full = labelComponents(maskFromRows(['111', '111']))
    expect(full.count).toBe(1)
    expect(full.stats[0].bbox).toEqual({ x: 0, y: 0, width: 3, height: 2 })
    expect(labelComponents(maskFromRows(['1'])).stats[0].centroid).toEqual({ x: 0, y: 0 })
  })

  it('extracts one component as a mask', () => {
    const comps = labelComponents(maskFromRows(['1001', '1001']))
    expect(Array.from(componentMask(comps, 2).data)).toEqual([0, 0, 0, 1, 0, 0, 0, 1])
  })
})

describe('markOuterBackground', () => {
  it('marks background reachable from the frame through 4-connected steps only', () => {
    const mask = maskFromRows(['11111', '10101', '11011', '10001', '11111'])
    // (1,1) and (3,1) are closed off; (2,2)…(1..3,3) form one enclosed region; nothing reaches the frame.
    expect(Array.from(markOuterBackground(mask))).toEqual(new Array(25).fill(0))
    const open = maskFromRows(['00000', '01110', '01010', '01110', '00000'])
    const marked = markOuterBackground(open)
    expect(marked[12]).toBe(0) // the hole
    expect(marked[0]).toBe(1)
    expect(marked[6]).toBe(0) // foreground is never marked
    // A diagonal gap does not let the background leak (4-connected background).
    const diag = maskFromRows(['0110', '1001', '1001', '0110'])
    expect(markOuterBackground(diag)[5]).toBe(0)
  })
})
