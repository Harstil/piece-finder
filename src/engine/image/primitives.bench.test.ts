/**
 * Micro-benchmark of the per-frame primitives at camera resolution. Skipped unless BENCH=1; the
 * verbose reporter is needed to see the printed numbers (Vitest hides console output of passing tests):
 *   BENCH=1 npx vitest run src/engine --reporter=verbose
 * Prints the median of several timed runs (after warm-up) for: Lab conversion of a 1920×1080 frame,
 * area resize of that frame to 512 px, a perspective warp of a piece to a 96×96 canonical Lab patch,
 * and contour tracing (labelling included) on a 1080×1920 mask holding 40 piece-like blobs.
 * Numbers are desktop Node; the iPhone is expected to be a few times slower.
 */

import { describe, expect, it } from 'vitest'
import { findContours } from '../geom/contour.ts'
import { rasterizePolygon } from '../geom/polygon.ts'
import { rectToQuadHomography, warpLab } from '../geom/warp.ts'
import type { Mask, Point, RGBAImage } from '../types.ts'
import { createLab, createMask, createRGBA } from './create.ts'
import { rgbaToLab } from './lab.ts'
import { fitSize, resizeAreaRGBA } from './resize.ts'

const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env
const enabled = env?.BENCH === '1'

function median(fn: () => void, runs: number): number {
  for (let i = 0; i < 3; i++) fn()
  const times: number[] = []
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now()
    fn()
    times.push(performance.now() - t0)
  }
  times.sort((a, b) => a - b)
  return times[times.length >> 1]
}

function syntheticFrame(width: number, height: number): RGBAImage {
  const img = createRGBA(width, height)
  let s = 1
  for (let i = 0; i < width * height; i++) {
    s = (s * 1664525 + 1013904223) >>> 0
    const x = i % width
    const y = (i - x) / width
    img.data[i * 4] = (x * 255) / width + (s & 15)
    img.data[i * 4 + 1] = (y * 255) / height + ((s >>> 4) & 15)
    img.data[i * 4 + 2] = 128 + 100 * Math.sin(x / 37 + y / 53)
    img.data[i * 4 + 3] = 255
  }
  return img
}

/** 40 wobbly blobs (piece-sized, ~150 px) scattered over a portrait 1080×1920 mask. */
function syntheticMask(): Mask {
  const width = 1080
  const height = 1920
  const mask = createMask(width, height)
  const blob = createMask(width, height)
  let s = 7
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 2 ** 32
  }
  for (let k = 0; k < 40; k++) {
    const cx = 100 + rand() * (width - 200)
    const cy = 100 + rand() * (height - 200)
    const phase = rand() * 6.28
    const poly: Point[] = []
    for (let i = 0; i < 180; i++) {
      const t = (i / 180) * 2 * Math.PI
      const r = 70 + 14 * Math.sin(4 * t + phase)
      poly.push({ x: cx + r * Math.cos(t), y: cy + r * Math.sin(t) })
    }
    rasterizePolygon(poly, width, height, blob)
    for (let i = 0; i < mask.data.length; i++) mask.data[i] |= blob.data[i]
  }
  return mask
}

describe.skipIf(!enabled)('primitive benchmarks (BENCH=1)', () => {
  it('times the per-frame primitives', () => {
    const frame = syntheticFrame(1920, 1080)
    const lab = createLab(1920, 1080)
    const small = fitSize(1920, 1080, 512)
    const smallOut = createRGBA(small.width, small.height)
    const patch = createLab(96, 96)
    const valid = createMask(96, 96)
    const quad: Point[] = [{ x: 900, y: 400 }, { x: 1090, y: 450 }, { x: 1040, y: 640 }, { x: 850, y: 590 }]
    const H = rectToQuadHomography(quad, 96, 96)!
    const mask = syntheticMask()
    let contours = 0

    const results = {
      'rgbaToLab 1920x1080': median(() => rgbaToLab(frame, lab), 15),
      [`resizeAreaRGBA 1920x1080 -> ${small.width}x${small.height}`]: median(() => resizeAreaRGBA(frame, small.width, small.height, smallOut), 15),
      'warpLab 1920x1080 -> 96x96 (+valid)': median(() => warpLab(lab, H, 96, 96, { out: patch, valid }), 200),
      'findContours 1080x1920 (40 blobs)': median(() => {
        contours = findContours(mask).length
      }, 15),
    }
    for (const [name, ms] of Object.entries(results)) console.log(`${name.padEnd(48)} ${ms.toFixed(2)} ms`)
    expect(contours).toBeGreaterThan(0)
  })
})
