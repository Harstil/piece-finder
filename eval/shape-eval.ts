/**
 * Shape analysis evaluation: `node eval/shape-eval.ts v1/val100 v1/val500 ...` (dataset names under datasets/).
 *
 * For every matchable scene piece (face-up, ≥ 60 % visible) runs analyzeShape on
 *   (A) the ground-truth outline of the printed top face, and
 *   (B) the contour traced from the instance mask (top face + visible side wall, like a real segmenter),
 * matches the returned corners to the ground truth cyclically (both are clockwise in the image) and
 * reports corner error relative to the core side, side-kind and piece-type accuracy, and timing.
 * Everything downstream (canonical warp, rotation, border constraints) depends on these numbers.
 */

import { join } from 'node:path'
import { findContours } from '../src/engine/geom/index.ts'
import { analyzeShape } from '../src/engine/shape.ts'
import type { Mask, PieceShape, Point, SideKind } from '../src/engine/types.ts'
import {
  isMatchable,
  loadLabels,
  openDataset,
  readJson,
  toPoint,
  toQuad,
  type PieceJson,
  type ReferenceJson,
  type SceneJson,
} from './lib/dataset.ts'

/** Pieces at least this visible count as unoccluded in the mask-mode breakdown. Guessed. */
const FULLY_VISIBLE = 0.98

interface Tally {
  n: number
  nulls: number
  within05: number
  within10: number
  sidesRight: number
  sidesTotal: number
  typeRight: number
  errs: number[]
  ms: number[]
}

const newTally = (): Tally => ({ n: 0, nulls: 0, within05: 0, within10: 0, sidesRight: 0, sidesTotal: 0, typeRight: 0, errs: [], ms: [] })

function pieceType(sides: readonly SideKind[]): string {
  const f = sides.filter((s) => s === 'flat').length
  return f === 2 ? 'corner' : f === 1 ? 'edge' : 'interior'
}

/** Score one analysis against ground truth. */
function score(t: Tally, shape: PieceShape | null, gtCorners: Point[], gtSides: SideKind[], ms: number): void {
  t.n++
  t.ms.push(ms)
  if (shape === null) {
    t.nulls++
    t.errs.push(Infinity)
    return
  }
  const core = Math.sqrt(Math.abs(quadArea(gtCorners)))
  let best = Infinity
  let shift = 0
  for (let s = 0; s < 4; s++) {
    let worst = 0
    for (let i = 0; i < 4; i++) {
      const a = shape.corners[(i + s) % 4]
      worst = Math.max(worst, Math.hypot(a.x - gtCorners[i].x, a.y - gtCorners[i].y))
    }
    if (worst < best) {
      best = worst
      shift = s
    }
  }
  const err = best / core
  t.errs.push(err)
  if (err <= 0.05) t.within05++
  if (err <= 0.1) t.within10++
  for (let i = 0; i < 4; i++) {
    t.sidesTotal++
    if (shape.sides[(i + shift) % 4] === gtSides[i]) t.sidesRight++
  }
  if (shape.pieceType === pieceType(gtSides)) t.typeRight++
}

function quadArea(q: Point[]): number {
  let a = 0
  for (let i = 0; i < 4; i++) {
    const p = q[i]
    const r = q[(i + 1) % 4]
    a += p.x * r.y - r.x * p.y
  }
  return a / 2
}

/** Contour of instance `index` in the label map, traced by the engine (largest outer contour). */
function maskContour(labels: Mask, index: number, outline: Point[]): Point[] | null {
  let x0 = Infinity
  let y0 = Infinity
  let x1 = -Infinity
  let y1 = -Infinity
  for (const p of outline) {
    x0 = Math.min(x0, p.x)
    y0 = Math.min(y0, p.y)
    x1 = Math.max(x1, p.x)
    y1 = Math.max(y1, p.y)
  }
  const pad = Math.ceil(0.25 * Math.max(x1 - x0, y1 - y0)) + 2
  const ox = Math.max(0, Math.floor(x0) - pad)
  const oy = Math.max(0, Math.floor(y0) - pad)
  const w = Math.min(labels.width, Math.ceil(x1) + pad + 1) - ox
  const h = Math.min(labels.height, Math.ceil(y1) + pad + 1) - oy
  const crop: Mask = { width: w, height: h, data: new Uint8Array(w * h) }
  for (let y = 0; y < h; y++) {
    const row = (oy + y) * labels.width + ox
    for (let x = 0; x < w; x++) crop.data[y * w + x] = labels.data[row + x] === index ? 1 : 0
  }
  const contours = findContours(crop, { externalOnly: true })
  if (contours.length === 0) return null
  let best = contours[0]
  for (const c of contours) if (c.length > best.length) best = c
  return best.map((p) => ({ x: p.x + ox, y: p.y + oy }))
}

function pct(a: number, b: number): string {
  return b === 0 ? '—' : `${((100 * a) / b).toFixed(1)}%`
}

function quantile(xs: number[], q: number): number {
  const s = [...xs].sort((a, b) => a - b)
  return s.length === 0 ? NaN : s[Math.min(s.length - 1, Math.floor(q * s.length))]
}

function line(name: string, t: Tally): string {
  return [
    name.padEnd(22),
    `n=${String(t.n).padStart(4)}`,
    `null ${pct(t.nulls, t.n).padStart(6)}`,
    `≤0.05 ${pct(t.within05, t.n).padStart(6)}`,
    `≤0.10 ${pct(t.within10, t.n).padStart(6)}`,
    `med ${quantile(t.errs, 0.5).toFixed(3)}`,
    `p90 ${quantile(t.errs, 0.9).toFixed(3)}`,
    `sides ${pct(t.sidesRight, t.sidesTotal).padStart(6)}`,
    `type ${pct(t.typeRight, t.n).padStart(6)}`,
    `${quantile(t.ms, 0.5).toFixed(2)} ms`,
  ].join('  ')
}

async function main(): Promise<void> {
  const names = process.argv.slice(2)
  if (names.length === 0) throw new Error('usage: node eval/shape-eval.ts <dataset> [...]  (e.g. v1/val500)')
  for (const name of names) {
    const ds = await openDataset(name)
    const A = newTally()
    const B = newTally()
    const Bfull = newTally()
    const puzzleCache = new Map<string, { sides: Map<number, SideKind[]> }>()
    for (const sceneId of ds.meta.scenes) {
      const base = join(ds.root, 'scenes', sceneId)
      const scene = await readJson<SceneJson>(`${base}.json`)
      let puzzle = puzzleCache.get(scene.puzzleId)
      if (puzzle === undefined) {
        const dir = join(ds.root, 'puzzles', scene.puzzleId)
        await readJson<ReferenceJson>(join(dir, 'reference.json'))
        const pieces = await readJson<PieceJson[]>(join(dir, 'pieces.json'))
        puzzle = { sides: new Map(pieces.map((p) => [p.id, p.sides])) }
        puzzleCache.set(scene.puzzleId, puzzle)
      }
      const labels = await loadLabels(`${base}_mask.png`)
      const center = { x: (scene.width - 1) / 2, y: (scene.height - 1) / 2 }
      for (const p of scene.pieces) {
        if (!isMatchable(p)) continue
        const gtCorners = toQuad(p.corners)
        const gtSides = puzzle.sides.get(p.pieceId) as SideKind[]
        const outline = p.outline.map(toPoint)
        let t0 = performance.now()
        score(A, analyzeShape(outline), gtCorners, gtSides, performance.now() - t0)
        const contour = maskContour(labels, p.index, outline)
        t0 = performance.now()
        const shapeB = contour === null ? null : analyzeShape(contour, { frameCenter: center })
        const msB = performance.now() - t0
        score(B, shapeB, gtCorners, gtSides, msB)
        if (p.visibleFraction >= FULLY_VISIBLE) score(Bfull, shapeB, gtCorners, gtSides, msB)
      }
    }
    console.log(line(`${name} A:outline`, A))
    console.log(line(`${name} B:mask`, B))
    console.log(line(`${name} B:mask, unoccluded`, Bfull))
  }
}

await main()
