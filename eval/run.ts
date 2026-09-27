/**
 * Matching evaluation: `node eval/run.ts --sets v1/val500,v1/val1000 [--shape outline] [--limit 200]`
 *
 * Runs the real engine (buildReference → analyzeShape → canonicalize → matchPiece) on dataset scenes and
 * reports top-1 / top-5 cell accuracy, rotation accuracy, verdict precision and latency. Every engine
 * choice is judged by these numbers, so:
 * - the reference is built from reference.jpg and the ground-truth *reference corners* only (never
 *   motif.png), as the app builds it from the user's corners;
 * - `--shape oracle` gives the matcher ground-truth corners with a per-piece rotated start index, so it
 *   cannot learn "rotation 0" from the oracle; `outline` runs analyzeShape on the printed-face outline;
 *   `mask` runs it on the contour traced from the instance mask (side wall included, like a segmenter);
 * - only face-up, ≥ 60 %-visible pieces count (DATASET.md);
 * - `--fit` refuses test sets.
 *
 * Options:
 *   --sets a,b        dataset names under datasets/ (required)
 *   --shape m         oracle | outline | mask (default outline)
 *   --limit N         at most N pieces per set (spread evenly over scenes)
 *   --excluded f      pretend a fraction f of the other cells is already placed
 *   --off cue,cue     disable cues (lumaZncc, chroma, gradient, histogram, shapePrior)
 *   --by-puzzle       also print top-1 per puzzle (reference kind, grid, photo px per cell, motif style)
 *   --fit             coordinate-search the fusion weights on the given (non-test) sets and print them
 *   --out name        also write eval/reports/<name>.md
 */

import { mkdir, writeFile } from 'node:fs/promises'
import { execSync } from 'node:child_process'
import { join } from 'node:path'
import { findContours } from '../src/engine/geom/index.ts'
import {
  DEFAULT_FUSION,
  analyzeShape,
  buildReference,
  canonicalize,
  matchPiece,
  type CanonicalPiece,
  type CueName,
  type FusionParams,
  type Mask,
  type MatchOptions,
  type PieceShape,
  type Point,
  type ReferenceModel,
  type SideKind,
} from '../src/engine/index.ts'
import {
  angleDiff,
  isMatchable,
  loadPuzzle,
  loadScene,
  openDataset,
  scenesByPuzzle,
  toPoint,
  toQuad,
  type ScenePieceJson,
} from './lib/dataset.ts'

type ShapeMode = 'oracle' | 'outline' | 'mask'

interface Item {
  set: string
  puzzle: string
  ref: ReferenceModel
  piece: CanonicalPiece
  truthCell: number
  truthUp: number
  truthType: string
  canonMs: number
}

interface Args {
  sets: string[]
  shape: ShapeMode
  limit: number
  excluded: number
  off: CueName[]
  fit: boolean
  byPuzzle: boolean
  out: string | null
}

function parseArgs(argv: string[]): Args {
  const get = (name: string): string | undefined => {
    const i = argv.indexOf(`--${name}`)
    return i >= 0 ? argv[i + 1] : undefined
  }
  const sets = (get('sets') ?? '').split(',').filter(Boolean)
  if (sets.length === 0) throw new Error('usage: node eval/run.ts --sets v1/val500[,…] [--shape oracle|outline|mask] [--limit N] [--fit]')
  const shape = (get('shape') ?? 'outline') as ShapeMode
  if (!['oracle', 'outline', 'mask'].includes(shape)) throw new Error(`unknown --shape ${shape}`)
  return {
    sets,
    shape,
    limit: Number(get('limit') ?? Infinity),
    excluded: Number(get('excluded') ?? 0),
    off: ((get('off') ?? '').split(',').filter(Boolean) as CueName[]),
    fit: argv.includes('--fit'),
    byPuzzle: argv.includes('--by-puzzle'),
    out: get('out') ?? null,
  }
}

/** Deterministic pseudo-random in [0, 1) from integers (for the oracle start index and exclusions). */
function hash01(...xs: number[]): number {
  let h = 2166136261
  for (const x of xs) {
    h ^= x
    h = Math.imul(h, 16777619)
    h ^= h >>> 13
  }
  return (h >>> 0) / 4294967296
}

function pieceTypeOf(sides: readonly SideKind[]): 'corner' | 'edge' | 'interior' {
  const flats = sides.filter((s) => s === 'flat').length
  return flats === 2 ? 'corner' : flats === 1 ? 'edge' : 'interior'
}

function oracleShape(p: ScenePieceJson, sides: SideKind[], sceneIndex: number): PieceShape {
  const c = toQuad(p.corners)
  const s = Math.floor(hash01(p.pieceId, sceneIndex, 17) * 4)
  const corners = [c[s], c[(s + 1) % 4], c[(s + 2) % 4], c[(s + 3) % 4]] as PieceShape['corners']
  const rs = [sides[s], sides[(s + 1) % 4], sides[(s + 2) % 4], sides[(s + 3) % 4]] as PieceShape['sides']
  return { corners, sides: rs, pieceType: pieceTypeOf(sides), confidence: 1 }
}

/** Largest outer contour of instance `index` in the label map, traced by the engine. */
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
  const contours = findContours(crop)
  if (contours.length === 0) return null
  let best = contours[0]
  for (const c of contours) if (c.length > best.length) best = c
  return best.map((p) => ({ x: p.x + ox, y: p.y + oy }))
}

interface CollectStats {
  pieces: number
  shapeFailures: number
}

/** Loads a set and turns its matchable pieces into canonical pieces (≤ limit, spread over scenes). */
async function collect(name: string, mode: ShapeMode, limit: number, stats: CollectStats): Promise<Item[]> {
  const ds = await openDataset(name)
  const byPuzzle = await scenesByPuzzle(ds)
  const totalScenes = ds.meta.scenes.length
  const perScene = Number.isFinite(limit) ? Math.max(1, Math.ceil(limit / totalScenes)) : Infinity
  const items: Item[] = []
  for (const puzzleId of ds.meta.puzzles) {
    const puzzle = await loadPuzzle(ds, puzzleId)
    const grid = { cols: puzzle.ref.cols, rows: puzzle.ref.rows }
    const ref = buildReference(puzzle.reference, toQuad(puzzle.ref.referenceCorners), grid)
    const sidesById = new Map(puzzle.pieces.map((p) => [p.id, p.sides]))
    for (const sceneId of byPuzzle.get(puzzleId) ?? []) {
      const scene = await loadScene(ds, sceneId)
      const sceneIndex = ds.meta.scenes.indexOf(sceneId)
      const center = { x: (scene.json.width - 1) / 2, y: (scene.json.height - 1) / 2 }
      let taken = 0
      for (const p of scene.json.pieces) {
        if (!isMatchable(p) || taken >= perScene || items.length >= limit) continue
        taken++
        stats.pieces++
        const sides = sidesById.get(p.pieceId) as SideKind[]
        const outline = p.outline.map(toPoint)
        let shape: PieceShape | null
        let contour: Point[] = outline
        if (mode === 'oracle') shape = oracleShape(p, sides, sceneIndex)
        else if (mode === 'outline') shape = analyzeShape(outline)
        else {
          const traced = maskContour(scene.labels, p.index, outline)
          shape = traced === null ? null : analyzeShape(traced, { frameCenter: center })
          if (traced !== null) contour = traced
        }
        if (shape === null) {
          stats.shapeFailures++
          continue
        }
        const t0 = performance.now()
        const piece = canonicalize(scene.frame, contour, shape)
        items.push({
          set: name,
          puzzle: `${puzzleId} ${puzzle.ref.referenceKind ?? '?'} ${puzzle.ref.cols}x${puzzle.ref.rows} ${ref.nativeCellPx.toFixed(0)}px/cell ${puzzle.ref.motifStyle ?? ''}`,
          ref,
          piece,
          truthCell: p.cell,
          truthUp: p.upAngleDeg,
          truthType: pieceTypeOf(sides),
          canonMs: performance.now() - t0,
        })
      }
    }
  }
  return items
}

interface Result {
  n: number
  top1: number
  top5: number
  rot: number
  joint: number
  logLik: number
  verdicts: Record<string, { n: number; right: number }>
  byType: Record<string, { n: number; top1: number }>
  byPuzzle: Record<string, { n: number; top1: number }>
  matchMs: number[]
  canonMs: number[]
}

function evaluate(items: Item[], opts: MatchOptions, excluded: number): Result {
  const res: Result = {
    n: 0,
    top1: 0,
    top5: 0,
    rot: 0,
    joint: 0,
    logLik: 0,
    verdicts: { strong: { n: 0, right: 0 }, likely: { n: 0, right: 0 }, unsure: { n: 0, right: 0 } },
    byType: {},
    byPuzzle: {},
    matchMs: [],
    canonMs: [],
  }
  items.forEach((it, i) => {
    let excludedCells: Set<number> | undefined
    if (excluded > 0) {
      excludedCells = new Set()
      const cells = it.ref.grid.cols * it.ref.grid.rows
      for (let c = 0; c < cells; c++) if (c !== it.truthCell && hash01(i, c, 99) < excluded) excludedCells.add(c)
    }
    const m = matchPiece(it.ref, it.piece, { ...opts, excludedCells })
    res.n++
    res.matchMs.push(m.timingsMs.total)
    res.canonMs.push(it.canonMs)
    const rank = m.candidates.findIndex((c) => c.cell === it.truthCell)
    const right = rank === 0
    const rotRight = right && Math.abs(angleDiff(m.candidates[0].upAngleDeg, it.truthUp)) < 45
    if (right) res.top1++
    if (rank >= 0) res.top5++
    if (rotRight) res.rot++
    if (rotRight) res.joint++
    res.logLik += Math.log(Math.max(1e-4, rank >= 0 ? m.candidates[rank].prob : 0))
    const v = res.verdicts[m.verdict]
    v.n++
    if (right) v.right++
    const t = (res.byType[it.truthType] ??= { n: 0, top1: 0 })
    t.n++
    if (right) t.top1++
    const pz = (res.byPuzzle[it.puzzle] ??= { n: 0, top1: 0 })
    pz.n++
    if (right) pz.top1++
  })
  return res
}

const pct = (a: number, b: number): string => (b === 0 ? '—' : `${((100 * a) / b).toFixed(1)}%`)
function quantile(xs: number[], q: number): number {
  const s = [...xs].sort((a, b) => a - b)
  return s.length === 0 ? NaN : s[Math.min(s.length - 1, Math.floor(q * s.length))]
}

function row(name: string, r: Result): string {
  const v = r.verdicts
  const types = ['corner', 'edge', 'interior'].map((t) => `${t[0]}:${pct(r.byType[t]?.top1 ?? 0, r.byType[t]?.n ?? 0)}`).join(' ')
  return `| ${name} | ${r.n} | ${pct(r.top1, r.n)} | ${pct(r.top5, r.n)} | ${pct(r.rot, r.top1)} | ${pct(v.strong.right, v.strong.n)} of ${pct(v.strong.n, r.n)} | ${pct(v.likely.right, v.likely.n)} of ${pct(v.likely.n, r.n)} | ${types} | ${quantile(r.canonMs, 0.5).toFixed(1)} / ${quantile(r.matchMs, 0.5).toFixed(1)} / ${quantile(r.matchMs, 0.95).toFixed(1)} |`
}

const HEADER =
  '| set | n | top-1 | top-5 | rotation (given top-1) | strong: right of share | likely: right of share | top-1 by type | canon / match p50 / p95 ms |\n|---|---|---|---|---|---|---|---|---|'

/** Coordinate search over the ranking weights, then a 1-D search of the temperature. */
function fit(items: Item[], base: MatchOptions): FusionParams {
  let params: FusionParams = { ...DEFAULT_FUSION }
  const objective = (p: FusionParams): number => {
    const r = evaluate(items, { ...base, fusion: p }, 0)
    return r.top1 + 0.02 * r.logLik
  }
  let best = objective(params)
  console.log(`fit start: objective ${best.toFixed(2)} on ${items.length} pieces`)
  const keys: (keyof FusionParams)[] = ['wLuma', 'wChroma', 'wChromaMean', 'wGradient', 'wStats', 'wShapeMismatch', 'lumaTextureRef']
  for (let round = 0; round < 2; round++) {
    for (const key of keys) {
      for (const factor of [0.5, 0.75, 1.33, 2]) {
        const trial = { ...params, [key]: params[key] * factor }
        const score = objective(trial)
        if (score > best + 1e-9) {
          best = score
          params = trial
          console.log(`  ${key} ×${factor} → ${params[key].toFixed(3)}  objective ${best.toFixed(2)}`)
        }
      }
    }
  }
  let bestT = params.temperature
  let bestLL = -Infinity
  for (const t of [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2]) {
    const r = evaluate(items, { ...base, fusion: { ...params, temperature: t } }, 0)
    if (r.logLik > bestLL) {
      bestLL = r.logLik
      bestT = t
    }
  }
  params = { ...params, temperature: bestT }
  console.log(`temperature → ${bestT} (mean log-lik ${(bestLL / items.length).toFixed(3)})`)
  return params
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2))
  const cues: MatchOptions['cues'] = Object.fromEntries(args.off.map((c) => [c, false]))
  const base: MatchOptions = { cues }
  if (args.fit) {
    const all: Item[] = []
    for (const name of args.sets) {
      const ds = await openDataset(name)
      if (ds.meta.split === 'test') throw new Error(`refusing to fit on test set ${name}`)
      all.push(...(await collect(name, args.shape, args.limit, { pieces: 0, shapeFailures: 0 })))
    }
    const params = fit(all, base)
    console.log(JSON.stringify(params, null, 2))
    return
  }
  let commit = 'unknown'
  try {
    commit = execSync('git rev-parse --short HEAD').toString().trim()
  } catch {
    // not a git checkout
  }
  const lines = [
    `# Matching eval — shape ${args.shape}${args.excluded > 0 ? `, ${args.excluded * 100}% placed` : ''}${args.off.length > 0 ? `, off: ${args.off.join(',')}` : ''}`,
    `commit ${commit}`,
    '',
    HEADER,
  ]
  console.log(lines.join('\n'))
  for (const name of args.sets) {
    const stats = { pieces: 0, shapeFailures: 0 }
    const items = await collect(name, args.shape, args.limit, stats)
    const r = evaluate(items, base, args.excluded)
    const line = row(`${name}${stats.shapeFailures > 0 ? ` (${stats.shapeFailures} shape fails)` : ''}`, r)
    console.log(line)
    lines.push(line)
    if (args.byPuzzle) for (const [pz, v] of Object.entries(r.byPuzzle)) console.log(`    ${pz}: ${pct(v.top1, v.n)} of ${v.n}`)
  }
  if (args.out !== null) {
    await mkdir(join('eval', 'reports'), { recursive: true })
    await writeFile(join('eval', 'reports', `${args.out}.md`), `${lines.join('\n')}\n`)
  }
}

await main()
