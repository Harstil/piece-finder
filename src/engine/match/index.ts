/**
 * The matcher: where in the box picture does this canonical piece belong, and turned which way?
 *
 * Localisation is classification over grid cells. Every (cell, rotation) is a hypothesis; the piece's
 * border sides decide which are allowed (a flat side must lie on the puzzle's edge, a tab or blank must
 * not), then they are scored coarse-to-fine (levels.ts):
 *   level 8 px  — every allowed hypothesis, no shift (a few thousand at 1000 pieces);
 *   level 16 px — the best TOP_COARSE, each at the best of ±SHIFT_MID px (grids are not exact);
 *   level 32 px — the best TOP_MID, ±SHIFT_FINE px around the doubled mid-level shift.
 * Scores fuse texture, colour, structure and statistics cues (fusion.ts). Probabilities are a softmax
 * over the finalists, collapsed to one entry per cell. The coarse level also yields a per-cell heatmap.
 *
 * Prior art this follows: johnb8005/puzzle-piece-finder and YordalStun/PuzzleSorter both moved from
 * keypoints to dense masked ZNCC at a known scale; published keypoint accuracy collapses beyond ~100
 * pieces. Here the scale and perspective come from the piece's own corners (canonical.ts), so the
 * search is only over cells, 4 rotations and a few pixels of shift.
 */

import { upAngleDeg } from '../pose.ts'
import type { ReferenceLevel, ReferenceModel } from '../reference.ts'
import type { Candidate, CanonicalPiece, CueName, FusionParams, MatchOptions, MatchResult, Rotation } from '../types.ts'
import { resolveFusion, verdictFor } from './fusion.ts'
import { prepareViews, type PieceView } from './prepare.ts'

/** Hypotheses kept after the coarse level. Guessed; eval/run.ts reports how often the truth survives. */
const TOP_COARSE = 48
/** Hypotheses kept after the mid level. Guessed, same check. */
const TOP_MID = 12
/** Shift search radius at the mid level (px at 16 px/cell ≈ ±0.125 cell). Guessed: grid jitter + corner error. */
const SHIFT_MID = 2
/** Shift search radius at the finest level, around twice the mid shift (px at 32 px/cell). */
const SHIFT_FINE = 2
/** Shape confidence from which the border logic is a hard filter instead of a soft penalty. Guessed. */
const HARD_SHAPE_CONFIDENCE = 0.3
/** Fewer overlapping samples than this and a hypothesis is not scored (−∞). */
const MIN_SAMPLES = 8
/** Candidates returned by default. */
const DEFAULT_TOP_K = 5

interface CueFlags {
  luma: boolean
  chroma: boolean
  gradient: boolean
  stats: boolean
  shape: boolean
}

function cueFlags(cues: MatchOptions['cues']): CueFlags {
  const on = (name: CueName): boolean => cues?.[name] !== false
  return { luma: on('lumaZncc'), chroma: on('chroma'), gradient: on('gradient'), stats: on('histogram'), shape: on('shapePrior') }
}

/**
 * Fused score of view `v` against the reference window whose top-left padded pixel index is `base`.
 * Returns −Infinity when the window is (nearly) outside the motif.
 */
function scoreWindow(level: ReferenceLevel, v: PieceView, base: number, f: FusionParams, cues: CueFlags): number {
  const { L, A, B, GX, GY, valid } = level
  const off = v.off
  const pl = v.pl
  const pa = v.pa
  const pb = v.pb
  let n = 0
  let sL = 0
  let sLL = 0
  let dL = 0
  let sA = 0
  let sAA = 0
  let dA = 0
  let sB = 0
  let sBB = 0
  let dB = 0
  for (let k = 0; k < v.n; k++) {
    const q = base + off[k]
    if (valid[q] === 0) continue
    const l = L[q]
    const a = A[q]
    const b = B[q]
    n++
    sL += l
    sLL += l * l
    dL += pl[k] * l
    sA += a
    sAA += a * a
    dA += pa[k] * a
    sB += b
    sBB += b * b
    dB += pb[k] * b
  }
  if (n < MIN_SAMPLES) return -Infinity

  let score = 0
  const meanL = sL / n
  if (cues.luma) {
    const varL = sLL - sL * meanL
    const zncc = varL > 1e-6 ? dL / Math.sqrt(varL) : 0
    score += f.wLuma * Math.min(1, v.stdL / f.lumaTextureRef) * zncc
  }
  const meanA = sA / n
  const meanB = sB / n
  if (cues.chroma) {
    const varA = sAA - sA * meanA
    const varB = sBB - sB * meanB
    const za = varA > 1e-6 ? dA / Math.sqrt(varA) : 0
    const zb = varB > 1e-6 ? dB / Math.sqrt(varB) : 0
    const wa = v.stdA
    const wb = v.stdB
    const tex = Math.min(1, (wa + wb) / f.lumaTextureRef)
    score += f.wChroma * tex * ((za * wa + zb * wb) / (wa + wb + 1e-6))
    score -= (f.wChromaMean * Math.hypot(v.meanA - meanA, v.meanB - meanB)) / 10
  }
  if (cues.stats) {
    const stdL = Math.sqrt(Math.max(0, sLL / n - meanL * meanL))
    score -= f.wStats * (Math.abs(v.meanL - meanL) / 20 + Math.abs(v.stdL - stdL) / 10)
  }
  if (cues.gradient && v.gn > 0) {
    let dG = 0
    let sGG = 0
    const goff = v.goff
    const gx = v.gx
    const gy = v.gy
    for (let k = 0; k < v.gn; k++) {
      const q = base + goff[k]
      const x = GX[q]
      const y = GY[q]
      dG += gx[k] * x + gy[k] * y
      sGG += x * x + y * y
    }
    if (sGG > 1e-6) score += f.wGradient * (dG / Math.sqrt(sGG))
  }
  return score
}

/** Best score over integer shifts (cx ± radius, cy ± radius) of the window of cell (col,row). */
function bestShift(
  level: ReferenceLevel,
  v: PieceView,
  col: number,
  row: number,
  cx: number,
  cy: number,
  radius: number,
  f: FusionParams,
  cues: CueFlags,
): { score: number; dx: number; dy: number } {
  const span = level.size + 2 * level.margin
  const x0 = col * level.size
  const y0 = row * level.size
  let best = -Infinity
  let bdx = 0
  let bdy = 0
  for (let dy = cy - radius; dy <= cy + radius; dy++) {
    const y = y0 + dy
    if (y < 0 || y + span > level.height) continue
    for (let dx = cx - radius; dx <= cx + radius; dx++) {
      const x = x0 + dx
      if (x < 0 || x + span > level.width) continue
      const s = scoreWindow(level, v, y * level.width + x, f, cues)
      if (s > best) {
        best = s
        bdx = dx
        bdy = dy
      }
    }
  }
  return { score: best, dx: bdx, dy: bdy }
}

/** Indices of the `k` largest finite values of `scores` (first `count` entries), best first. */
function topIndices(scores: Float64Array, count: number, k: number): number[] {
  const idx: number[] = []
  for (let i = 0; i < count; i++) if (Number.isFinite(scores[i])) idx.push(i)
  idx.sort((a, b) => scores[b] - scores[a])
  if (idx.length > k) idx.length = k
  return idx
}

export function matchPiece(ref: ReferenceModel, piece: CanonicalPiece, opts: MatchOptions = {}): MatchResult {
  const t0 = performance.now()
  const f = resolveFusion(opts.fusion)
  const cues = cueFlags(opts.cues)
  const { cols, rows } = ref.grid
  const cellCount = cols * rows
  const excluded = opts.excludedCells
  const views = prepareViews(piece, ref.levels.map((l) => l.width))
  const t1 = performance.now()

  // Allowed hypotheses and their shape penalties.
  const sides = piece.shape.sides
  const flatByRot: boolean[][] = []
  for (let r = 0; r < 4; r++) {
    const flat: boolean[] = []
    for (let j = 0; j < 4; j++) flat.push(sides[(j - r + 4) % 4] === 'flat')
    flatByRot.push(flat)
  }
  const mismatches = (cell: number, r: number): number => {
    const col = cell % cols
    const row = (cell - col) / cols
    const fl = flatByRot[r]
    return (
      Number(fl[0] !== (row === 0)) +
      Number(fl[1] !== (col === cols - 1)) +
      Number(fl[2] !== (row === rows - 1)) +
      Number(fl[3] !== (col === 0))
    )
  }
  const hypCell = new Int32Array(cellCount * 4)
  const hypRot = new Uint8Array(cellCount * 4)
  const hypPenalty = new Float64Array(cellCount * 4)
  let count = 0
  const enumerate = (hard: boolean): void => {
    count = 0
    for (let cell = 0; cell < cellCount; cell++) {
      if (excluded?.has(cell) === true) continue
      for (let r = 0; r < 4; r++) {
        const mm = cues.shape ? mismatches(cell, r) : 0
        if (hard && mm > 0) continue
        hypCell[count] = cell
        hypRot[count] = r
        hypPenalty[count] = f.wShapeMismatch * mm
        count++
      }
    }
  }
  enumerate(cues.shape && piece.shape.confidence >= HARD_SHAPE_CONFIDENCE)
  if (count === 0) enumerate(false)

  // Level 0: every hypothesis, no shift.
  const coarse = ref.levels[0]
  const coarseViews = views[0]
  const scores = new Float64Array(count)
  for (let h = 0; h < count; h++) {
    const cell = hypCell[h]
    const col = cell % cols
    const row = (cell - col) / cols
    const base = row * coarse.size * coarse.width + col * coarse.size
    scores[h] = scoreWindow(coarse, coarseViews[hypRot[h]], base, f, cues) - hypPenalty[h]
  }
  const heat = coarseHeat(scores, hypCell, count, cellCount, f.coarseTemperature)
  const t2 = performance.now()

  // Level 1: shift search on the best coarse hypotheses.
  const mid = ref.levels[1]
  const midList = topIndices(scores, count, TOP_COARSE).map((h) => {
    const cell = hypCell[h]
    const col = cell % cols
    const row = (cell - col) / cols
    const best = bestShift(mid, views[1][hypRot[h]], col, row, 0, 0, SHIFT_MID, f, cues)
    return { h, cell, col, row, r: hypRot[h] as Rotation, score: best.score - hypPenalty[h], dx: best.dx, dy: best.dy }
  })
  midList.sort((a, b) => b.score - a.score)
  const t3 = performance.now()

  // Level 2: finalists.
  const fine = ref.levels[2]
  const finals = midList.slice(0, TOP_MID).map((m) => {
    const best = bestShift(fine, views[2][m.r], m.col, m.row, 2 * m.dx, 2 * m.dy, SHIFT_FINE, f, cues)
    return { ...m, score: best.score - hypPenalty[m.h] }
  })
  finals.sort((a, b) => b.score - a.score)

  // One entry per cell (its best rotation), then probabilities.
  const seen = new Set<number>()
  const perCell = finals.filter((c) => Number.isFinite(c.score) && !seen.has(c.cell) && (seen.add(c.cell), true))
  const top = perCell.length > 0 ? perCell[0].score : 0
  let z = 0
  const weights = perCell.map((c) => {
    const w = Math.exp((c.score - top) / f.temperature)
    z += w
    return w
  })
  const topK = opts.topK ?? DEFAULT_TOP_K
  const candidates: Candidate[] = perCell.slice(0, topK).map((c, i) => ({
    col: c.col,
    row: c.row,
    cell: c.cell,
    rotation: c.r,
    upAngleDeg: upAngleDeg(piece.shape.corners, c.r),
    score: c.score,
    prob: z > 0 ? (f.finalistMass * weights[i]) / z : 0,
  }))
  const t4 = performance.now()

  const p1 = candidates[0]?.prob ?? 0
  const p2 = candidates[1]?.prob ?? 0
  return {
    candidates,
    pieceType: piece.shape.pieceType,
    verdict: verdictFor(p1, p2),
    heat,
    timingsMs: { prepare: t1 - t0, coarse: t2 - t1, mid: t3 - t2, fine: t4 - t3, total: t4 - t0 },
  }
}

/** Per-cell softmax of the best coarse score over rotations. */
function coarseHeat(scores: Float64Array, cells: Int32Array, count: number, cellCount: number, temperature: number): Float32Array {
  const best = new Float64Array(cellCount).fill(-Infinity)
  let top = -Infinity
  for (let h = 0; h < count; h++) {
    const s = scores[h]
    if (s > best[cells[h]]) best[cells[h]] = s
    if (s > top) top = s
  }
  const heat = new Float32Array(cellCount)
  if (!Number.isFinite(top)) return heat
  let z = 0
  for (let c = 0; c < cellCount; c++) {
    if (!Number.isFinite(best[c])) continue
    const w = Math.exp((best[c] - top) / temperature)
    heat[c] = w
    z += w
  }
  for (let c = 0; c < cellCount; c++) heat[c] /= z
  return heat
}
