/**
 * Piece shape analysis: a piece outline → its 4 core corners, the kind of each side (flat / tab /
 * blank), the piece type (corner / edge / interior) and a confidence.
 *
 * Everything downstream hangs on the corners: the canonical warp maps them to a square, the matcher's
 * rotation is counted in quarter turns of them, and flat sides pin a border piece's position and
 * rotation. So this module must find the corners of the square-ish *core*, never a tab tip or the
 * shoulder of a blank, including under perspective (up to ~25° tilt), on irregular cuts (corner
 * angles far from 90°) and on outlines traced from segmentation masks (pixel staircase, side wall).
 *
 * Pipeline (prior art: Yao & Shao 2003 candidate quartets; puzzle-bot's shoulder-line refinement):
 *   1. Resample the outline at uniform arc length, ~SAMPLES_PER_CORE points per core side.
 *   2. Turning angle at scale TURN_SCALE × core. A core corner turns by its full exterior angle at
 *      any scale below the shoulder length; a round tab head turns only by ≈ scale / radius. Local
 *      maxima above CANDIDATE_MIN_TURN become candidates (core corners, blank mouths, tab heads).
 *   3. Score every clockwise 4-subset of candidates. Each side must leave both of its corners along
 *      straight shoulders that lie on the chord (a tab or blank only in the middle), each corner's
 *      local turn must equal the quad's exterior angle there (this rejects the "diamond of tab tips"),
 *      and the quad must have a plausible area, angles and side balance. Pairwise side features are
 *      computed once per candidate pair, so a quad costs O(1).
 *   4. Refine each corner as the intersection of lines fitted to its two adjacent shoulders
 *      (robust to dinged or rounded corners and to the mask's side-wall chamfer).
 *   5. Classify each side by its largest signed deviation from the corner-to-corner chord.
 *
 * Side wall (mode B, real masks): a segmentation mask covers the top face plus the piece's visible
 * cardboard wall, which parallax shows on the side facing the camera's nadir. Contours traced from
 * such a mask sit outward on that side. When `frameCenter` is given, each refined side line facing the
 * nadir is pulled back by the predicted wall width before the corners are intersected (see
 * wallDisplacement for the model and eval/shape-eval.ts for the measured effect).
 *
 * Conventions (types.ts): image coordinates, y down; contours clockwise on screen, so convex corners
 * have positive turning angles and a side's outward normal is its direction turned counter-clockwise.
 */

import { orientClockwise, perimeter, resampleClosed, signedArea, turningAngles } from './geom/index.ts'
import { scratchF64 } from './image/create.ts'
import type { PieceShape, PieceType, Point, Quad, SideKind } from './types.ts'

export interface ShapeOptions {
  /**
   * The frame point below the camera (the principal point, ≈ the frame centre for a phone held
   * flat). Set it when the contour comes from a segmentation mask that includes the side wall: the
   * sides facing this point are then pulled back by the predicted wall width. Leave it unset for
   * outlines of the printed top face (e.g. ground-truth outlines).
   */
  frameCenter?: Point
  /**
   * Camera focal length in frame px, used with frameCenter. Default: DEFAULT_FOCAL_PER_LONG_SIDE ×
   * the frame's long side, the frame being assumed centred on frameCenter.
   */
  focalPx?: number
  /** Board thickness ÷ core side, used with frameCenter. Default BOARD_THICKNESS. */
  thickness?: number
}

// ---------------------------------------------------------------------------------------------------
// Thresholds. "Measured" = chosen from eval/shape-eval.ts on datasets/smoke, smoke1000 and v1/val*;
// "guessed" = set from geometry and never tuned.
// ---------------------------------------------------------------------------------------------------

/** Fewer contour points than this cannot describe a piece. Guessed. */
const MIN_CONTOUR_POINTS = 8
/** Outlines smaller than this (px²) are noise, not pieces. Guessed: an 8 × 8 px piece is unusable. */
const MIN_AREA_PX = 64
/** Resampling density along the outline, per core side (√area). Measured: 48 → ~0.02 core spacing. */
const SAMPLES_PER_CORE = 48
const MIN_SAMPLES = 96
const MAX_SAMPLES = 640
/**
 * Turning-angle scale as a fraction of the core. It must stay below the shoulder length (≥ 0.27 of a
 * side on real and synthetic cuts) and above pixel noise and blank-mouth fillets (≤ 0.04 core).
 * From the task spec (core / 12); measured.
 */
const TURN_SCALE = 1 / 12
/** Candidates must turn at least this much (rad). Measured: true corners turn ≥ 55°; tab heads ~40°. */
const CANDIDATE_MIN_TURN = (40 * Math.PI) / 180
/** Keep at most this many candidates (strongest turns). Guessed: 4 corners + 2 per blank + tab heads. */
const MAX_CANDIDATES = 16
/** A side's chord must be at least this fraction of the core (√area). Guessed. */
const MIN_SIDE_CORE = 0.35
/** A side's arc can be at most this fraction of the whole perimeter. Guessed (worst real case ~0.4). */
const MAX_SIDE_ARC_FRACTION = 0.55
/** Shoulder window checked at each end of a side, as a fraction of the chord. Tabs start ≥ 0.27. Guessed. */
const SHOULDER_CHECK = 0.2
/** The first few samples at a corner are skipped in shoulder checks (rounded corners, wall chamfer). */
const CORNER_SKIP = 0.03

// Quad cost terms. Each is (excess / tolerance)², so ~1 per term is "at the edge of normal".
/** Shoulder deviation from the chord, fraction of the chord. Guessed from shoulder slopes ≤ 4° + bow. */
const SHOULDER_TOL = 0.03
/** |local turn − quad exterior angle| (rad). Guessed: shoulder slopes ±4° per side + rounding. */
const TURN_MATCH_TOL = (12 * Math.PI) / 180
/** Interior angles beyond 90° ± ANGLE_FREE (rad) start to cost. Guessed: irregular cuts + tilt. */
const ANGLE_FREE = (25 * Math.PI) / 180
const ANGLE_TOL = (10 * Math.PI) / 180
/** Quad area ÷ outline area outside [AREA_MIN, AREA_MAX] costs. Guessed: 4 tabs ≈ 0.8, 4 blanks ≈ 1.3. */
const AREA_MIN = 0.68
const AREA_MAX = 1.4
const AREA_TOL = 0.08
/** Opposite sides may differ by this ratio for free (perspective + irregular jitter). Guessed. */
const OPPOSITE_RATIO_FREE = 1.45
/** Adjacent sides may differ by this ratio for free (cells up to 1.25:1 + jitter + tilt). Guessed. */
const ADJACENT_RATIO_FREE = 1.75
const RATIO_TOL = 0.1
/** The arc of a side may project at most this far beyond its chord ends (fraction of chord). Guessed. */
const OVERHANG_FREE = 0.12
/** A side's excursion (tab height / blank depth) beyond this fraction of the chord costs. Guessed. */
const EXCURSION_FREE = 0.5
const EXCURSION_TOL = 0.05
/** Small reward for larger quads (per unit of area ratio), to break near-ties. Guessed. */
const AREA_REWARD = 1.0
/** Cost at which confidence has dropped to 1/e. Measured (calibration table in shape-eval). */
const CONFIDENCE_COST_SCALE = 8
/** Best-quad cost above this means "not a piece": return null. Measured. */
const MAX_COST = 200

// Refinement.
/** Shoulder fit window along a tabbed/blank side: [REFINE_SKIP, REFINE_TABBED] of the chord. Measured. */
const REFINE_SKIP = 0.04
const REFINE_TABBED = 0.22
/** On a flat side the fit may run this far (fraction of the chord). Measured. */
const REFINE_FLAT = 0.45
/** Lines closer than this to parallel (sin of the angle) are not intersected. Guessed. */
const MIN_INTERSECT_SIN = Math.sin((25 * Math.PI) / 180)
/** A refined corner further than this (fraction of core) from its candidate is rejected. Guessed. */
const MAX_REFINE_SHIFT = 0.15
/** IRLS robust fit: Cauchy scale as a fraction of the chord, floor in px. Guessed. */
const FIT_SCALE = 0.01
const FIT_SCALE_MIN_PX = 0.75

// Side classification.
/** |signed deviation| / chord below this is a flat side. Measured: flats ≤ 0.03, tabs/blanks ≥ 0.17. */
const FLAT_MAX_DEVIATION = 0.1

// Side wall (only with frameCenter).
/** Board thickness ÷ core side. Measured on real pieces (tools/synth/render.py): 0.085–0.12. */
const BOARD_THICKNESS = 0.1
/**
 * Focal length ÷ long frame side when focalPx is not given: the iPhone main camera's ~69° field of view
 * across the long side (26 mm equivalent). Guessed from the spec sheet; the synth draws 0.65–0.81.
 */
const DEFAULT_FOCAL_PER_LONG_SIDE = 0.73

const DEG = Math.PI / 180

/**
 * Analyses a piece outline (clockwise in image coordinates, dense). Returns the 4 core corners
 * (clockwise, arbitrary start), side kinds, piece type and confidence, or null when the outline is
 * clearly not a jigsaw piece.
 */
export function analyzeShape(contour: Point[], opts: ShapeOptions = {}): PieceShape | null {
  if (contour.length < MIN_CONTOUR_POINTS) return null
  const outline = orientClockwise(contour)
  const area = signedArea(outline)
  if (!(area >= MIN_AREA_PX)) return null
  const core = Math.sqrt(area)

  // 1. Uniform resampling.
  const per = perimeter(outline)
  const n = Math.max(MIN_SAMPLES, Math.min(MAX_SAMPLES, Math.round((per / core) * SAMPLES_PER_CORE)))
  const pts = resampleClosed(outline, n)
  const h = per / n
  const xs = scratchF64('shape.xs', n)
  const ys = scratchF64('shape.ys', n)
  for (let i = 0; i < n; i++) {
    xs[i] = pts[i].x
    ys[i] = pts[i].y
  }

  // 2. Turning angles and candidates.
  const turn = turningAngles(pts, TURN_SCALE * core, scratchF64('shape.turn', n).subarray(0, n))
  const cand = findCandidates(turn, n, Math.max(1, Math.round((TURN_SCALE * core) / h)))
  const k = cand.length
  if (k < 4) return null

  // 3. Pairwise side features, then every clockwise quad.
  const sides = sideFeatures(xs, ys, n, h, cand, core)
  const best = bestQuad(xs, ys, turn, cand, sides, area)
  if (best === null || best.cost > MAX_COST) return null

  // 4–5. Classify with the candidate corners, refine with those kinds, classify again.
  const idx = best.idx
  let corners = idx.map((c) => ({ x: xs[cand[c]], y: ys[cand[c]] })) as Quad
  let kinds = classifySides(xs, ys, n, cand, idx, corners)
  const lines = shoulderLines(xs, ys, n, h, cand, idx, corners, kinds)
  if (opts.frameCenter !== undefined) pullBackWall(lines, corners, core, opts)
  corners = intersectLines(lines, corners, core)
  kinds = classifySides(xs, ys, n, cand, idx, corners)

  let flats = 0
  for (const s of kinds) if (s === 'flat') flats++
  if (flats >= 3) return null
  const opposite = flats === 2 && kinds[0] === kinds[2]
  const pieceType: PieceType = flats === 2 ? 'corner' : flats === 1 ? 'edge' : 'interior'
  const confidence = Math.exp(-Math.max(0, best.cost) / CONFIDENCE_COST_SCALE) * (opposite ? 0.25 : 1)
  return { corners, sides: kinds, pieceType, confidence }
}

// ---------------------------------------------------------------------------------------------------
// Candidates
// ---------------------------------------------------------------------------------------------------

/** Sample indices of local turning maxima (±w samples) above CANDIDATE_MIN_TURN, strongest first, then sorted by index. */
function findCandidates(turn: Float64Array, n: number, w: number): number[] {
  const found: number[] = []
  for (let i = 0; i < n; i++) {
    const t = turn[i]
    if (t < CANDIDATE_MIN_TURN) continue
    let isMax = true
    for (let d = 1; d <= w && isMax; d++) {
      if (turn[(i + d) % n] > t || turn[(i - d + n) % n] >= t) isMax = false
    }
    if (isMax) found.push(i)
  }
  found.sort((a, b) => turn[b] - turn[a])
  if (found.length > MAX_CANDIDATES) found.length = MAX_CANDIDATES
  return found.sort((a, b) => a - b)
}

// ---------------------------------------------------------------------------------------------------
// Side features and quad search
// ---------------------------------------------------------------------------------------------------

/** Per ordered candidate pair (a → b, forward along the contour): chord length and a side cost. */
interface SideTable {
  k: number
  /** Chord length, or −1 when the pair cannot be a side. */
  len: Float64Array
  /** Shoulder + overhang + excursion cost of the pair as a side. */
  cost: Float64Array
}

function sideFeatures(xs: Float64Array, ys: Float64Array, n: number, h: number, cand: number[], core: number): SideTable {
  const k = cand.length
  const len = scratchF64('shape.sideLen', k * k).subarray(0, k * k)
  const cost = scratchF64('shape.sideCost', k * k).subarray(0, k * k)
  len.fill(-1)
  cost.fill(0)
  const maxArc = MAX_SIDE_ARC_FRACTION * n
  for (let a = 0; a < k; a++) {
    for (let b = 0; b < k; b++) {
      if (a === b) continue
      const ia = cand[a]
      const ib = cand[b]
      const arc = (ib - ia + n) % n
      if (arc > maxArc) continue
      const dx = xs[ib] - xs[ia]
      const dy = ys[ib] - ys[ia]
      const L = Math.hypot(dx, dy)
      if (L < MIN_SIDE_CORE * core) continue
      const ux = dx / L
      const uy = dy / L
      // Shoulder windows at both ends, by arc length (shoulders run along the chord).
      const m = Math.max(2, Math.round((SHOULDER_CHECK * L) / h))
      const skip = Math.round((CORNER_SKIP * L) / h)
      let devStart = 0
      let devEnd = 0
      let tMin = 0
      let tMax = 1
      let excursion = 0
      for (let j = 1; j < arc; j++) {
        const i = (ia + j) % n
        const px = xs[i] - xs[ia]
        const py = ys[i] - ys[ia]
        const t = (px * ux + py * uy) / L
        const d = Math.abs(px * uy - py * ux) / L
        if (j > skip && j <= m && d > devStart) devStart = d
        if (arc - j > skip && arc - j <= m && d > devEnd) devEnd = d
        if (t < tMin) tMin = t
        if (t > tMax) tMax = t
        if (d > excursion) excursion = d
      }
      const overhang = Math.max(0, -tMin - OVERHANG_FREE) + Math.max(0, tMax - 1 - OVERHANG_FREE)
      const exc = Math.max(0, excursion - EXCURSION_FREE)
      len[a * k + b] = L
      cost[a * k + b] =
        (devStart / SHOULDER_TOL) ** 2 + (devEnd / SHOULDER_TOL) ** 2 + (overhang / EXCURSION_TOL) ** 2 + (exc / EXCURSION_TOL) ** 2
    }
  }
  return { k, len, cost }
}

interface QuadChoice {
  /** Candidate indices (into cand), clockwise. */
  idx: [number, number, number, number]
  cost: number
}

function bestQuad(xs: Float64Array, ys: Float64Array, turn: Float64Array, cand: number[], sides: SideTable, area: number): QuadChoice | null {
  const { k, len, cost } = sides
  let best: QuadChoice | null = null
  const q = [0, 0, 0, 0]
  const vx = [0, 0, 0, 0]
  const vy = [0, 0, 0, 0]
  const L = [0, 0, 0, 0]
  for (let a = 0; a < k; a++) {
    for (let b = a + 1; b < k; b++) {
      if (len[a * k + b] < 0) continue
      for (let c = b + 1; c < k; c++) {
        if (len[b * k + c] < 0) continue
        for (let d = c + 1; d < k; d++) {
          if (len[c * k + d] < 0 || len[d * k + a] < 0) continue
          q[0] = a
          q[1] = b
          q[2] = c
          q[3] = d
          let total = cost[a * k + b] + cost[b * k + c] + cost[c * k + d] + cost[d * k + a]
          if (best !== null && total >= best.cost + AREA_REWARD * AREA_MAX) continue
          for (let i = 0; i < 4; i++) {
            const p = cand[q[i]]
            const nx = cand[q[(i + 1) & 3]]
            vx[i] = xs[nx] - xs[p]
            vy[i] = ys[nx] - ys[p]
            L[i] = len[q[i] * k + q[(i + 1) & 3]]
          }
          // Corners: convex, plausible angle, and the local turn equals the quad's exterior angle.
          let convex = true
          for (let i = 0; i < 4 && convex; i++) {
            const j = (i + 3) & 3 // incoming side
            const cross = vx[j] * vy[i] - vy[j] * vx[i]
            if (cross <= 0) convex = false
            const exterior = Math.atan2(cross, vx[j] * vx[i] + vy[j] * vy[i])
            const interior = Math.PI - exterior
            total += ((turn[cand[q[i]]] - exterior) / TURN_MATCH_TOL) ** 2
            total += (Math.max(0, Math.abs(interior - 90 * DEG) - ANGLE_FREE) / ANGLE_TOL) ** 2
          }
          if (!convex) continue
          // Global: area, opposite and adjacent side balance.
          const quadArea = 0.5 * Math.abs(vx[0] * vy[1] - vy[0] * vx[1] + vx[2] * vy[3] - vy[2] * vx[3])
          const r = quadArea / area
          total += (Math.max(0, AREA_MIN - r, r - AREA_MAX) / AREA_TOL) ** 2 - AREA_REWARD * r
          const ro1 = Math.max(L[0], L[2]) / Math.min(L[0], L[2])
          const ro2 = Math.max(L[1], L[3]) / Math.min(L[1], L[3])
          total += (Math.max(0, ro1 - OPPOSITE_RATIO_FREE) / RATIO_TOL) ** 2 + (Math.max(0, ro2 - OPPOSITE_RATIO_FREE) / RATIO_TOL) ** 2
          for (let i = 0; i < 4; i++) {
            const ra = Math.max(L[i], L[(i + 1) & 3]) / Math.min(L[i], L[(i + 1) & 3])
            total += (Math.max(0, ra - ADJACENT_RATIO_FREE) / RATIO_TOL) ** 2
          }
          if (best === null || total < best.cost) best = { idx: [a, b, c, d], cost: total }
        }
      }
    }
  }
  if (best !== null) best.cost += AREA_REWARD // cost 0 ≈ a perfect quad of area ratio 1
  return best
}

// ---------------------------------------------------------------------------------------------------
// Side classification
// ---------------------------------------------------------------------------------------------------

/** Kind of each side (corner i → i+1) by its largest signed deviation from the chord. */
function classifySides(
  xs: Float64Array,
  ys: Float64Array,
  n: number,
  cand: number[],
  idx: readonly number[],
  corners: Quad,
): [SideKind, SideKind, SideKind, SideKind] {
  const out: SideKind[] = []
  for (let s = 0; s < 4; s++) {
    const A = corners[s]
    const B = corners[(s + 1) & 3]
    const L = Math.hypot(B.x - A.x, B.y - A.y)
    // Outward normal of a clockwise contour: the side direction turned counter-clockwise on screen.
    const nx = (B.y - A.y) / L
    const ny = -(B.x - A.x) / L
    const ia = cand[idx[s]]
    const arc = (cand[idx[(s + 1) & 3]] - ia + n) % n
    let out_ = 0
    let in_ = 0
    for (let j = 1; j < arc; j++) {
      const i = (ia + j) % n
      const d = ((xs[i] - A.x) * nx + (ys[i] - A.y) * ny) / L
      if (d > out_) out_ = d
      if (-d > in_) in_ = -d
    }
    out.push(Math.max(out_, in_) < FLAT_MAX_DEVIATION ? 'flat' : out_ > in_ ? 'tab' : 'blank')
  }
  return out as [SideKind, SideKind, SideKind, SideKind]
}

// ---------------------------------------------------------------------------------------------------
// Corner refinement
// ---------------------------------------------------------------------------------------------------

/** A line through (px, py) with unit direction (ux, uy). */
interface Line {
  px: number
  py: number
  ux: number
  uy: number
}

/**
 * Two lines per corner: `in` fitted to the end of the incoming side's contour, `out` to the start of
 * the outgoing side's. lines[2i] = corner i incoming, lines[2i+1] = corner i outgoing.
 */
function shoulderLines(
  xs: Float64Array,
  ys: Float64Array,
  n: number,
  h: number,
  cand: number[],
  idx: readonly number[],
  corners: Quad,
  kinds: readonly SideKind[],
): (Line | null)[] {
  const lines: (Line | null)[] = []
  for (let c = 0; c < 4; c++) {
    const prev = (c + 3) & 3
    const next = (c + 1) & 3
    const ic = cand[idx[c]]
    // Incoming side prev → c: walk backwards from the corner.
    const Lin = dist(corners[prev], corners[c])
    const arcIn = (ic - cand[idx[prev]] + n) % n
    lines.push(fitWindow(xs, ys, n, ic, -1, arcIn, Lin, h, kinds[prev] === 'flat'))
    const Lout = dist(corners[c], corners[next])
    const arcOut = (cand[idx[next]] - ic + n) % n
    lines.push(fitWindow(xs, ys, n, ic, 1, arcOut, Lout, h, kinds[c] === 'flat'))
  }
  return lines
}

/** Robust total-least-squares line through samples [skip, reach] steps from `from` in direction `dir`. */
function fitWindow(
  xs: Float64Array,
  ys: Float64Array,
  n: number,
  from: number,
  dir: 1 | -1,
  arc: number,
  L: number,
  h: number,
  flat: boolean,
): Line | null {
  const j0 = Math.max(1, Math.round((REFINE_SKIP * L) / h))
  const j1 = Math.min(arc - 1, Math.round(((flat ? REFINE_FLAT : REFINE_TABBED) * L) / h))
  if (j1 - j0 < 2) return null
  const count = j1 - j0 + 1
  const wx = scratchF64('shape.fitX', count)
  const wy = scratchF64('shape.fitY', count)
  const ww = scratchF64('shape.fitW', count)
  for (let j = 0; j < count; j++) {
    const i = (((from + dir * (j0 + j)) % n) + n) % n
    wx[j] = xs[i]
    wy[j] = ys[i]
    ww[j] = 1
  }
  const scale = Math.max(FIT_SCALE * L, FIT_SCALE_MIN_PX)
  let line: Line | null = null
  // IRLS with Cauchy weights: two reweighting passes are enough for shoulder-sized windows.
  for (let iter = 0; iter < 3; iter++) {
    line = weightedLine(wx, wy, ww, count)
    if (line === null) return null
    for (let j = 0; j < count; j++) {
      const r = ((wx[j] - line.px) * line.uy - (wy[j] - line.py) * line.ux) / scale
      ww[j] = 1 / (1 + r * r)
    }
  }
  return line
}

function weightedLine(xs: Float64Array, ys: Float64Array, w: Float64Array, count: number): Line | null {
  let sw = 0
  let mx = 0
  let my = 0
  for (let j = 0; j < count; j++) {
    sw += w[j]
    mx += w[j] * xs[j]
    my += w[j] * ys[j]
  }
  if (!(sw > 0)) return null
  mx /= sw
  my /= sw
  let sxx = 0
  let sxy = 0
  let syy = 0
  for (let j = 0; j < count; j++) {
    const dx = xs[j] - mx
    const dy = ys[j] - my
    sxx += w[j] * dx * dx
    sxy += w[j] * dx * dy
    syy += w[j] * dy * dy
  }
  // Principal direction of the 2×2 scatter matrix.
  const theta = 0.5 * Math.atan2(2 * sxy, sxx - syy)
  return { px: mx, py: my, ux: Math.cos(theta), uy: Math.sin(theta) }
}

/** Corner i = intersection of its incoming and outgoing lines; the candidate when that fails. */
function intersectLines(lines: (Line | null)[], fallback: Quad, core: number): Quad {
  const out: Point[] = []
  for (let c = 0; c < 4; c++) {
    const a = lines[2 * c]
    const b = lines[2 * c + 1]
    const f = fallback[c]
    if (a === null || b === null) {
      out.push({ x: f.x, y: f.y })
      continue
    }
    const det = a.ux * b.uy - a.uy * b.ux
    if (Math.abs(det) < MIN_INTERSECT_SIN) {
      out.push({ x: f.x, y: f.y })
      continue
    }
    const t = ((b.px - a.px) * b.uy - (b.py - a.py) * b.ux) / det
    const p = { x: a.px + t * a.ux, y: a.py + t * a.uy }
    out.push(dist(p, f) <= MAX_REFINE_SHIFT * core ? p : { x: f.x, y: f.y })
  }
  return out as Quad
}

// ---------------------------------------------------------------------------------------------------
// Side wall
// ---------------------------------------------------------------------------------------------------

/**
 * Pulls each fitted line back by the side wall it carries. Model (tools/synth/scene.py, and plain
 * parallax): the wall's foot is the top face scaled towards the nadir by (D − T) / D, so in the frame
 * the mask is the top face swept by d = (nadir − p) · T / f, with T = thickness × core. A side whose
 * outward normal n has d·n > 0 is seen displaced outward by d·n; the others are not displaced.
 */
function pullBackWall(lines: (Line | null)[], corners: Quad, core: number, opts: ShapeOptions): void {
  const c = opts.frameCenter as Point
  const focal = opts.focalPx ?? DEFAULT_FOCAL_PER_LONG_SIDE * (2 * Math.max(c.x, c.y) + 1)
  const k = ((opts.thickness ?? BOARD_THICKNESS) * core) / focal
  const mx = 0.25 * (corners[0].x + corners[1].x + corners[2].x + corners[3].x)
  const my = 0.25 * (corners[0].y + corners[1].y + corners[2].y + corners[3].y)
  const dx = (c.x - mx) * k
  const dy = (c.y - my) * k
  for (const line of lines) {
    if (line === null) continue
    // Outward normal: this line's side runs clockwise, but a fitted direction has an arbitrary sign,
    // so orient the normal away from the core centre.
    let nx = line.uy
    let ny = -line.ux
    if ((line.px - mx) * nx + (line.py - my) * ny < 0) {
      nx = -nx
      ny = -ny
    }
    const s = dx * nx + dy * ny
    if (s > 0) {
      line.px -= s * nx
      line.py -= s * ny
    }
  }
}

function dist(a: Point, b: Point): number {
  return Math.hypot(b.x - a.x, b.y - a.y)
}

