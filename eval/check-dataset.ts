/**
 * Dataset contract check: `node eval/check-dataset.ts datasets/smoke`.
 *
 * Three parts of Piece Finder were written independently: the Python generator (tools/synth), the
 * TypeScript engine primitives (src/engine) and the dataset contract (docs/DATASET.md + types.ts).
 * If they disagree on a convention (pixel centres, corner order, grid indexing, clockwise, angle
 * direction), every metric computed later is silently wrong. This script reads a dataset with sharp
 * and checks it numerically, *through the engine's own primitives*, so it tests the engine and the
 * generator against each other rather than each against itself:
 *
 *   (a) referenceCorners + engine homography/warp rectify reference.jpg onto motif.png (ZNCC, and
 *       the sub-pixel shift that would improve it — a half-pixel convention error shows up there);
 *   (b) each scene piece's frame corners (motif order TL,TR,BR,BL) warped to a square match its motif
 *       cell (ZNCC), the true rotation beats the other three, and an exhaustive search over every
 *       cell × 4 rotations lands on the labelled (col,row) — so corners, grid indexing and orientation
 *       all agree; pieces.json corners must sit on the cut lattice of cell (col,row);
 *   (c) upAngleDeg equals the frame direction of the motif-up vector (types.ts definition);
 *   (d) the instance mask agrees with the outline rasterised by the engine: the whole top face is in the
 *       mask, and every mask pixel outside the outline lies within the side-wall band (DATASET.md: the
 *       mask is top face + visible side wall, the outline is the top face). Raw IoU is reported too;
 *   (e) the engine contour tracer returns clockwise contours that sit on the outline — on the engine's
 *       own raster of the outline (tracer + rasteriser pixel conventions), and on the instance mask
 *       along the sides where no wall shows.
 *
 * Exits 1 when a check fails its threshold. Thresholds below say whether they are measured or guessed.
 */

import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import sharp from 'sharp'
import {
  composeHomography,
  findContours,
  homographyFromQuad,
  isClockwise,
  pointSegmentDistance,
  rasterizePolygon,
  rectToQuadHomography,
  signedArea,
  warpGray,
  warpMask,
  type Homography,
} from '../src/engine/geom/index.ts'
import { distanceTransform, gaussianBlur, resizeAreaGray, rgbaToGray } from '../src/engine/image/index.ts'
import type { GrayImage, Mask, Point, Quad } from '../src/engine/types.ts'

// ---------------------------------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------------------------------

/** (a) Rectified reference vs motif. Guessed: lids carry badges/logos/glare, so perfect is impossible. */
const REF_ZNCC_MIN = 0.5
/** (a)/(b) Largest acceptable mean alignment offset, in output px. Guessed: 1/4 px is well below blur. */
const MAX_MEAN_SHIFT_PX = 0.25
/** (b) Median ZNCC of textured, ≥60 %-visible face-up pieces vs their cell. Guessed from synth's 0.5. */
const PIECE_ZNCC_MEDIAN_MIN = 0.5
/** (b) Fraction of textured pieces whose true rotation wins. Guessed (synth's own check uses 0.9). */
const ROTATION_WIN_MIN = 0.9
/** (b) Fraction of textured pieces whose exhaustive search (all cells × 4 rotations) is correct. Guessed. */
const EXHAUSTIVE_TOP1_MIN = 0.8
/** (b) A motif patch with grey std below this (0..255) is "untextured" and excluded from rankings. Guessed. */
const TEXTURE_STD_MIN = 8
/** (b) Grid-cut lattice points must match pieces.json corners within this many px (rounding is 0.01). */
const LATTICE_TOL_PX = 0.02
/** (b) Irregular cuts jitter interior lattice points by ≤ 12 % of a cell (tools/synth/cut.py). */
const IRREGULAR_JITTER = 0.12
/** (c) Max |upAngleDeg − motif-up direction|. Guessed: generator rounds to 0.1°, perspective adds a bit. */
const UP_ANGLE_TOL_DEG = 1.0
/**
 * (d) Reported only: IoU of mask vs rasterised outline. It cannot be a hard gate at 0.95 because the mask
 * legitimately includes the visible side wall (measured on smoke: median 0.94, min 0.86, all explained
 * by the wall band below).
 */
const MASK_IOU_REPORT = 0.95
/** (d) Share of the rasterised top face that must be in the mask. Measured 1.000 on smoke; gate guessed. */
const TOP_FACE_RECALL_MIN = 0.97
/**
 * (d) Mask pixels outside the outline must be within this distance of it: the side wall. Board thickness
 * ≤ 0.12 × core (tools/synth/render.py, measured on real pieces) seen far off-axis on a tilted frame gives
 * ≤ ~0.14 × core; 0.16 × core + 1.5 px as in tools/synth/test_synth.py (guessed margin).
 */
const WALL_BAND_CORE = 0.16
const WALL_BAND_PX = 1.5
/** (e) On the engine's own raster, traced pixel centres lie within this of the outline (p99, px). Guessed. */
const RASTER_TRACE_P99_MAX = 1.0
/** (e) Median distance, outline → traced mask contour, px (the wall-free sides dominate). Guessed. */
const MASK_TRACE_MEDIAN_MAX = 1.0

/** Patch size for the exhaustive cell search (px). Small enough for 100 cells × 4 rotations per piece. */
const SEARCH_SIZE = 48
/** Output size for the reference rectification check (long side, px). */
const REF_LONG_SIDE = 640
/** Sub-pixel shift search: ±SHIFT_RANGE output px in SHIFT_STEP steps. */
const SHIFT_RANGE = 1.5
const SHIFT_STEP = 0.25
/** Pixels of the piece's top face within this distance of its outline are ignored in ZNCC (cut rim). */
const RIM_ERODE_PX = 2

// ---------------------------------------------------------------------------------------------------
// Dataset types (docs/DATASET.md)
// ---------------------------------------------------------------------------------------------------

type XY = [number, number]
interface Meta {
  puzzles: string[]
  scenes: string[]
}
interface ReferenceJson {
  cols: number
  rows: number
  motifSize: XY
  referenceCorners: [XY, XY, XY, XY]
  cut: 'grid' | 'irregular'
}
interface PieceJson {
  id: number
  col: number
  row: number
  cell: number
  corners: [XY, XY, XY, XY]
  outline: XY[]
}
interface ScenePieceJson {
  index: number
  pieceId: number
  col: number
  row: number
  cell: number
  upAngleDeg: number
  corners: [XY, XY, XY, XY]
  outline: XY[]
  visibleFraction: number
  faceUp: boolean
}
interface SceneJson {
  puzzleId: string
  width: number
  height: number
  pieces: ScenePieceJson[]
}

const pt = ([x, y]: XY): Point => ({ x, y })
const quad = (c: readonly XY[]): Quad => [pt(c[0]), pt(c[1]), pt(c[2]), pt(c[3])]

// ---------------------------------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------------------------------

async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, 'utf8')) as T
}

async function loadGray(path: string): Promise<GrayImage> {
  const { data, info } = await sharp(path).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  if (info.channels !== 4) throw new Error(`${path}: expected RGBA, got ${info.channels} channels`)
  return rgbaToGray({ width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) })
}

async function loadLabels(path: string): Promise<Mask> {
  const { data, info } = await sharp(path).extractChannel(0).raw().toBuffer({ resolveWithObject: true })
  return { width: info.width, height: info.height, data: new Uint8Array(data.buffer, data.byteOffset, data.length) }
}

/** A copy of `rect` of `img` (clipped to the image) and its offset. */
function cropGray(img: GrayImage, x0: number, y0: number, x1: number, y1: number): { img: GrayImage; ox: number; oy: number } {
  const ox = Math.max(0, Math.floor(x0))
  const oy = Math.max(0, Math.floor(y0))
  const w = Math.min(img.width, Math.ceil(x1) + 1) - ox
  const h = Math.min(img.height, Math.ceil(y1) + 1) - oy
  const data = new Float32Array(w * h)
  for (let y = 0; y < h; y++) data.set(img.data.subarray((oy + y) * img.width + ox, (oy + y) * img.width + ox + w), y * w)
  return { img: { width: w, height: h, data }, ox, oy }
}

/** Zero-normalised cross-correlation over pixels where `valid` (if given) is nonzero. */
function zncc(a: Float32Array, b: Float32Array, valid?: Uint8Array): number {
  let n = 0
  let sa = 0
  let sb = 0
  for (let i = 0; i < a.length; i++) {
    if (valid !== undefined && valid[i] === 0) continue
    n++
    sa += a[i]
    sb += b[i]
  }
  if (n < 16) return NaN
  const ma = sa / n
  const mb = sb / n
  let ab = 0
  let aa = 0
  let bb = 0
  for (let i = 0; i < a.length; i++) {
    if (valid !== undefined && valid[i] === 0) continue
    const da = a[i] - ma
    const db = b[i] - mb
    ab += da * db
    aa += da * da
    bb += db * db
  }
  return aa > 0 && bb > 0 ? ab / Math.sqrt(aa * bb) : NaN
}

function stdDev(a: Float32Array, valid?: Uint8Array): number {
  let n = 0
  let s = 0
  let ss = 0
  for (let i = 0; i < a.length; i++) {
    if (valid !== undefined && valid[i] === 0) continue
    n++
    s += a[i]
    ss += a[i] * a[i]
  }
  return n > 0 ? Math.sqrt(Math.max(0, ss / n - (s / n) ** 2)) : 0
}

/** dst→src homography with the output shifted by (dx, dy) px: output pixel p samples where p + (dx, dy) did. */
function shifted(H: Homography, dx: number, dy: number): Homography {
  return composeHomography(H, Float64Array.of(1, 0, dx, 0, 1, dy, 0, 0, 1))
}

/**
 * The output shift (dx, dy) that maximises ZNCC(render(dx, dy), target), by grid search with a
 * parabolic refinement on each axis. Non-zero means the two sides disagree about where things are.
 */
function bestShift(render: (dx: number, dy: number) => Float32Array, target: Float32Array, valid?: Uint8Array): { dx: number; dy: number; score: number } {
  const steps = Math.round(SHIFT_RANGE / SHIFT_STEP)
  const n = 2 * steps + 1
  const grid = new Float64Array(n * n)
  let best = 0
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const s = zncc(render((i - steps) * SHIFT_STEP, (j - steps) * SHIFT_STEP), target, valid)
      grid[j * n + i] = Number.isNaN(s) ? -Infinity : s
      if (grid[j * n + i] > grid[best]) best = j * n + i
    }
  }
  const bi = best % n
  const bj = Math.floor(best / n)
  const refine = (m: number, c: number, p: number): number => {
    const den = m - 2 * c + p
    return Number.isFinite(den) && den < 0 ? (0.5 * (m - p)) / den : 0
  }
  const ox = bi > 0 && bi < n - 1 ? refine(grid[bj * n + bi - 1], grid[best], grid[bj * n + bi + 1]) : 0
  const oy = bj > 0 && bj < n - 1 ? refine(grid[(bj - 1) * n + bi], grid[best], grid[(bj + 1) * n + bi]) : 0
  return { dx: (bi - steps + ox) * SHIFT_STEP, dy: (bj - steps + oy) * SHIFT_STEP, score: grid[best] }
}

/** Shortest distance from p to the closed polyline `poly`. */
function distToPolyline(p: Point, poly: readonly Point[]): number {
  let d = Infinity
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) d = Math.min(d, pointSegmentDistance(p, poly[j], poly[i]))
  return d
}

/** Angle in degrees, clockwise from image-up, in [0, 360). */
function upAngle(dx: number, dy: number): number {
  const a = (Math.atan2(dx, -dy) * 180) / Math.PI
  return (a + 360) % 360
}

function angleDiff(a: number, b: number): number {
  const d = Math.abs(a - b) % 360
  return d > 180 ? 360 - d : d
}

function median(v: number[]): number {
  if (v.length === 0) return NaN
  const s = [...v].sort((a, b) => a - b)
  return s.length % 2 ? s[(s.length - 1) >> 1] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2])
}
const mean = (v: number[]): number => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN)
const fmt = (v: number, d = 3): string => (Number.isFinite(v) ? v.toFixed(d) : String(v))

// ---------------------------------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------------------------------

const failures: string[] = []
function expect(ok: boolean, what: string): void {
  if (!ok) failures.push(what)
}

interface Puzzle {
  id: string
  ref: ReferenceJson
  pieces: Map<number, PieceJson>
  motif: GrayImage
  /** Motif blurred for SEARCH_SIZE patches, and every cell's patch at that size. */
  searchPatches: Map<number, Float32Array>
  cellPx: number
}

/** (a) Reference rectification, and the pieces.json lattice check from (b). */
async function checkPuzzle(root: string, id: string): Promise<Puzzle> {
  const dir = join(root, 'puzzles', id)
  const ref = await readJson<ReferenceJson>(join(dir, 'reference.json'))
  const piecesArr = await readJson<PieceJson[]>(join(dir, 'pieces.json'))
  const motif = await loadGray(join(dir, 'motif.png'))
  const [mw, mh] = ref.motifSize
  expect(motif.width === mw && motif.height === mh, `${id}: motif.png is ${motif.width}x${motif.height}, reference.json says ${mw}x${mh}`)

  // (a) Downscale both sides with the engine's area resize, mapping corners with the pixel-centre rule
  // p' = (p + 0.5) / k − 0.5, then rectify with rectToQuadHomography (quad → outer pixel corners).
  const s = REF_LONG_SIDE / Math.max(mw, mh)
  const ow = Math.round(mw * s)
  const oh = Math.round(mh * s)
  const motifSmall = resizeAreaGray(motif, ow, oh)
  const refFull = await loadGray(join(dir, 'reference.jpg'))
  const corners = quad(ref.referenceCorners)
  const quadW = Math.hypot(corners[1].x - corners[0].x, corners[1].y - corners[0].y)
  const k = Math.max(1, quadW / ow / 1.5)
  const rw = Math.round(refFull.width / k)
  const rh = Math.round(refFull.height / k)
  const refSmall = resizeAreaGray(refFull, rw, rh)
  const kx = refFull.width / rw
  const ky = refFull.height / rh
  const cs = corners.map((c) => ({ x: (c.x + 0.5) / kx - 0.5, y: (c.y + 0.5) / ky - 0.5 }))
  const H = rectToQuadHomography(cs, ow, oh)
  if (H === null) throw new Error(`${id}: degenerate referenceCorners`)
  const render = (dx: number, dy: number): Float32Array => warpGray(refSmall, shifted(H, dx, dy), ow, oh).data.slice()
  const z = zncc(render(0, 0), motifSmall.data)
  const sh = bestShift(render, motifSmall.data)
  const cellOut = ow / ref.cols
  const zQuarter = zncc(render(cellOut / 4, 0), motifSmall.data)
  const zRot = (() => {
    // Corners cycled by one (a 90° misorder) rectified to the transposed size, then compared to the
    // motif rotated back — i.e. what a TL/TR/BR/BL mix-up would give.
    const Hr = rectToQuadHomography([cs[1], cs[2], cs[3], cs[0]], oh, ow)
    if (Hr === null) return NaN
    const r = warpGray(refSmall, Hr, oh, ow)
    // Rotate r (oh×ow) counter-clockwise by 90° into ow×oh for comparison.
    const back = new Float32Array(ow * oh)
    for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) back[y * ow + x] = r.data[x * oh + (oh - 1 - y)]
    return zncc(back, motifSmall.data)
  })()
  const shiftMotifPx = Math.hypot(sh.dx, sh.dy) / s
  console.log(
    `(a) ${id}: reference ZNCC ${fmt(z)} | best shift (${fmt(sh.dx, 2)}, ${fmt(sh.dy, 2)}) out px = ${fmt(shiftMotifPx, 2)} motif px -> ZNCC ${fmt(sh.score)} | ` +
      `quarter-cell shift ${fmt(zQuarter)} | corners cycled ${fmt(zRot)}`,
  )
  expect(z >= REF_ZNCC_MIN, `(a) ${id}: reference ZNCC ${fmt(z)} < ${REF_ZNCC_MIN}`)
  expect(Math.hypot(sh.dx, sh.dy) <= MAX_MEAN_SHIFT_PX * 2, `(a) ${id}: rectification is off by (${fmt(sh.dx, 2)}, ${fmt(sh.dy, 2)}) output px`)
  expect(z > zQuarter && z > zRot, `(a) ${id}: a wrong alignment scores as well as the true one`)

  // (b, motif side) pieces.json: cell = row·cols + col, and corners on the cut lattice of (col, row).
  const cw = mw / ref.cols
  const ch = mh / ref.rows
  let maxLattice = 0
  for (const p of piecesArr) {
    expect(p.cell === p.row * ref.cols + p.col && p.id === p.cell, `(b) ${id} piece ${p.id}: cell ${p.cell} != row*cols+col`)
    const lattice: XY[] = [
      [p.col, p.row],
      [p.col + 1, p.row],
      [p.col + 1, p.row + 1],
      [p.col, p.row + 1],
    ]
    lattice.forEach(([c, r], i) => {
      const d = Math.hypot(p.corners[i][0] - (-0.5 + c * cw), p.corners[i][1] - (-0.5 + r * ch))
      maxLattice = Math.max(maxLattice, d / (ref.cut === 'grid' ? 1 : Math.min(cw, ch)))
    })
    expect(isClockwise(p.outline.map(pt)), `(b) ${id} piece ${p.id}: pieces.json outline is not clockwise`)
  }
  const latticeOk = ref.cut === 'grid' ? maxLattice <= LATTICE_TOL_PX : maxLattice <= IRREGULAR_JITTER * Math.SQRT2 + 1e-6
  console.log(
    `(b) ${id}: ${piecesArr.length} pieces, ${ref.cols}x${ref.rows} ${ref.cut} cut, max corner-to-lattice ${fmt(maxLattice, 4)} ${ref.cut === 'grid' ? 'px' : 'cells'}`,
  )
  expect(latticeOk, `(b) ${id}: pieces.json corners are off the (col,row) lattice by ${fmt(maxLattice, 4)}`)

  // Search patches: blur so a SEARCH_SIZE patch does not alias, then warp every cell.
  const cellPx = Math.sqrt(cw * ch)
  const blurred = gaussianBlur(motif, Math.max(0.5, (0.5 * cellPx) / SEARCH_SIZE))
  const searchPatches = new Map<number, Float32Array>()
  const pieces = new Map<number, PieceJson>()
  for (const p of piecesArr) {
    pieces.set(p.id, p)
    const Hc = rectToQuadHomography(quad(p.corners), SEARCH_SIZE, SEARCH_SIZE)
    if (Hc !== null) searchPatches.set(p.cell, warpGray(blurred, Hc, SEARCH_SIZE, SEARCH_SIZE).data)
  }
  return { id, ref, pieces, motif, searchPatches, cellPx }
}

interface Stats {
  zTrue: number[]
  rotWins: number
  rotTotal: number
  exhaustiveTop1: number
  exhaustiveTotal: number
  wrongCells: string[]
  shifts: { dx: number; dy: number }[]
  upErr: number[]
  upErrMidpoint: number[]
  upErrFaceDown: number[]
  iou: number[]
  recall: number[]
  wallBand: number[]
  wallFails: string[]
  wallTowardCentre: number
  wallCounted: number
  rasterTraceP99: number[]
  rasterTraceMean: number[]
  maskTraceMedian: number[]
  contourCw: number
  contourTotal: number
  cornerWindingOk: number
  cornerWindingTotal: number
}

function newStats(): Stats {
  return {
    zTrue: [], rotWins: 0, rotTotal: 0, exhaustiveTop1: 0, exhaustiveTotal: 0, wrongCells: [], shifts: [],
    upErr: [], upErrMidpoint: [], upErrFaceDown: [], iou: [], recall: [], wallBand: [], wallFails: [],
    wallTowardCentre: 0, wallCounted: 0, rasterTraceP99: [], rasterTraceMean: [], maskTraceMedian: [],
    contourCw: 0, contourTotal: 0, cornerWindingOk: 0, cornerWindingTotal: 0,
  }
}

/** (b)–(e) for one scene. */
async function checkScene(root: string, sceneId: string, puzzles: Map<string, Puzzle>, st: Stats): Promise<void> {
  const scene = await readJson<SceneJson>(join(root, 'scenes', `${sceneId}.json`))
  const puzzle = puzzles.get(scene.puzzleId)
  if (puzzle === undefined) throw new Error(`${sceneId}: unknown puzzle ${scene.puzzleId}`)
  const frame = await loadGray(join(root, 'scenes', `${sceneId}.jpg`))
  const labels = await loadLabels(join(root, 'scenes', `${sceneId}_mask.png`))
  expect(frame.width === scene.width && frame.height === scene.height, `${sceneId}: jpg size != json size`)
  expect(labels.width === scene.width && labels.height === scene.height, `${sceneId}: mask size != json size`)

  for (const sp of scene.pieces) {
    const tag = `${sceneId}#${sp.index} (piece ${sp.pieceId} c${sp.col} r${sp.row})`
    const mp = puzzle.pieces.get(sp.pieceId)
    if (mp === undefined) {
      failures.push(`${tag}: pieceId not in pieces.json`)
      continue
    }
    expect(mp.col === sp.col && mp.row === sp.row && mp.cell === sp.cell, `${tag}: col/row/cell differ from pieces.json`)
    const fc = quad(sp.corners)
    const outline = sp.outline.map(pt)

    // Corner winding: motif-order corners are clockwise in the image iff the piece lies face up.
    st.cornerWindingTotal++
    if (signedArea(fc) > 0 === sp.faceUp) st.cornerWindingOk++

    // (c) upAngleDeg vs the motif-up vector through the core centre, mapped by the corner homography.
    const motifToFrame = homographyFromQuad(quad(mp.corners), fc)
    if (motifToFrame !== null) {
      const mc = quad(mp.corners)
      const cx = (mc[0].x + mc[1].x + mc[2].x + mc[3].x) / 4
      const cy = (mc[0].y + mc[1].y + mc[2].y + mc[3].y) / 4
      const eps = 1e-3 * puzzle.cellPx
      const a = applyH(motifToFrame, cx, cy)
      const b = applyH(motifToFrame, cx, cy - eps)
      const e = angleDiff(upAngle(b.x - a.x, b.y - a.y), sp.upAngleDeg)
      const mid = angleDiff(
        upAngle((fc[0].x + fc[1].x - fc[2].x - fc[3].x) / 2, (fc[0].y + fc[1].y - fc[2].y - fc[3].y) / 2),
        sp.upAngleDeg,
      )
      if (sp.faceUp) {
        st.upErr.push(e)
        st.upErrMidpoint.push(mid)
      } else st.upErrFaceDown.push(e)
    }

    // Work in a crop around the piece.
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity
    for (const p of outline) {
      x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y)
    }
    const pad = 12
    const crop = cropGray(frame, x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    const { ox, oy } = crop
    const cw = crop.img.width
    const chh = crop.img.height
    const inst: Mask = { width: cw, height: chh, data: new Uint8Array(cw * chh) }
    for (let y = 0; y < chh; y++) for (let x = 0; x < cw; x++) inst.data[y * cw + x] = labels.data[(oy + y) * labels.width + ox + x] === sp.index ? 1 : 0
    const local = outline.map((p) => ({ x: p.x - ox, y: p.y - oy }))
    const raster = rasterizePolygon(local, cw, chh)

    // (d) mask vs rasterised outline, for fully visible pieces whose outline lies inside the frame.
    const fullyVisible = sp.visibleFraction >= 1 && x0 >= 0 && y0 >= 0 && x1 <= scene.width - 1 && y1 <= scene.height - 1
    if (fullyVisible) {
      let inter = 0, uni = 0, rasterN = 0
      for (let i = 0; i < inst.data.length; i++) {
        const m = inst.data[i], r = raster.data[i]
        if (m && r) inter++
        if (m || r) uni++
        if (r) rasterN++
      }
      st.iou.push(inter / uni)
      st.recall.push(inter / rasterN)
      // Mask pixels outside the outline: how far out (the wall band), and on which side.
      const outside: Mask = { width: cw, height: chh, data: raster.data.map((v) => (v ? 0 : 1)) }
      const dist = distanceTransform(outside).data
      let far = 0, ex = 0, ey = 0, en = 0
      for (let y = 0; y < chh; y++) {
        for (let x = 0; x < cw; x++) {
          const i = y * cw + x
          if (!inst.data[i] || raster.data[i]) continue
          far = Math.max(far, dist[i])
          ex += x; ey += y; en++
        }
      }
      const core = coreSideOf(fc)
      const band = WALL_BAND_CORE * core + WALL_BAND_PX
      st.wallBand.push(far / core)
      if (far > band) st.wallFails.push(`${tag}: mask reaches ${fmt(far, 1)} px beyond the outline (band ${fmt(band, 1)} px)`)
      // Parallax puts the visible wall on the side of the piece facing the camera's nadir, which for these
      // gently tilted frames lies near the image centre. Only counted when the wall is substantial.
      if (en > 0.02 * rasterN) {
        let rx = 0, ry = 0, rn = 0
        for (let y = 0; y < chh; y++) for (let x = 0; x < cw; x++) if (raster.data[y * cw + x]) { rx += x; ry += y; rn++ }
        const wx = ex / en - rx / rn, wy = ey / en - ry / rn
        const tx = scene.width / 2 - (ox + rx / rn), ty = scene.height / 2 - (oy + ry / rn)
        st.wallCounted++
        if (wx * tx + wy * ty > 0) st.wallTowardCentre++
      }

      // (e) tracer on the engine's own raster of the outline: pixel-centre contour vs the true edge.
      const rc = findContours(raster)
      if (rc.length > 0) {
        const c = rc.reduce((a, b) => (b.length > a.length ? b : a))
        const d = c.map((q) => distToPolyline(q, local)).sort((a, b) => a - b)
        st.rasterTraceP99.push(d[Math.floor(0.99 * (d.length - 1))])
        st.rasterTraceMean.push(mean(d))
      }
      // (e) tracer on the instance mask: clockwise, and on the outline wherever no wall shows.
      const mc = findContours(inst)
      if (mc.length > 0) {
        const c = mc.reduce((a, b) => (b.length > a.length ? b : a))
        st.contourTotal++
        if (isClockwise(c)) st.contourCw++
        st.maskTraceMedian.push(median(local.map((q) => distToPolyline(q, c))))
      }
    }

    // (b) photometric: only face-up pieces that are mostly visible.
    if (!sp.faceUp || sp.visibleFraction < 0.6) continue
    const lc = fc.map((p) => ({ x: p.x - ox, y: p.y - oy }))
    const coreSide = coreSideOf(lc)
    const S = Math.max(32, Math.min(192, Math.round(coreSide)))
    // Valid: top face in the mask, away from the cut rim.
    const top: Mask = { width: cw, height: chh, data: new Uint8Array(cw * chh) }
    for (let i = 0; i < top.data.length; i++) top.data[i] = inst.data[i] & raster.data[i]
    const frameBlur = gaussianBlur(crop.img, 1.0)
    const erodedTop = erodeBox(top, RIM_ERODE_PX)

    // Motif patch at S, from a blurred crop of the motif around the cell.
    const mc = quad(mp.corners)
    let mx0 = Infinity, my0 = Infinity, mx1 = -Infinity, my1 = -Infinity
    for (const p of mc) {
      mx0 = Math.min(mx0, p.x); my0 = Math.min(my0, p.y); mx1 = Math.max(mx1, p.x); my1 = Math.max(my1, p.y)
    }
    const mcrop = cropGray(puzzle.motif, mx0 - 8, my0 - 8, mx1 + 8, my1 + 8)
    const scale = puzzle.cellPx / S
    const motifBlur = gaussianBlur(mcrop.img, Math.max(1.0, 0.5 * scale))
    const Hm = rectToQuadHomography(mc.map((p) => ({ x: p.x - mcrop.ox, y: p.y - mcrop.oy })), S, S)
    const Hf = rectToQuadHomography(lc, S, S)
    if (Hm === null || Hf === null) continue
    const motifPatch = warpGray(motifBlur, Hm, S, S).data.slice()
    const valid = warpMask(erodedTop, Hf, S, S).data.slice()
    const renderF = (dx: number, dy: number): Float32Array => warpGray(frameBlur, shifted(Hf, dx, dy), S, S).data.slice()
    const zt = zncc(renderF(0, 0), motifPatch, valid)
    const textured = stdDev(motifPatch, valid) >= TEXTURE_STD_MIN
    if (!textured || Number.isNaN(zt)) continue
    st.zTrue.push(zt)

    // Rotation: corners cycled r steps.
    let rotBest = 0
    let rotBestScore = zt
    for (let r = 1; r < 4; r++) {
      const Hr = rectToQuadHomography([lc[r % 4], lc[(r + 1) % 4], lc[(r + 2) % 4], lc[(r + 3) % 4]], S, S)
      if (Hr === null) continue
      const vr = warpMask(erodedTop, Hr, S, S).data.slice()
      const zr = zncc(warpGray(frameBlur, Hr, S, S).data, motifPatch, vr)
      if (zr > rotBestScore) {
        rotBestScore = zr
        rotBest = r
      }
    }
    st.rotTotal++
    if (rotBest === 0) st.rotWins++

    // Sub-pixel alignment of frame vs motif (in output px ≈ frame px).
    const sh = bestShift(renderF, motifPatch, valid)
    st.shifts.push({ dx: sh.dx, dy: sh.dy })

    // Exhaustive search over all cells × 4 rotations at SEARCH_SIZE.
    const frameSearch = gaussianBlur(crop.img, Math.max(0.5, (0.5 * coreSide) / SEARCH_SIZE))
    let best = { cell: -1, rot: -1, score: -Infinity }
    for (let r = 0; r < 4; r++) {
      const Hr = rectToQuadHomography([lc[r % 4], lc[(r + 1) % 4], lc[(r + 2) % 4], lc[(r + 3) % 4]], SEARCH_SIZE, SEARCH_SIZE)
      if (Hr === null) continue
      const fp = warpGray(frameSearch, Hr, SEARCH_SIZE, SEARCH_SIZE).data.slice()
      const vr = warpMask(erodedTop, Hr, SEARCH_SIZE, SEARCH_SIZE).data.slice()
      for (const [cell, patch] of puzzle.searchPatches) {
        const z = zncc(fp, patch, vr)
        if (z > best.score) best = { cell, rot: r, score: z }
      }
    }
    st.exhaustiveTotal++
    if (best.cell === sp.cell && best.rot === 0) st.exhaustiveTop1++
    else st.wrongCells.push(`${tag} -> cell ${best.cell} (c${best.cell % puzzle.ref.cols} r${Math.floor(best.cell / puzzle.ref.cols)}) rot ${best.rot}, true ZNCC ${fmt(zt)}`)
  }
}

/** Mean length of the 4 core sides. */
function coreSideOf(c: readonly Point[]): number {
  let s = 0
  for (let i = 0; i < 4; i++) s += Math.hypot(c[(i + 1) % 4].x - c[i].x, c[(i + 1) % 4].y - c[i].y)
  return s / 4
}

function applyH(H: Homography, x: number, y: number): Point {
  const w = H[6] * x + H[7] * y + H[8]
  return { x: (H[0] * x + H[1] * y + H[2]) / w, y: (H[3] * x + H[4] * y + H[5]) / w }
}

/** Binary erosion with a (2r+1)² box, zero outside the image. Only used to drop the cut rim. */
function erodeBox(m: Mask, r: number): Mask {
  const { width: w, height: h } = m
  const out: Mask = { width: w, height: h, data: new Uint8Array(w * h) }
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      let all = 1
      for (let dy = -r; dy <= r && all; dy++) for (let dx = -r; dx <= r; dx++) if (!m.data[(y + dy) * w + x + dx]) { all = 0; break }
      out.data[y * w + x] = all
    }
  }
  return out
}

async function main(): Promise<void> {
  const root = process.argv[2]
  if (!root) {
    console.error('usage: node eval/check-dataset.ts <datasetDir>')
    process.exit(2)
  }
  const t0 = performance.now()
  const meta = await readJson<Meta>(join(root, 'meta.json'))
  const puzzles = new Map<string, Puzzle>()
  for (const id of meta.puzzles) puzzles.set(id, await checkPuzzle(root, id))
  const st = newStats()
  for (const s of meta.scenes) await checkScene(root, s, puzzles, st)

  const meanShift = { dx: mean(st.shifts.map((s) => s.dx)), dy: mean(st.shifts.map((s) => s.dy)) }
  const maxAbs = (v: number[]): number => (v.length ? Math.max(...v) : NaN)
  console.log(
    `(b) scene pieces (face-up, ≥60 % visible, textured): n=${st.zTrue.length} | ZNCC vs own cell median ${fmt(median(st.zTrue))}, ` +
      `min ${fmt(Math.min(...st.zTrue))} | true rotation wins ${st.rotWins}/${st.rotTotal} | ` +
      `exhaustive search (all cells × 4 rotations) top-1 ${st.exhaustiveTop1}/${st.exhaustiveTotal}`,
  )
  console.log(
    `(b) alignment: mean best shift frame→motif (${fmt(meanShift.dx, 3)}, ${fmt(meanShift.dy, 3)}) px, ` +
      `median |shift| ${fmt(median(st.shifts.map((s) => Math.hypot(s.dx, s.dy))), 3)} px`,
  )
  for (const w of st.wrongCells) console.log(`    exhaustive miss: ${w}`)
  console.log(`(b) corner winding matches faceUp: ${st.cornerWindingOk}/${st.cornerWindingTotal}`)
  console.log(
    `(c) upAngleDeg vs motif-up vector: face-up max err ${fmt(maxAbs(st.upErr), 3)}° (n=${st.upErr.length}), ` +
      `vs edge-midpoint rule ${fmt(maxAbs(st.upErrMidpoint), 3)}°; face-down max err ${fmt(maxAbs(st.upErrFaceDown), 3)}° (n=${st.upErrFaceDown.length})`,
  )
  const iouBelow = st.iou.filter((v) => v < MASK_IOU_REPORT).length
  console.log(
    `(d) fully visible pieces n=${st.iou.length}: top-face recall median ${fmt(median(st.recall))}, min ${fmt(Math.min(...st.recall))} | ` +
      `mask beyond outline max ${fmt(maxAbs(st.wallBand), 3)} x core (band ${WALL_BAND_CORE} x core + ${WALL_BAND_PX} px) | ` +
      `wall on the side facing the image centre ${st.wallTowardCentre}/${st.wallCounted}`,
  )
  console.log(
    `    raw IoU mask vs outline (mask includes the side wall): median ${fmt(median(st.iou))}, min ${fmt(Math.min(...st.iou))}, ` +
      `${iouBelow}/${st.iou.length} below ${MASK_IOU_REPORT}`,
  )
  for (const f of st.wallFails) console.log(`    ${f}`)
  console.log(
    `(e) tracer on engine raster of outline: mean dist ${fmt(median(st.rasterTraceMean), 3)} px (median over pieces), ` +
      `p99 max ${fmt(maxAbs(st.rasterTraceP99), 3)} px | tracer on instance mask: clockwise ${st.contourCw}/${st.contourTotal}, ` +
      `outline→contour median dist ${fmt(median(st.maskTraceMedian), 3)} px (worst piece ${fmt(maxAbs(st.maskTraceMedian), 3)})`,
  )

  expect(median(st.zTrue) >= PIECE_ZNCC_MEDIAN_MIN, `(b) median piece ZNCC ${fmt(median(st.zTrue))} < ${PIECE_ZNCC_MEDIAN_MIN}`)
  expect(st.rotWins >= ROTATION_WIN_MIN * st.rotTotal, `(b) true rotation wins only ${st.rotWins}/${st.rotTotal}`)
  expect(st.exhaustiveTop1 >= EXHAUSTIVE_TOP1_MIN * st.exhaustiveTotal, `(b) exhaustive top-1 only ${st.exhaustiveTop1}/${st.exhaustiveTotal}`)
  expect(Math.hypot(meanShift.dx, meanShift.dy) <= MAX_MEAN_SHIFT_PX, `(b) systematic frame/motif offset (${fmt(meanShift.dx)}, ${fmt(meanShift.dy)}) px`)
  expect(st.cornerWindingOk === st.cornerWindingTotal, `(b) corner winding disagrees with faceUp`)
  expect(maxAbs(st.upErr) <= UP_ANGLE_TOL_DEG, `(c) upAngleDeg off by up to ${fmt(maxAbs(st.upErr))}°`)
  expect(Math.min(...st.recall) >= TOP_FACE_RECALL_MIN, `(d) top-face recall down to ${fmt(Math.min(...st.recall))}`)
  expect(st.wallFails.length === 0, `(d) ${st.wallFails.length} masks reach beyond the side-wall band`)
  expect(st.wallTowardCentre >= 0.8 * st.wallCounted, `(d) side wall faces the image centre for only ${st.wallTowardCentre}/${st.wallCounted}`)
  expect(maxAbs(st.rasterTraceP99) <= RASTER_TRACE_P99_MAX, `(e) contour of the engine raster is up to ${fmt(maxAbs(st.rasterTraceP99))} px off the outline`)
  expect(st.contourCw === st.contourTotal, `(e) ${st.contourTotal - st.contourCw} traced contours are not clockwise`)
  expect(median(st.maskTraceMedian) <= MASK_TRACE_MEDIAN_MAX, `(e) outline→mask contour median distance ${fmt(median(st.maskTraceMedian))} px`)

  console.log(`checked ${meta.puzzles.length} puzzles, ${meta.scenes.length} scenes in ${((performance.now() - t0) / 1000).toFixed(1)} s`)
  if (failures.length > 0) {
    console.log(`FAILED (${failures.length}):`)
    for (const f of failures) console.log(`  - ${f}`)
    process.exit(1)
  }
  console.log('all checks passed')
}

await main()
