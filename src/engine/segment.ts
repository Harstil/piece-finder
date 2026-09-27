/**
 * Piece segmentation for live camera frames (classical; the learned any-table model replaces the
 * background test later — the rest of this pipeline stays).
 *
 * 1. Background model. The frame is shrunk to ~MODEL_LONG_SIDE px and its Lab colours clustered with
 *    k-means (K clusters, warm-started from the previous frame so the model is stable while hovering).
 *    Clusters that make up a real share of the frame's border band are "table": a plain mat gives one,
 *    wood grain or a plaid cloth two or three. Pieces lie inside the frame and are printed in many
 *    colours, so they rarely own a border cluster.
 * 2. Coarse pieces. A pixel is foreground when it is far (in per-cluster standard deviations, with
 *    lightness down-weighted so soft shadows stay "table") from every table cluster. After a close/open
 *    and hole filling, connected components of plausible piece size are the candidates.
 * 3. Precise outline. Each candidate is re-segmented in a crop of the full-resolution frame with the
 *    same table model, and its outer contour is traced there, so corners are found at full precision.
 * 4. Validation by shape. analyzeShape must find a jigsaw core (4 corners, tabs/blanks); blobs that are
 *    not pieces (fingers, clutter, two touching pieces) fail it and are dropped.
 */

import { findContours } from './geom/index.ts'
import {
  clipRect,
  close,
  cropRGBA,
  ellipseKernel,
  fillHoles,
  fitSize,
  labelComponents,
  open,
  resizeAreaRGBA,
  rgbaToLab,
} from './image/index.ts'
import { analyzeShape } from './shape.ts'
import type { LabImage, Mask, PieceInstance, PieceShape, Point, RGBAImage } from './types.ts'

/** Long side of the frame used for the table model and coarse detection. Guessed: ~12 px per small piece core. */
const MODEL_LONG_SIDE = 400
/** Number of colour clusters. Guessed: table (1–3) + pieces' dominant colours. */
const K = 6
const KMEANS_ITERATIONS = 8
/** Width of the border band, as a fraction of the short side. */
const BORDER_BAND = 0.06
/** A cluster holding at least this share of the border band is table. Guessed. */
const TABLE_BORDER_SHARE = 0.1
/** Lightness weight in colour distances: shadows change L far more than a, b. Guessed. */
const L_WEIGHT = 0.5
/** Darker-than-table lightness differences count this much (shadows). Guessed, checked in eval/segment-eval.ts. */
const SHADOW_L_FACTOR = 0.25
/** Shadows dim a table colour at most to this share of its lightness. Guessed. */
const SHADOW_MIN_DIM = 0.35
/** Foreground when farther than this many cluster std devs from every table cluster. Guessed. */
const FOREGROUND_SIGMAS = 3
/** Minimum per-cluster std (Lab units) so a perfectly uniform mat does not make everything foreground. */
const MIN_CLUSTER_STD = 3
/** Piece area bounds as a fraction of the frame area. Guessed: 40 px cores at 1280 px up to a fifth of the frame. */
const MIN_AREA_FRACTION = 0.0008
const MAX_AREA_FRACTION = 0.2
/** Crop margin around a coarse blob for the full-resolution pass, fraction of its larger side. */
const REFINE_MARGIN = 0.25
/** Shape confidence below which a blob is not reported. Guessed; see eval/segment-eval.ts. */
const MIN_SHAPE_CONFIDENCE = 0.02
/** Pieces touching the frame edge are cut off; ignore blobs within this many coarse px of it. */
const EDGE_GUARD = 1

export interface SegmentOptions {
  /** Use the cluster centres of the previous frame as the k-means start (stability while hovering). */
  warmStart?: TableModel | null
  /** Shape confidence below which a blob is dropped. Default MIN_SHAPE_CONFIDENCE. */
  minConfidence?: number
}

export interface TableModel {
  /** K cluster centres, [L, a, b] each (weighted space: L already multiplied by L_WEIGHT). */
  centres: Float64Array
  stds: Float64Array
  table: boolean[]
}

export interface SegmentedPiece extends PieceInstance {
  shape: PieceShape
}

export interface SegmentResult {
  pieces: SegmentedPiece[]
  model: TableModel
  /** Coarse foreground mask at model resolution (debug overlay). */
  coarse: Mask
}

export function segmentPieces(frame: RGBAImage, opts: SegmentOptions = {}): SegmentResult {
  const size = fitSize(frame.width, frame.height, MODEL_LONG_SIDE)
  const small = size.width < frame.width ? resizeAreaRGBA(frame, size.width, size.height) : frame
  const lab = rgbaToLab(small)
  const model = fitTable(lab, opts.warmStart ?? null)
  const fg = foreground(lab, model)
  const cleaned = fillHoles(open(close(fg, ellipseKernel(3)), ellipseKernel(3)))

  const scale = frame.width / small.width
  const comps = labelComponents(cleaned)
  const area = small.width * small.height
  const pieces: SegmentedPiece[] = []
  for (const stat of comps.stats) {
    if (stat.area < MIN_AREA_FRACTION * area || stat.area > MAX_AREA_FRACTION * area) continue
    const b = stat.bbox
    if (b.x <= EDGE_GUARD || b.y <= EDGE_GUARD || b.x + b.width >= small.width - EDGE_GUARD || b.y + b.height >= small.height - EDGE_GUARD) {
      continue
    }
    const refined = refine(frame, model, opts.minConfidence ?? MIN_SHAPE_CONFIDENCE, (b.x - 0.5) * scale, (b.y - 0.5) * scale, (b.x + b.width - 0.5) * scale, (b.y + b.height - 0.5) * scale)
    if (refined !== null) pieces.push(refined)
  }
  return { pieces, model, coarse: cleaned }
}

/** k-means on (weighted) Lab, then mark the clusters that dominate the border band as table. */
function fitTable(lab: LabImage, warm: TableModel | null): TableModel {
  const { width: w, height: h } = lab
  const n = w * h
  const centres = new Float64Array(K * 3)
  if (warm !== null) centres.set(warm.centres)
  else {
    // Deterministic start: K pixels spread along the diagonal and the border.
    for (let k = 0; k < K; k++) {
      const t = (k + 0.5) / K
      const i = Math.floor(t * (h - 1)) * w + Math.floor(t * (w - 1))
      centres[k * 3] = lab.L[i] * L_WEIGHT
      centres[k * 3 + 1] = lab.a[i]
      centres[k * 3 + 2] = lab.b[i]
    }
  }
  const assign = new Uint8Array(n)
  const sums = new Float64Array(K * 4)
  for (let iter = 0; iter < KMEANS_ITERATIONS; iter++) {
    sums.fill(0)
    for (let i = 0; i < n; i++) {
      const l = lab.L[i] * L_WEIGHT
      const a = lab.a[i]
      const b = lab.b[i]
      let best = 0
      let bestD = Infinity
      for (let k = 0; k < K; k++) {
        const dl = l - centres[k * 3]
        const da = a - centres[k * 3 + 1]
        const db = b - centres[k * 3 + 2]
        const d = dl * dl + da * da + db * db
        if (d < bestD) {
          bestD = d
          best = k
        }
      }
      assign[i] = best
      sums[best * 4] += l
      sums[best * 4 + 1] += a
      sums[best * 4 + 2] += b
      sums[best * 4 + 3]++
    }
    for (let k = 0; k < K; k++) {
      const c = sums[k * 4 + 3]
      if (c > 0) {
        centres[k * 3] = sums[k * 4] / c
        centres[k * 3 + 1] = sums[k * 4 + 1] / c
        centres[k * 3 + 2] = sums[k * 4 + 2] / c
      } else {
        // Re-seed an empty cluster on a pixel far from its centre (deterministic: the first such).
        const i = (k * 7919) % n
        centres[k * 3] = lab.L[i] * L_WEIGHT
        centres[k * 3 + 1] = lab.a[i]
        centres[k * 3 + 2] = lab.b[i]
      }
    }
  }
  // Per-cluster spread and border share.
  const sq = new Float64Array(K)
  const count = new Float64Array(K)
  const border = new Float64Array(K)
  let borderTotal = 0
  const band = Math.max(1, Math.round(BORDER_BAND * Math.min(w, h)))
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      const k = assign[i]
      const dl = lab.L[i] * L_WEIGHT - centres[k * 3]
      const da = lab.a[i] - centres[k * 3 + 1]
      const db = lab.b[i] - centres[k * 3 + 2]
      sq[k] += dl * dl + da * da + db * db
      count[k]++
      if (x < band || y < band || x >= w - band || y >= h - band) {
        border[k]++
        borderTotal++
      }
    }
  }
  const stds = new Float64Array(K)
  const table: boolean[] = []
  for (let k = 0; k < K; k++) {
    stds[k] = Math.max(MIN_CLUSTER_STD, count[k] > 0 ? Math.sqrt(sq[k] / (3 * count[k])) : MIN_CLUSTER_STD)
    table.push(border[k] >= TABLE_BORDER_SHARE * borderTotal)
  }
  if (!table.some(Boolean)) {
    // Nothing dominates the border (pieces everywhere): take the single largest border cluster.
    let best = 0
    for (let k = 1; k < K; k++) if (border[k] > border[best]) best = k
    table[best] = true
  }
  return { centres, stds, table }
}

/**
 * 1 where a pixel is far from every table cluster. Shadow-tolerant: a pixel darker than a table cluster
 * is compared with that cluster dimmed to the pixel's lightness (a shadow scales a and b down with L),
 * and its remaining lightness difference counts only SHADOW_L_FACTOR as much. Without this, the soft
 * shadow around every piece joins the piece, fills its blank mouths and rounds its corners.
 */
function foreground(lab: LabImage, model: TableModel): Mask {
  const n = lab.width * lab.height
  const out: Mask = { width: lab.width, height: lab.height, data: new Uint8Array(n) }
  const { centres, stds, table } = model
  for (let i = 0; i < n; i++) {
    const l = lab.L[i] * L_WEIGHT
    const a = lab.a[i]
    const b = lab.b[i]
    let near = false
    for (let k = 0; k < K && !near; k++) {
      if (!table[k]) continue
      const cl = centres[k * 3]
      let dl = l - cl
      let ca = centres[k * 3 + 1]
      let cb = centres[k * 3 + 2]
      if (dl < 0 && cl > 0) {
        const dim = Math.max(SHADOW_MIN_DIM, l / cl)
        ca *= dim
        cb *= dim
        dl *= SHADOW_L_FACTOR
      }
      const da = a - ca
      const db = b - cb
      const s = FOREGROUND_SIGMAS * stds[k]
      if (dl * dl + da * da + db * db <= s * s) near = true
    }
    out.data[i] = near ? 0 : 1
  }
  return out
}

/** Full-resolution outline of the blob inside [x0,x1]×[y0,y1] (frame px), validated as a piece. */
function refine(frame: RGBAImage, model: TableModel, minConfidence: number, x0: number, y0: number, x1: number, y1: number): SegmentedPiece | null {
  const m = REFINE_MARGIN * Math.max(x1 - x0, y1 - y0)
  const rect = clipRect(x0 - m, y0 - m, x1 + m, y1 + m, frame.width, frame.height)
  if (rect.width < 8 || rect.height < 8) return null
  const crop = cropRGBA(frame, rect)
  const lab = rgbaToLab(crop)
  const fg = foreground(lab, model)
  const r = Math.max(3, Math.round(Math.max(rect.width, rect.height) / 80) * 2 + 1)
  const cleaned = fillHoles(open(close(fg, ellipseKernel(r)), ellipseKernel(r)))
  const contours = findContours(cleaned)
  if (contours.length === 0) return null
  let best = contours[0]
  for (const c of contours) if (c.length > best.length) best = c
  const contour: Point[] = best.map((p) => ({ x: p.x + rect.x, y: p.y + rect.y }))
  const shape = analyzeShape(contour, { frameCenter: { x: (frame.width - 1) / 2, y: (frame.height - 1) / 2 } })
  if (shape === null || shape.confidence < minConfidence) return null
  let bx0 = Infinity
  let by0 = Infinity
  let bx1 = -Infinity
  let by1 = -Infinity
  let area2 = 0
  for (let i = 0; i < contour.length; i++) {
    const p = contour[i]
    const q = contour[(i + 1) % contour.length]
    area2 += p.x * q.y - q.x * p.y
    bx0 = Math.min(bx0, p.x)
    by0 = Math.min(by0, p.y)
    bx1 = Math.max(bx1, p.x)
    by1 = Math.max(by1, p.y)
  }
  return {
    contour,
    bbox: { x: bx0, y: by0, width: bx1 - bx0, height: by1 - by0 },
    area: Math.abs(area2) / 2,
    score: shape.confidence,
    shape,
  }
}
