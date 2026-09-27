/**
 * Per-piece tracking across camera frames, and evidence accumulation while the user hovers.
 *
 * A single frame's match can be unlucky (motion blur, glare, a corner misread). While the user holds the
 * camera over a pile, the same piece is seen in many frames, so each tracked piece averages its per-frame
 * cell probabilities; the answer "locks" once the leader has been stable for a few matches. Tracks are
 * associated frame to frame by bounding-box IoU (the pattern of desk-detect-poc's IoU tracker), and a
 * track that is not seen for a while is dropped.
 *
 * Evidence is kept per cell, not per (cell, rotation): rotation is relative to the piece's corner order,
 * which can change between frames, so the rotate-arrow always comes from the latest frame's match.
 */

import type { Candidate, PieceShape, PieceType, Point, Quad, Rect, TrackState } from './types.ts'

/** Minimum bbox IoU to continue a track. Guessed: hand shake moves a piece ≤ ~1/3 of its size per frame. */
const MIN_IOU = 0.3
/** Frames a track survives unseen (the piece may flicker in and out of segmentation). */
const MAX_UNSEEN_FRAMES = 6
/** Matches needed before an answer can lock, and the averaged probability and margin it needs. Guessed. */
const LOCK_MATCHES = 2
const LOCK_PROB = 0.55
const LOCK_MARGIN = 0.25
/** Cells kept per track (the rest of the evidence is dropped). */
const KEEP_CELLS = 24
/** Candidates reported per track. */
const REPORT = 5

export interface Detection {
  contour: Point[]
  bbox: Rect
  shape: PieceShape
}

export interface Track {
  id: number
  bbox: Rect
  contour: Point[]
  corners: Quad
  pieceType: PieceType
  /** Shape from the latest detection (what canonicalize needs). */
  shape: PieceShape
  lastSeen: number
  framesSeen: number
  matches: number
  /** Sum over matches of each cell's probability. */
  evidence: Map<number, number>
  /** Latest candidate per cell (col/row and the rotate-arrow for display). */
  latest: Map<number, Candidate>
  /** True when this frame's detection has not been matched yet. */
  pending: boolean
}

export interface TrackerState {
  tracks: Track[]
  nextId: number
}

export function createTracker(): TrackerState {
  return { tracks: [], nextId: 1 }
}

function iou(a: Rect, b: Rect): number {
  const x0 = Math.max(a.x, b.x)
  const y0 = Math.max(a.y, b.y)
  const x1 = Math.min(a.x + a.width, b.x + b.width)
  const y1 = Math.min(a.y + a.height, b.y + b.height)
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0)
  const union = a.width * a.height + b.width * b.height - inter
  return union > 0 ? inter / union : 0
}

/** Associates this frame's detections with tracks (greedy by IoU). Returns the tracks seen this frame. */
export function associate(state: TrackerState, detections: Detection[], frameId: number): Track[] {
  const pairs: { t: Track; d: number; iou: number }[] = []
  for (const t of state.tracks) {
    detections.forEach((d, i) => {
      const v = iou(t.bbox, d.bbox)
      if (v >= MIN_IOU) pairs.push({ t, d: i, iou: v })
    })
  }
  pairs.sort((a, b) => b.iou - a.iou)
  const usedTrack = new Set<Track>()
  const usedDet = new Set<number>()
  const seen: Track[] = []
  const update = (t: Track, d: Detection): void => {
    t.bbox = d.bbox
    t.contour = d.contour
    t.corners = d.shape.corners
    t.pieceType = d.shape.pieceType
    t.shape = d.shape
    t.lastSeen = frameId
    t.framesSeen++
    t.pending = true
    seen.push(t)
  }
  for (const p of pairs) {
    if (usedTrack.has(p.t) || usedDet.has(p.d)) continue
    usedTrack.add(p.t)
    usedDet.add(p.d)
    update(p.t, detections[p.d])
  }
  detections.forEach((d, i) => {
    if (usedDet.has(i)) return
    const t: Track = {
      id: state.nextId++,
      bbox: d.bbox,
      contour: d.contour,
      corners: d.shape.corners,
      pieceType: d.shape.pieceType,
      shape: d.shape,
      lastSeen: frameId,
      framesSeen: 0,
      matches: 0,
      evidence: new Map(),
      latest: new Map(),
      pending: true,
    }
    state.tracks.push(t)
    update(t, d)
  })
  state.tracks = state.tracks.filter((t) => frameId - t.lastSeen <= MAX_UNSEEN_FRAMES)
  return seen
}

/** Adds one frame's match to a track. */
export function addEvidence(t: Track, candidates: Candidate[]): void {
  t.matches++
  t.pending = false
  for (const c of candidates) {
    t.evidence.set(c.cell, (t.evidence.get(c.cell) ?? 0) + c.prob)
    t.latest.set(c.cell, c)
  }
  if (t.evidence.size > KEEP_CELLS) {
    const keep = [...t.evidence.entries()].sort((a, b) => b[1] - a[1]).slice(0, KEEP_CELLS)
    t.evidence = new Map(keep)
    for (const cell of [...t.latest.keys()]) if (!t.evidence.has(cell)) t.latest.delete(cell)
  }
}

/** Removes a cell from every track's evidence (the user marked it placed). */
export function forgetCell(state: TrackerState, cell: number): void {
  for (const t of state.tracks) {
    t.evidence.delete(cell)
    t.latest.delete(cell)
  }
}

/** The track as the UI sees it: averaged candidates, lock state and region probability. */
export function trackView(t: Track, region: ReadonlySet<number> | null): TrackState {
  const ranked = [...t.evidence.entries()].sort((a, b) => b[1] - a[1])
  const candidates: Candidate[] = []
  for (const [cell, sum] of ranked.slice(0, REPORT)) {
    const c = t.latest.get(cell)
    if (c !== undefined) candidates.push({ ...c, prob: t.matches > 0 ? sum / t.matches : 0 })
  }
  const p1 = candidates[0]?.prob ?? 0
  const p2 = candidates[1]?.prob ?? 0
  let regionProb: number | null = null
  if (region !== null) {
    regionProb = 0
    for (const [cell, sum] of t.evidence) if (region.has(cell)) regionProb += sum / Math.max(1, t.matches)
  }
  return {
    trackId: t.id,
    contour: t.contour,
    corners: t.corners,
    pieceType: t.pieceType,
    candidates,
    locked: t.matches >= LOCK_MATCHES && p1 >= LOCK_PROB && p1 - p2 >= LOCK_MARGIN,
    regionProb,
    framesSeen: t.framesSeen,
  }
}
