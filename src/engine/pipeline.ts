/**
 * The live pipeline: one camera frame in, tracked pieces with their best cells out.
 *
 *   segmentPieces  → outlines + shapes of the pieces in view        (every frame)
 *   associate      → which tracked piece each outline is             (every frame)
 *   canonicalize + matchPiece → cell probabilities                   (within a time budget per frame)
 *   addEvidence    → per-track averaged answer, locked when stable
 *
 * Matching every piece in every frame would take too long with many pieces in view, so each frame
 * matches as many as MATCH_BUDGET_MS allows, unlocked and least-matched pieces first; locked pieces are
 * re-checked only now and then. DOM-free like the rest of the engine: the worker feeds it pixels, and
 * eval code can run it on dataset frames.
 */

import { canonicalize } from './canonical.ts'
import { matchPiece } from './match/index.ts'
import type { ReferenceModel } from './reference.ts'
import { segmentPieces, type TableModel } from './segment.ts'
import { addEvidence, associate, createTracker, forgetCell, trackView, type Track, type TrackerState } from './tracker.ts'
import type { FrameResult, RGBAImage } from './types.ts'

/** Time per frame spent matching pieces (ms, after segmentation). Guessed for ~3–6 fps on a phone. */
const MATCH_BUDGET_MS = 120
/** A locked piece is re-matched only every this many frames it is seen. */
const LOCKED_RECHECK_FRAMES = 8

export interface Session {
  ref: ReferenceModel
  placed: Set<number>
  region: Set<number> | null
  tracker: TrackerState
  table: TableModel | null
}

export function createSession(ref: ReferenceModel, placed: Iterable<number> = []): Session {
  return { ref, placed: new Set(placed), region: null, tracker: createTracker(), table: null }
}

export function setPlaced(session: Session, cells: Iterable<number>): void {
  const next = new Set(cells)
  for (const cell of next) if (!session.placed.has(cell)) forgetCell(session.tracker, cell)
  session.placed = next
}

export function stepFrame(session: Session, frame: RGBAImage, frameId: number, budgetMs = MATCH_BUDGET_MS): FrameResult {
  const t0 = performance.now()
  const seg = segmentPieces(frame, { warmStart: session.table })
  session.table = seg.model
  const detections = seg.pieces.map((p) => ({ contour: p.contour, bbox: p.bbox, shape: p.shape }))
  const seen = associate(session.tracker, detections, frameId)
  const t1 = performance.now()

  const due = (t: Track): boolean => {
    const view = trackView(t, null)
    return !view.locked || t.framesSeen % LOCKED_RECHECK_FRAMES === 0
  }
  const queue = seen.filter(due).sort((a, b) => a.matches - b.matches)
  let matched = 0
  for (const t of queue) {
    if (matched > 0 && performance.now() - t1 > budgetMs) break
    const piece = canonicalize(frame, t.contour, t.shape)
    const result = matchPiece(session.ref, piece, { excludedCells: session.placed })
    addEvidence(t, result.candidates)
    matched++
  }
  const t2 = performance.now()

  return {
    frameId,
    width: frame.width,
    height: frame.height,
    tracks: seen.map((t) => trackView(t, session.region)),
    timingsMs: { segment: t1 - t0, match: t2 - t1, matched, pieces: seen.length },
  }
}
