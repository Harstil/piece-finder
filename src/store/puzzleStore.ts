/**
 * The active puzzle, kept on the phone in IndexedDB (idb-keyval).
 *
 * What is stored is exactly what is needed to rebuild the engine's reference model after the app is
 * reopened: the box photo as the user took or picked it, the 4 picture corners they marked, the grid,
 * and which cells they have marked as placed. The reference itself is rebuilt in the worker (a second
 * or two) rather than stored, so a stored puzzle never goes stale when the engine improves.
 *
 * Storage can be unavailable (private browsing, blocked site data), so every access is wrapped: the app
 * still works for the current visit, it just won't remember the puzzle.
 */

import { del, get, set } from 'idb-keyval'
import type { GridSpec, Quad } from '../engine/types.ts'

const KEY = 'piece-finder:active-puzzle:v1'

export interface PuzzleRecord {
  id: string
  createdAt: number
  /** The original box photo (JPEG/PNG as picked). */
  photo: Blob
  /** Picture corners TL, TR, BR, BL in the photo's pixel coordinates (EXIF orientation applied). */
  corners: Quad
  grid: GridSpec
  placed: number[]
}

export async function loadPuzzle(): Promise<PuzzleRecord | null> {
  try {
    return (await get<PuzzleRecord>(KEY)) ?? null
  } catch {
    return null
  }
}

export async function savePuzzle(record: PuzzleRecord): Promise<void> {
  try {
    await set(KEY, record)
  } catch {
    // Storage unavailable: the puzzle lives only for this visit.
  }
}

export async function clearPuzzle(): Promise<void> {
  try {
    await del(KEY)
  } catch {
    // Nothing stored, or storage unavailable.
  }
}
