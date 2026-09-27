/**
 * The active puzzle as the UI sees it: loaded from the phone's storage at start-up, handed to the engine
 * worker (which builds its reference model from the box photo), and kept in sync as the user marks
 * pieces as placed or picks a Region-finder area.
 *
 * States: 'loading' (reading storage), 'none' (no puzzle yet), 'preparing' (the worker is building the
 * reference, a second or two on a phone), 'ready', or 'error'. The straightened box picture comes back
 * from the worker as pixels and is turned into an ImageBitmap once, so every screen can draw it cheaply.
 */

import { useCallback, useEffect, useState } from 'react'
import type { GridSpec, Quad, ReferenceSummary } from '../engine/types.ts'
import { clearPuzzle, loadPuzzle, savePuzzle, type PuzzleRecord } from '../store/puzzleStore.ts'
import { getEngine } from '../worker/client.ts'
import { errorMessage } from './errorMessage.ts'

export interface PuzzleView {
  record: PuzzleRecord
  summary: ReferenceSummary
  /** The straightened box picture (motif rectangle only). */
  picture: ImageBitmap
}

export type PuzzleStatus =
  | { kind: 'loading' }
  | { kind: 'none' }
  | { kind: 'preparing'; record: PuzzleRecord }
  | { kind: 'ready'; view: PuzzleView }
  | { kind: 'error'; message: string; record: PuzzleRecord | null }

export interface PuzzleControls {
  status: PuzzleStatus
  /** Cells of the Region finder's area (not stored: it is a per-session search). */
  region: ReadonlySet<number> | null
  create(photo: Blob, corners: Quad, grid: GridSpec): Promise<void>
  setPlaced(cells: number[]): void
  setRegion(cells: ReadonlySet<number> | null): void
  discard(): Promise<void>
}

/**
 * Decodes the stored photo exactly as the corner editor displayed it: through an <img>, which applies
 * the camera's EXIF orientation. (createImageBitmap's own EXIF handling differs between browsers, and a
 * rotated bitmap would put the user's corners on the wrong part of the photo.)
 */
async function decodePhoto(blob: Blob): Promise<ImageBitmap> {
  const url = URL.createObjectURL(blob)
  try {
    const img = new Image()
    img.src = url
    await img.decode()
    const canvas = document.createElement('canvas')
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    const ctx = canvas.getContext('2d')
    if (ctx === null) throw new Error('This browser cannot draw the box photo.')
    ctx.drawImage(img, 0, 0)
    return await createImageBitmap(canvas)
  } finally {
    URL.revokeObjectURL(url)
  }
}

async function prepare(record: PuzzleRecord): Promise<PuzzleView> {
  const photo = await decodePhoto(record.photo)
  const info = await getEngine().setPuzzle(photo, record.corners, record.grid, record.placed)
  const { width, height, data } = info.preview
  const picture = await createImageBitmap(new ImageData(new Uint8ClampedArray(data), width, height))
  return { record, summary: info.summary, picture }
}

export function usePuzzle(): PuzzleControls {
  const [status, setStatus] = useState<PuzzleStatus>({ kind: 'loading' })
  const [region, setRegionState] = useState<ReadonlySet<number> | null>(null)

  const open = useCallback(async (record: PuzzleRecord) => {
    setStatus({ kind: 'preparing', record })
    try {
      const view = await prepare(record)
      setStatus({ kind: 'ready', view })
    } catch (error) {
      setStatus({ kind: 'error', message: errorMessage(error), record })
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    void loadPuzzle().then((record) => {
      if (cancelled) return
      if (record === null) setStatus({ kind: 'none' })
      else void open(record)
    })
    return () => {
      cancelled = true
    }
  }, [open])

  const create = useCallback(
    async (photo: Blob, corners: Quad, grid: GridSpec) => {
      const record: PuzzleRecord = { id: `${Date.now()}`, createdAt: Date.now(), photo, corners, grid, placed: [] }
      setRegionState(null)
      await savePuzzle(record)
      await open(record)
    },
    [open],
  )

  const setPlaced = useCallback(
    (cells: number[]) => {
      if (status.kind !== 'ready') return
      const record = { ...status.view.record, placed: cells }
      setStatus({ kind: 'ready', view: { ...status.view, record } })
      void savePuzzle(record)
      void getEngine().setPlaced(cells)
    },
    [status],
  )

  const setRegion = useCallback((cells: ReadonlySet<number> | null) => {
    setRegionState(cells)
    void getEngine().setRegion(cells === null ? null : [...cells])
  }, [])

  const discard = useCallback(async () => {
    await clearPuzzle()
    setRegionState(null)
    setStatus({ kind: 'none' })
  }, [])

  return { status, region, create, setPlaced, setRegion, discard }
}
