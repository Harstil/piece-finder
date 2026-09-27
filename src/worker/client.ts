/**
 * UI-thread handle to the engine worker.
 *
 * One worker for the whole app, created on first use: the puzzle reference is loaded into it once and
 * stays there across screens. Calls go through Comlink, and every call also races the worker's
 * `error` event, because Comlink alone would wait forever if the worker script failed to load or crashed.
 */

import { transfer, wrap } from 'comlink'
import type { FrameResult, GridSpec, Quad } from '../engine/types.ts'
import type { CapabilityReport, EngineApi, PuzzleInfo } from './api.ts'

export interface EngineClient {
  probe(): Promise<CapabilityReport>
  /** Transfers the photo to the worker; it is unusable (closed) on this side afterwards. */
  setPuzzle(photo: ImageBitmap, corners: Quad, grid: GridSpec, placed: number[]): Promise<PuzzleInfo>
  setPlaced(cells: number[]): Promise<void>
  setRegion(cells: number[] | null): Promise<void>
  /** Transfers the bitmap to the worker; it is unusable (closed) on this side afterwards. */
  processFrame(bitmap: ImageBitmap, frameId: number): Promise<FrameResult>
}

let client: EngineClient | null = null

export function getEngine(): EngineClient {
  client ??= createClient()
  return client
}

function createClient(): EngineClient {
  const worker = new Worker(new URL('./engine.worker.ts', import.meta.url), {
    type: 'module',
    name: 'piece-finder-engine',
  })
  const remote = wrap<EngineApi>(worker)
  const failed = new Promise<never>((_resolve, reject) => {
    worker.addEventListener('error', (event) => {
      reject(new Error(`Engine worker failed: ${event.message || 'the script could not be loaded'}`))
    })
  })
  // Callers see the rejection through the race below; this only marks it as handled.
  failed.catch(() => {})

  return {
    probe: () => Promise.race([remote.probe(), failed]),
    setPuzzle: (photo, corners, grid, placed) =>
      Promise.race([remote.setPuzzle(transfer(photo, [photo]), corners, grid, placed), failed]),
    setPlaced: (cells) => Promise.race([remote.setPlaced(cells), failed]),
    setRegion: (cells) => Promise.race([remote.setRegion(cells), failed]),
    processFrame: (bitmap, frameId) =>
      Promise.race([remote.processFrame(transfer(bitmap, [bitmap]), frameId), failed]),
  }
}
