/**
 * UI-thread handle to the engine worker.
 *
 * One worker for the whole app, created on first use: later phases load the puzzle reference into
 * it once and keep it there across screens. Calls go through Comlink, and every call also races
 * the worker's `error` event, because Comlink alone would wait forever if the worker script
 * failed to load or crashed.
 */

import { transfer, wrap } from 'comlink'
import type { FrameResult } from '../engine/types.ts'
import type { CapabilityReport, EngineApi } from './api.ts'

export interface EngineClient {
  probe(): Promise<CapabilityReport>
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
    processFrame: (bitmap, frameId) =>
      Promise.race([remote.processFrame(transfer(bitmap, [bitmap]), frameId), failed]),
  }
}
