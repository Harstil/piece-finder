/**
 * Frame pump: feeds live camera frames to the engine worker, one at a time.
 *
 * Why one at a time (backpressure): the phone produces 30 frames a second and the engine will
 * take longer than 33 ms per frame once segmentation and matching run. Queueing frames would only
 * add latency, so the pump waits for each result before it grabs the next frame; whatever frames
 * arrive meanwhile are skipped. The overlay then always describes a frame that is at most one
 * processing step old.
 *
 * Per frame: wait for a new video frame (requestVideoFrameCallback, or requestAnimationFrame where
 * that is missing) → createImageBitmap(video) → transfer the bitmap to the worker (no copy) →
 * await the FrameResult. The bitmap has the video's intrinsic size (videoWidth × videoHeight,
 * already upright for a portrait phone), which is the frame coordinate space of every result.
 *
 * It measures what the Camera check screen reports: capture→result latency and processed FPS.
 */

import type { FrameResult } from '../engine/types.ts'

/**
 * How far back the FPS and latency averages look (design choice): long enough to smooth
 * frame-to-frame jitter, short enough that a change (torch on, lens switch) shows within a second.
 */
const STATS_WINDOW_MS = 1000

/** How often stats are published to the UI (design choice): 4 Hz keeps the HUD readable and cheap. */
const STATS_PUBLISH_MS = 250

export interface FramePumpStats {
  /** Results received per second over the last STATS_WINDOW_MS. */
  processedFps: number
  /** Mean time from grabbing a frame to receiving its result, over the same window. */
  latencyMs: number
  /** The most recent result (its width/height are the frame coordinate space). */
  lastResult: FrameResult
  /** Results received since the pump started. */
  framesProcessed: number
}

export interface FramePumpHandlers {
  onStats(stats: FramePumpStats): void
  /** The pump stops after reporting an error; it restarts only when the caller starts a new one. */
  onError(error: unknown): void
}

export type ProcessFrame = (bitmap: ImageBitmap, frameId: number) => Promise<FrameResult>

/** Starts pumping frames from `video`; returns a function that stops it. */
export function startFramePump(
  video: HTMLVideoElement,
  processFrame: ProcessFrame,
  handlers: FramePumpHandlers,
): () => void {
  let stopped = false
  let frameId = 0
  let framesProcessed = 0
  let lastPublish = 0
  let pendingCallback: (() => void) | null = null
  /** Completion time and latency of recent results, oldest first. */
  const recent: { doneAt: number; latencyMs: number }[] = []

  function waitForNextFrame(): void {
    if (typeof video.requestVideoFrameCallback === 'function') {
      const handle = video.requestVideoFrameCallback(() => void pumpOnce())
      pendingCallback = () => video.cancelVideoFrameCallback(handle)
    } else {
      const handle = requestAnimationFrame(() => void pumpOnce())
      pendingCallback = () => cancelAnimationFrame(handle)
    }
  }

  async function pumpOnce(): Promise<void> {
    pendingCallback = null
    if (stopped) return
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || video.videoWidth === 0) {
      waitForNextFrame()
      return
    }
    let bitmap: ImageBitmap | null = null
    try {
      const capturedAt = performance.now()
      bitmap = await createImageBitmap(video)
      if (stopped) {
        bitmap.close()
        return
      }
      frameId += 1
      const result = await processFrame(bitmap, frameId)
      if (stopped) return
      record(result, capturedAt)
    } catch (error) {
      // If the call failed before the transfer (worker never loaded, clone error), this side still
      // owns the bitmap; closing an already transferred bitmap is a no-op.
      bitmap?.close()
      if (!stopped) {
        stopped = true
        handlers.onError(error)
      }
      return
    }
    waitForNextFrame()
  }

  function record(result: FrameResult, capturedAt: number): void {
    const now = performance.now()
    framesProcessed += 1
    recent.push({ doneAt: now, latencyMs: now - capturedAt })
    while (recent.length > 0 && now - recent[0].doneAt > STATS_WINDOW_MS) recent.shift()
    if (now - lastPublish < STATS_PUBLISH_MS) return
    lastPublish = now
    const span = now - recent[0].doneAt
    handlers.onStats({
      // n results span n - 1 intervals; with a single result there is no rate yet.
      processedFps: recent.length > 1 && span > 0 ? ((recent.length - 1) * 1000) / span : 0,
      latencyMs: recent.reduce((sum, r) => sum + r.latencyMs, 0) / recent.length,
      lastResult: result,
      framesProcessed,
    })
  }

  waitForNextFrame()
  return () => {
    stopped = true
    pendingCallback?.()
    pendingCallback = null
  }
}
