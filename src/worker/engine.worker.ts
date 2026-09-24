/**
 * The engine worker: everything heavy runs here, off the UI thread, so the camera view and
 * overlays stay smooth.
 *
 * For now it proves the frame path end to end: a camera frame arrives as a transferred ImageBitmap,
 * its pixels are read through an OffscreenCanvas (the step every later stage needs), and the time
 * that took is reported. Segmentation, tracking and matching (src/engine/) plug in here later and
 * fill `tracks`.
 */

import { expose } from 'comlink'
import type { FrameResult } from '../engine/types.ts'
import type { EngineApi } from './api.ts'
import { probeCapabilities } from './probe.ts'

/** Reused across frames: allocating a canvas per frame costs more than reading the pixels. */
let canvas: OffscreenCanvas | null = null
let context: OffscreenCanvasRenderingContext2D | null = null

function readPixels(bitmap: ImageBitmap): ImageData {
  if (typeof OffscreenCanvas !== 'function') {
    throw new Error('OffscreenCanvas is not available in this worker')
  }
  if (canvas === null || context === null) {
    canvas = new OffscreenCanvas(bitmap.width, bitmap.height)
    // willReadFrequently keeps the canvas in CPU memory, so getImageData is not a GPU readback.
    context = canvas.getContext('2d', { willReadFrequently: true })
    if (context === null) throw new Error('OffscreenCanvas has no 2D context in this worker')
  }
  if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
    canvas.width = bitmap.width
    canvas.height = bitmap.height
  }
  context.drawImage(bitmap, 0, 0)
  return context.getImageData(0, 0, bitmap.width, bitmap.height)
}

const api: EngineApi = {
  probe: probeCapabilities,

  async processFrame(bitmap: ImageBitmap, frameId: number): Promise<FrameResult> {
    try {
      const start = performance.now()
      const pixels = readPixels(bitmap)
      const readPixelsMs = performance.now() - start
      return {
        frameId,
        width: pixels.width,
        height: pixels.height,
        tracks: [],
        timingsMs: { readPixels: readPixelsMs },
      }
    } finally {
      bitmap.close()
    }
  },
}

expose(api)
