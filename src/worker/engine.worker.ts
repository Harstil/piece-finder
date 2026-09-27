/**
 * The engine worker: everything heavy runs here, off the UI thread, so the camera view and overlays
 * stay smooth.
 *
 * setPuzzle builds the reference model from the box photo and opens a scanning session
 * (src/engine/pipeline.ts). Each camera frame is then read through an OffscreenCanvas — shrunk to
 * PROCESS_LONG_SIDE on the way, which the 2D canvas does on the GPU far faster than the engine could —
 * and stepped through the pipeline: segmentation, tracking and matching.
 */

import { expose, transfer } from 'comlink'
import { buildReference, referenceSummary } from '../engine/index.ts'
import { createSession, setPlaced, stepFrame, type Session } from '../engine/pipeline.ts'
import type { FrameResult, GridSpec, Quad, RGBAImage } from '../engine/types.ts'
import type { EngineApi, PuzzleInfo } from './api.ts'
import { probeCapabilities } from './probe.ts'

/**
 * Long side camera frames are processed at. Measured on desktop Node (eval/pipeline-bench.ts): pieces
 * keep ≥ 40 px cores at arm's length while segmentation stays ~150 ms; the phone is a few times slower.
 */
const PROCESS_LONG_SIDE = 1280
/** Box photos larger than this are shrunk first: iOS caps canvas area at ~16.7 MP. */
const PHOTO_LONG_SIDE = 4096

/** Reused across frames: allocating a canvas per frame costs more than reading the pixels. */
let canvas: OffscreenCanvas | null = null
let context: OffscreenCanvasRenderingContext2D | null = null
let session: Session | null = null

function readPixels(bitmap: ImageBitmap, maxLongSide: number): RGBAImage {
  if (typeof OffscreenCanvas !== 'function') {
    throw new Error('OffscreenCanvas is not available in this worker')
  }
  const scale = Math.min(1, maxLongSide / Math.max(bitmap.width, bitmap.height))
  const width = Math.max(1, Math.round(bitmap.width * scale))
  const height = Math.max(1, Math.round(bitmap.height * scale))
  if (canvas === null || context === null) {
    canvas = new OffscreenCanvas(width, height)
    // willReadFrequently keeps the canvas in CPU memory, so getImageData is not a GPU readback.
    context = canvas.getContext('2d', { willReadFrequently: true })
    if (context === null) throw new Error('OffscreenCanvas has no 2D context in this worker')
  }
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = 'high'
  context.drawImage(bitmap, 0, 0, width, height)
  const pixels = context.getImageData(0, 0, width, height)
  return { width, height, data: pixels.data }
}

const api: EngineApi = {
  probe: probeCapabilities,

  async setPuzzle(photo: ImageBitmap, corners: Quad, grid: GridSpec, placed: number[]): Promise<PuzzleInfo> {
    try {
      const start = performance.now()
      const scale = Math.min(1, PHOTO_LONG_SIDE / Math.max(photo.width, photo.height))
      const pixels = readPixels(photo, PHOTO_LONG_SIDE)
      // Corners arrive in the photo's own pixel coordinates; follow the shrink (pixel-edge convention).
      const scaled = corners.map((c) => ({ x: (c.x + 0.5) * scale - 0.5, y: (c.y + 0.5) * scale - 0.5 })) as Quad
      const ref = buildReference(pixels, scaled, grid)
      session = createSession(ref, placed)
      const preview = { width: ref.preview.width, height: ref.preview.height, data: new Uint8ClampedArray(ref.preview.data) }
      return transfer({ summary: referenceSummary(ref), preview, buildMs: performance.now() - start }, [preview.data.buffer])
    } finally {
      photo.close()
    }
  },

  async setPlaced(cells: number[]): Promise<void> {
    if (session !== null) setPlaced(session, cells)
  },

  async setRegion(cells: number[] | null): Promise<void> {
    if (session !== null) session.region = cells === null ? null : new Set(cells)
  },

  async processFrame(bitmap: ImageBitmap, frameId: number): Promise<FrameResult> {
    try {
      const start = performance.now()
      const frame = readPixels(bitmap, PROCESS_LONG_SIDE)
      const readPixelsMs = performance.now() - start
      if (session === null) {
        return { frameId, width: frame.width, height: frame.height, tracks: [], timingsMs: { readPixels: readPixelsMs } }
      }
      const result = stepFrame(session, frame, frameId)
      result.timingsMs.readPixels = readPixelsMs
      return result
    } finally {
      bitmap.close()
    }
  },
}

expose(api)
