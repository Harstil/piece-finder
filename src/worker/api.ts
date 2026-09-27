/**
 * The engine worker's API, as seen through Comlink.
 *
 * The UI thread never runs vision code: it hands the worker the box photo once (setPuzzle), then
 * streams camera frames as transferred ImageBitmaps (zero-copy) and draws the FrameResults it gets
 * back. This file is the typed seam between the two sides; engine.worker.ts implements it and
 * client.ts calls it.
 */

import type { FrameResult, GridSpec, Quad, ReferenceSummary } from '../engine/types.ts'

/** What this phone and browser can do, as seen from inside the worker. */
export interface CapabilityReport {
  /** COOP/COEP took effect; required for SharedArrayBuffer and threaded WASM. */
  crossOriginIsolated: boolean
  sharedArrayBuffer: boolean
  /** WebAssembly fixed-width SIMD (the onnxruntime-web WASM baseline). */
  wasmSimd: boolean
  webgpu: WebGpuReport
  /** OffscreenCanvas with a 2D context inside the worker (how frame pixels are read). */
  offscreenCanvas2d: boolean
  /** Logical cores the browser admits to; iOS clamps this (often to 2). */
  hardwareConcurrency: number
  /** Approximate RAM in GB (Chromium only), or null when not exposed. */
  deviceMemoryGb: number | null
  userAgent: string
}

export type WebGpuReport =
  | { available: true; adapter: GpuAdapterSummary | null }
  | { available: false; reason: string }

/** GPUAdapterInfo fields; any of them may be empty strings (browsers redact them). */
export interface GpuAdapterSummary {
  vendor: string
  architecture: string
  device: string
  description: string
}

/** RGBA pixels (ImageData-compatible), transferred back from the worker. */
export interface Pixels {
  width: number
  height: number
  data: Uint8ClampedArray
}

export interface PuzzleInfo {
  summary: ReferenceSummary
  /** The box picture straightened to the motif rectangle, for the UI to draw grids and highlights on. */
  preview: Pixels
  buildMs: number
}

export interface EngineApi {
  probe(): Promise<CapabilityReport>
  /**
   * Builds the reference model from the box photo (the bitmap is transferred and closed) and starts a
   * scanning session with the given cells already placed.
   */
  setPuzzle(photo: ImageBitmap, corners: Quad, grid: GridSpec, placed: number[]): Promise<PuzzleInfo>
  setPlaced(cells: number[]): Promise<void>
  /** Cells of the Region finder's area, or null to switch it off. */
  setRegion(cells: number[] | null): Promise<void>
  /**
   * Process one camera frame. The bitmap must be transferred (it is closed by the worker), so each
   * frame costs no copy on the UI thread. Without a puzzle it only reports timings.
   */
  processFrame(bitmap: ImageBitmap, frameId: number): Promise<FrameResult>
}
