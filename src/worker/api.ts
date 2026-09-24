/**
 * The engine worker's API, as seen through Comlink.
 *
 * The UI thread never runs vision code: it captures camera frames as ImageBitmaps and transfers
 * them to the worker (zero-copy), which answers with a FrameResult. This file is the typed seam
 * between the two sides; engine.worker.ts implements it and client.ts calls it. Later phases add
 * methods here (reference building, matching) as they are used.
 */

import type { FrameResult } from '../engine/types.ts'

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

export interface EngineApi {
  probe(): Promise<CapabilityReport>
  /**
   * Process one camera frame. The bitmap must be transferred (it is closed by the worker), so
   * each frame costs no copy on the UI thread.
   */
  processFrame(bitmap: ImageBitmap, frameId: number): Promise<FrameResult>
}
