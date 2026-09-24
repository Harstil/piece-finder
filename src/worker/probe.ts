/**
 * Capability probe, run inside the engine worker.
 *
 * Which inference backend and how many threads the engine can use depends on the phone: iOS clamps
 * hardwareConcurrency, WebGPU exists on some devices only, and threaded WASM needs cross-origin
 * isolation. The Camera check screen shows this report so those facts are observed on the real
 * device rather than assumed. It runs in the worker because that is where the engine will run.
 */

import type { CapabilityReport, GpuAdapterSummary, WebGpuReport } from './api.ts'

/**
 * Smallest module that uses SIMD: `(func (result v128) i32.const 0 i8x16.splat i8x16.popcnt)`.
 * The same bytes wasm-feature-detect uses; WebAssembly.validate only accepts it with SIMD support.
 */
const SIMD_PROBE_MODULE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15,
  253, 98, 11,
])

/**
 * How long to wait for a WebGPU adapter before reporting it unavailable (guessed; adapters normally
 * resolve in well under a second, this only guards against a browser that never answers).
 */
const WEBGPU_TIMEOUT_MS = 3000

export async function probeCapabilities(): Promise<CapabilityReport> {
  return {
    crossOriginIsolated: self.crossOriginIsolated,
    sharedArrayBuffer: typeof SharedArrayBuffer === 'function',
    wasmSimd: probeWasmSimd(),
    webgpu: await probeWebGpu(),
    offscreenCanvas2d: probeOffscreenCanvas2d(),
    hardwareConcurrency: navigator.hardwareConcurrency,
    deviceMemoryGb: (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? null,
    userAgent: navigator.userAgent,
  }
}

function probeWasmSimd(): boolean {
  try {
    return typeof WebAssembly === 'object' && WebAssembly.validate(SIMD_PROBE_MODULE)
  } catch {
    return false
  }
}

function probeOffscreenCanvas2d(): boolean {
  try {
    return typeof OffscreenCanvas === 'function' && new OffscreenCanvas(1, 1).getContext('2d') !== null
  } catch {
    return false
  }
}

async function probeWebGpu(): Promise<WebGpuReport> {
  if (!('gpu' in navigator) || !navigator.gpu) return { available: false, reason: 'no navigator.gpu' }
  try {
    const timeout = new Promise<'timeout'>((resolve) =>
      setTimeout(() => resolve('timeout'), WEBGPU_TIMEOUT_MS),
    )
    const adapter = await Promise.race([navigator.gpu.requestAdapter(), timeout])
    if (adapter === 'timeout') return { available: false, reason: 'requestAdapter timed out' }
    if (adapter === null) return { available: false, reason: 'no adapter' }
    return { available: true, adapter: summarizeAdapter(adapter) }
  } catch (error) {
    return { available: false, reason: error instanceof Error ? error.message : String(error) }
  }
}

/** `adapter.info` is recent (older WebGPU builds lack it), so it may be missing. */
function summarizeAdapter(adapter: GPUAdapter): GpuAdapterSummary | null {
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info
  if (!info) return null
  return {
    vendor: info.vendor,
    architecture: info.architecture,
    device: info.device,
    description: info.description,
  }
}
