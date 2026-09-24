/**
 * Image allocation and reuse helpers shared by every primitive in src/engine/image and src/engine/geom.
 *
 * The engine runs per camera frame on a phone, so the hot paths avoid garbage: every primitive takes an
 * optional `out` image and writes into it instead of allocating. These helpers implement that contract
 * in one place — allocate when `out` is missing, and throw when a caller hands in an `out` of the wrong
 * size (that is always a programming error, never a runtime condition to recover from). Unless a
 * function documents otherwise (blur, morphology), `out` must not share memory with the input.
 *
 * `scratch*` hands out module-level typed arrays for temporaries (row buffers, intermediate passes).
 * The engine is single-threaded inside its worker, so one buffer per named slot is safe as long as a
 * slot is never used by two functions that are live at the same time; each module uses its own slot names.
 */

import type { GrayImage, LabImage, Mask, RGBAImage } from '../types.ts'

export function createGray(width: number, height: number): GrayImage {
  return { width, height, data: new Float32Array(width * height) }
}

export function createLab(width: number, height: number): LabImage {
  const n = width * height
  return { width, height, L: new Float32Array(n), a: new Float32Array(n), b: new Float32Array(n) }
}

export function createMask(width: number, height: number): Mask {
  return { width, height, data: new Uint8Array(width * height) }
}

export function createRGBA(width: number, height: number): RGBAImage {
  return { width, height, data: new Uint8ClampedArray(width * height * 4) }
}

function checkSize(kind: string, out: { width: number; height: number }, width: number, height: number): void {
  if (out.width !== width || out.height !== height) {
    throw new Error(`${kind} output is ${out.width}x${out.height}, expected ${width}x${height}`)
  }
}

export function ensureGray(out: GrayImage | undefined, width: number, height: number): GrayImage {
  if (out === undefined) return createGray(width, height)
  checkSize('Gray', out, width, height)
  return out
}

export function ensureLab(out: LabImage | undefined, width: number, height: number): LabImage {
  if (out === undefined) return createLab(width, height)
  checkSize('Lab', out, width, height)
  return out
}

export function ensureMask(out: Mask | undefined, width: number, height: number): Mask {
  if (out === undefined) return createMask(width, height)
  checkSize('Mask', out, width, height)
  return out
}

export function ensureRGBA(out: RGBAImage | undefined, width: number, height: number): RGBAImage {
  if (out === undefined) return createRGBA(width, height)
  checkSize('RGBA', out, width, height)
  return out
}

const f32Slots = new Map<string, Float32Array>()
const f64Slots = new Map<string, Float64Array>()
const i32Slots = new Map<string, Int32Array>()
const u8Slots = new Map<string, Uint8Array>()

/** A reusable Float32Array of at least `length` elements. Contents are unspecified. */
export function scratchF32(slot: string, length: number): Float32Array {
  let buf = f32Slots.get(slot)
  if (buf === undefined || buf.length < length) {
    buf = new Float32Array(length)
    f32Slots.set(slot, buf)
  }
  return buf
}

/** A reusable Float64Array of at least `length` elements. Contents are unspecified. */
export function scratchF64(slot: string, length: number): Float64Array {
  let buf = f64Slots.get(slot)
  if (buf === undefined || buf.length < length) {
    buf = new Float64Array(length)
    f64Slots.set(slot, buf)
  }
  return buf
}

/** A reusable Int32Array of at least `length` elements. Contents are unspecified. */
export function scratchI32(slot: string, length: number): Int32Array {
  let buf = i32Slots.get(slot)
  if (buf === undefined || buf.length < length) {
    buf = new Int32Array(length)
    i32Slots.set(slot, buf)
  }
  return buf
}

/** A reusable Uint8Array of at least `length` elements. Contents are unspecified. */
export function scratchU8(slot: string, length: number): Uint8Array {
  let buf = u8Slots.get(slot)
  if (buf === undefined || buf.length < length) {
    buf = new Uint8Array(length)
    u8Slots.set(slot, buf)
  }
  return buf
}

/** A Mask over a reusable scratch buffer (see scratchU8); valid until the slot is next requested. */
export function scratchMask(slot: string, width: number, height: number): Mask {
  return { width, height, data: scratchU8(slot, width * height).subarray(0, width * height) }
}

/**
 * Index of `i` reflected into [0, n) without repeating the edge pixel (OpenCV BORDER_REFLECT_101:
 * ...2 1 | 0 1 2 ... n-1 | n-2 n-3...). Handles any overshoot, including kernels wider than the image.
 */
export function reflect101(i: number, n: number): number {
  if (n === 1) return 0
  const period = 2 * n - 2
  let j = i % period
  if (j < 0) j += period
  return j < n ? j : period - j
}
