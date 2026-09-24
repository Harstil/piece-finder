/**
 * Exact quarter-turn rotation of Gray, Lab and Mask images (no resampling).
 *
 * The matcher compares a canonical piece against every grid cell at 4 rotations; rotating the square
 * canonical patch by quarter turns is a pure pixel permutation, so it is exact and cheap.
 *
 * Direction convention (matches `Rotation` in types.ts): r means r × 90° CLOCKWISE AS SEEN ON SCREEN
 * (image coordinates, y down). For r = 1 the top-left pixel moves to the top-right, the top row becomes
 * the right column. r is taken modulo 4, negative values included (−1 ≡ 3 = a quarter turn
 * counter-clockwise). Non-square images work too: odd r swaps width and height.
 *
 * Pixel mapping for a W×H source (pixel centres):
 *   r = 1: (x, y) → (H−1−y, x)    r = 2: (x, y) → (W−1−x, H−1−y)    r = 3: (x, y) → (y, W−1−x)
 * The output must not share memory with the input.
 */

import { ensureGray, ensureLab, ensureMask } from '../image/create.ts'
import type { GrayImage, LabImage, Mask, Point } from '../types.ts'

function quarter(r: number): number {
  return ((Math.round(r) % 4) + 4) % 4
}

/** Size of a width×height image after r quarter turns. */
export function rotatedSize(width: number, height: number, r: number): { width: number; height: number } {
  return quarter(r) % 2 === 0 ? { width, height } : { width: height, height: width }
}

/** Where source pixel p of a width×height image lands after r clockwise quarter turns. */
export function rotatePoint(p: Point, r: number, width: number, height: number): Point {
  switch (quarter(r)) {
    case 1:
      return { x: height - 1 - p.y, y: p.x }
    case 2:
      return { x: width - 1 - p.x, y: height - 1 - p.y }
    case 3:
      return { x: p.y, y: width - 1 - p.x }
    default:
      return { x: p.x, y: p.y }
  }
}

type Plane = Float32Array | Uint8Array

/** Rotates one W×H plane into dst (sized for the rotation) by r quarter turns clockwise. */
function rotatePlane(src: Plane, W: number, H: number, r: number, dst: Plane): void {
  if (src.buffer === dst.buffer && src.byteOffset === dst.byteOffset) {
    throw new Error('rotate: output must not share memory with the input')
  }
  switch (r) {
    case 0:
      dst.set(src.subarray(0, W * H))
      return
    case 1: // dst is H wide, W tall: dst(x, y) = src(y, H−1−x)
      for (let y = 0; y < W; y++) {
        const row = y * H
        for (let x = 0; x < H; x++) dst[row + x] = src[(H - 1 - x) * W + y]
      }
      return
    case 2:
      for (let i = 0, n = W * H; i < n; i++) dst[i] = src[n - 1 - i]
      return
    default: // 3 — dst is H wide, W tall: dst(x, y) = src(W−1−y, x)
      for (let y = 0; y < W; y++) {
        const row = y * H
        const sx = W - 1 - y
        for (let x = 0; x < H; x++) dst[row + x] = src[x * W + sx]
      }
  }
}

export function rotateGray(src: GrayImage, r: number, out?: GrayImage): GrayImage {
  const q = quarter(r)
  const size = rotatedSize(src.width, src.height, q)
  const dst = ensureGray(out, size.width, size.height)
  rotatePlane(src.data, src.width, src.height, q, dst.data)
  return dst
}

export function rotateLab(src: LabImage, r: number, out?: LabImage): LabImage {
  const q = quarter(r)
  const size = rotatedSize(src.width, src.height, q)
  const dst = ensureLab(out, size.width, size.height)
  rotatePlane(src.L, src.width, src.height, q, dst.L)
  rotatePlane(src.a, src.width, src.height, q, dst.a)
  rotatePlane(src.b, src.width, src.height, q, dst.b)
  return dst
}

export function rotateMask(src: Mask, r: number, out?: Mask): Mask {
  const q = quarter(r)
  const size = rotatedSize(src.width, src.height, q)
  const dst = ensureMask(out, size.width, size.height)
  rotatePlane(src.data, src.width, src.height, q, dst.data)
  return dst
}
