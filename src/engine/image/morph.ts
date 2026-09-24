/**
 * Binary morphology: erode, dilate, open, close with rectangular or elliptical structuring elements,
 * and hole filling.
 *
 * Segmentation masks need cleaning before contours are traced: opening removes speckle and thin bridges
 * between touching pieces, closing seals cracks along printed edges, hole filling removes glare holes.
 *
 * Semantics follow cv2.erode / cv2.dilate / cv2.morphologyEx with default arguments (golden-tested):
 * - the anchor is the kernel centre (⌊w/2⌋, ⌊h/2⌋), also for even sizes, and dilation uses the same
 *   offsets as erosion (OpenCV does not reflect the kernel);
 * - pixels outside the image never erode anything and never dilate anything (OpenCV's default
 *   BORDER_CONSTANT with its "morphology default border value"), so shapes touching the frame keep it;
 * - elliptical kernels are exactly cv2.getStructuringElement(MORPH_ELLIPSE, (w, h)).
 * Input pixels are foreground when nonzero; outputs are 0/1.
 *
 * Speed: rectangles are separable and use sliding window counts — the binary specialisation of the
 * van Herk/Gil-Werman idea, O(1) per pixel whatever the kernel size: a pixel erodes iff the window holds
 * no background, dilates iff it holds any foreground. Ellipses are handled row by row of the kernel
 * with per-row prefix counts, O(kernel height) per pixel.
 */

import type { Mask } from '../types.ts'
import { markOuterBackground } from './components.ts'
import { ensureMask, scratchI32, scratchMask, scratchU8 } from './create.ts'

export type KernelShape = 'rect' | 'ellipse'

export interface StructuringElement {
  width: number
  height: number
  shape: KernelShape
}

export function rectKernel(width: number, height = width): StructuringElement {
  return { width, height, shape: 'rect' }
}

export function ellipseKernel(width: number, height = width): StructuringElement {
  return { width, height, shape: 'ellipse' }
}

/** Round half to even, like OpenCV's saturate_cast<int>(double). */
function roundEven(v: number): number {
  const r = Math.round(v)
  return Math.abs(v - Math.trunc(v)) === 0.5 && r % 2 !== 0 ? r - 1 : r
}

/**
 * The horizontal run [start, end) of kernel row i for each of the `height` rows, exactly as OpenCV's
 * getStructuringElement(MORPH_ELLIPSE) draws it. Returned as [start0, end0, start1, end1, ...].
 */
export function ellipseRuns(width: number, height: number): Int32Array {
  const runs = new Int32Array(height * 2)
  const r = height >> 1
  const c = width >> 1
  const invR2 = r > 0 ? 1 / (r * r) : 0
  for (let i = 0; i < height; i++) {
    const dy = i - r
    let j1 = 0
    let j2 = 0
    if (Math.abs(dy) <= r) {
      const dx = roundEven(c * Math.sqrt((r * r - dy * dy) * invR2))
      j1 = Math.max(c - dx, 0)
      j2 = Math.min(c + dx + 1, width)
    }
    runs[i * 2] = j1
    runs[i * 2 + 1] = j2
  }
  return runs
}

function checkKernel(se: StructuringElement): void {
  if (!(se.width >= 1 && se.height >= 1) || !Number.isInteger(se.width) || !Number.isInteger(se.height)) {
    throw new Error(`structuring element must be a positive integer size, got ${se.width}x${se.height}`)
  }
}

/**
 * Core of erode/dilate. Each window is searched for a deciding value: for erosion any background pixel
 * in the window gives 0, for dilation any foreground pixel gives 1. Out-of-image pixels never decide.
 */
function morph(src: Mask, se: StructuringElement, erodeOp: boolean, out: Mask | undefined): Mask {
  checkKernel(se)
  const { width: w, height: h } = src
  const dst = ensureMask(out, w, h)
  if (se.shape === 'rect') morphRect(src, se.width, se.height, erodeOp, dst)
  else morphEllipse(src, se.width, se.height, erodeOp, dst)
  return dst
}

function morphRect(src: Mask, kw: number, kh: number, erodeOp: boolean, dst: Mask): void {
  const { width: w, height: h } = src
  const s = src.data
  const d = dst.data
  const ax = kw >> 1
  const ay = kh >> 1
  const hit = scratchU8('morph.hit', w * h)
  const prefix = scratchI32('morph.prefix', w + 1)
  const lo = scratchI32('morph.lo', w)
  const hi = scratchI32('morph.hi', w)
  for (let x = 0; x < w; x++) {
    lo[x] = Math.max(0, x - ax)
    hi[x] = Math.min(w, x - ax + kw) // exclusive
  }

  // Horizontal pass: hit[i] = 1 when the row window holds the target value.
  for (let y = 0; y < h; y++) {
    const row = y * w
    prefix[0] = 0
    if (erodeOp) for (let x = 0; x < w; x++) prefix[x + 1] = prefix[x] + (s[row + x] === 0 ? 1 : 0)
    else for (let x = 0; x < w; x++) prefix[x + 1] = prefix[x] + (s[row + x] !== 0 ? 1 : 0)
    for (let x = 0; x < w; x++) hit[row + x] = prefix[hi[x]] - prefix[lo[x]] > 0 ? 1 : 0
  }

  // Vertical pass with running per-column counts of hits over rows [y - ay, y - ay + kh).
  const counts = scratchI32('morph.counts', w)
  counts.fill(0, 0, w)
  for (let r = 0; r < Math.min(h, kh - ay); r++) {
    const row = r * w
    for (let x = 0; x < w; x++) counts[x] += hit[row + x]
  }
  const hitValue = erodeOp ? 0 : 1
  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) d[row + x] = counts[x] > 0 ? hitValue : 1 - hitValue
    const leave = y - ay
    const enter = y - ay + kh
    if (leave >= 0) for (let x = 0, o = leave * w; x < w; x++) counts[x] -= hit[o + x]
    if (enter < h) for (let x = 0, o = enter * w; x < w; x++) counts[x] += hit[o + x]
  }
}

function morphEllipse(src: Mask, kw: number, kh: number, erodeOp: boolean, dst: Mask): void {
  const { width: w, height: h } = src
  const s = src.data
  const d = dst.data
  const ax = kw >> 1
  const ay = kh >> 1
  const runs = ellipseRuns(kw, kh)
  const stride = w + 1
  // Per-row prefix counts of the target value, so any horizontal run is two lookups.
  const prefix = scratchI32('morph.rowPrefix', stride * h)
  for (let y = 0; y < h; y++) {
    const row = y * w
    const p = y * stride
    prefix[p] = 0
    if (erodeOp) for (let x = 0; x < w; x++) prefix[p + x + 1] = prefix[p + x] + (s[row + x] === 0 ? 1 : 0)
    else for (let x = 0; x < w; x++) prefix[p + x + 1] = prefix[p + x] + (s[row + x] !== 0 ? 1 : 0)
  }
  const hitValue = erodeOp ? 0 : 1
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let hitAny = false
      for (let i = 0; i < kh && !hitAny; i++) {
        const yy = y + i - ay
        if (yy < 0 || yy >= h) continue
        const a = Math.max(0, x + runs[i * 2] - ax)
        const b = Math.min(w, x + runs[i * 2 + 1] - ax)
        if (b > a && prefix[yy * stride + b] - prefix[yy * stride + a] > 0) hitAny = true
      }
      d[y * w + x] = hitAny ? hitValue : 1 - hitValue
    }
  }
}

export function erode(src: Mask, se: StructuringElement, out?: Mask): Mask {
  return morph(src, se, true, out)
}

export function dilate(src: Mask, se: StructuringElement, out?: Mask): Mask {
  return morph(src, se, false, out)
}

/** Erode then dilate: removes specks and bridges thinner than the kernel. */
export function open(src: Mask, se: StructuringElement, out?: Mask): Mask {
  const mid = scratchMask('morph.mid', src.width, src.height)
  return morph(morph(src, se, true, mid), se, false, out)
}

/** Dilate then erode: seals gaps and cracks narrower than the kernel. */
export function close(src: Mask, se: StructuringElement, out?: Mask): Mask {
  const mid = scratchMask('morph.mid', src.width, src.height)
  return morph(morph(src, se, false, mid), se, true, out)
}

/** Sets every background pixel not 4-connected to the image frame (a hole) to foreground. */
export function fillHoles(src: Mask, out?: Mask): Mask {
  const { width: w, height: h } = src
  const outer = markOuterBackground(src, scratchMask('morph.outer', w, h).data)
  const dst = ensureMask(out, w, h)
  const s = src.data
  const d = dst.data
  for (let i = 0; i < w * h; i++) d[i] = s[i] !== 0 || outer[i] === 0 ? 1 : 0
  return dst
}
