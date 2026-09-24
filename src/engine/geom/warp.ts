/**
 * Perspective warps with bilinear sampling for Gray, Lab, RGBA and masks, plus a validity mask.
 *
 * This is how the engine brings things into a common frame: the box photo into the rectified motif, and
 * every piece (via its 4 core corners) into the canonical square that is compared against grid cells.
 *
 * API: warps take a destination-to-source homography (inverse mapping: for each output pixel, where to
 * sample the input), which is what the sampling loop needs and what lets callers compose transforms
 * freely. rectToQuadHomography builds it from a source quad (TL, TR, BR, BL) and an output size.
 *
 * Conventions:
 * - Pixel centres are at integer coordinates (see types.ts). rectToQuadHomography maps the quad's corners
 *   to the outer corners of the output image, (−0.5, −0.5) … (width − 0.5, height − 0.5), so the quad
 *   is covered edge to edge and every output pixel represents an equal share of it.
 * - A sample is valid when it lies inside the source's pixel-centre rectangle [0, w−1] × [0, h−1]
 *   (± VALID_EPSILON) with a positive homogeneous w. Invalid pixels get the fill value (0) and 0 in the
 *   optional `valid` mask. OpenCV (BORDER_CONSTANT) instead blends up to one pixel of border colour
 *   into samples just outside that rectangle; away from it the two agree within one grey level on
 *   8-bit data — OpenCV quantises coordinates to 1/32 px, this code does not (golden-tested).
 * - Masks are sampled bilinearly as 0/1 and thresholded at 0.5, which gives smoother warped outlines
 *   than nearest neighbour.
 */

import { ensureGray, ensureLab, ensureMask, ensureRGBA, scratchF64, scratchU8 } from '../image/create.ts'
import type { GrayImage, LabImage, Mask, Point, RGBAImage } from '../types.ts'
import { homographyFromQuad, type Homography } from './homography.ts'

/**
 * Samples this far outside the pixel-centre rectangle still count as inside (and are clamped). Covers
 * floating-point error of edge-aligned warps such as the identity. Guessed; far below any visible offset.
 */
const VALID_EPSILON = 1e-4

export interface WarpOptions<T> {
  /** Output image to fill instead of allocating one. */
  out?: T
  /** When given, receives 1 where the output pixel sampled inside the source and 0 elsewhere. */
  valid?: Mask
}

/**
 * dst→src homography that maps a width×height output rectangle onto `quad` (TL, TR, BR, BL in source
 * coordinates), corners to the output's outer pixel corners. Null for a degenerate quad.
 */
export function rectToQuadHomography(quad: readonly Point[], width: number, height: number): Homography | null {
  const rect: Point[] = [
    { x: -0.5, y: -0.5 },
    { x: width - 0.5, y: -0.5 },
    { x: width - 0.5, y: height - 0.5 },
    { x: -0.5, y: height - 0.5 },
  ]
  return homographyFromQuad(rect, quad)
}

/**
 * Source coordinates for output row y. Writes x into sx, y into sy (both clamped into the source) and
 * 1/0 validity into ok.
 */
function rowCoords(
  H: Homography,
  y: number,
  width: number,
  srcW: number,
  srcH: number,
  sx: Float64Array,
  sy: Float64Array,
  ok: Uint8Array,
): void {
  let X = H[1] * y + H[2]
  let Y = H[4] * y + H[5]
  let W = H[7] * y + H[8]
  const maxX = srcW - 1
  const maxY = srcH - 1
  for (let x = 0; x < width; x++) {
    let u = X / W
    let v = Y / W
    const inside = W > 0 && u >= -VALID_EPSILON && u <= maxX + VALID_EPSILON && v >= -VALID_EPSILON && v <= maxY + VALID_EPSILON
    if (inside) {
      u = u < 0 ? 0 : u > maxX ? maxX : u
      v = v < 0 ? 0 : v > maxY ? maxY : v
    }
    sx[x] = u
    sy[x] = v
    ok[x] = inside ? 1 : 0
    X += H[0]
    Y += H[3]
    W += H[6]
  }
}

interface Scratch {
  sx: Float64Array
  sy: Float64Array
  ok: Uint8Array
}

function rowScratch(width: number): Scratch {
  return { sx: scratchF64('warp.sx', width), sy: scratchF64('warp.sy', width), ok: scratchU8('warp.ok', width) }
}

function checkValid(valid: Mask | undefined, width: number, height: number): void {
  if (valid !== undefined) ensureMask(valid, width, height)
}

/** Bilinear sample of one plane at a clamped in-range position. */
function sample(p: ArrayLike<number>, w: number, h: number, u: number, v: number): number {
  let x0 = Math.floor(u)
  let y0 = Math.floor(v)
  if (x0 > w - 2) x0 = w > 1 ? w - 2 : 0
  if (y0 > h - 2) y0 = h > 1 ? h - 2 : 0
  const fx = u - x0
  const fy = v - y0
  const x1 = x0 + 1 < w ? x0 + 1 : x0
  const r0 = y0 * w
  const r1 = (y0 + 1 < h ? y0 + 1 : y0) * w
  const top = p[r0 + x0] + (p[r0 + x1] - p[r0 + x0]) * fx
  const bottom = p[r1 + x0] + (p[r1 + x1] - p[r1 + x0]) * fx
  return top + (bottom - top) * fy
}

export function warpGray(
  src: GrayImage,
  dstToSrc: Homography,
  width: number,
  height: number,
  opts: WarpOptions<GrayImage> = {},
): GrayImage {
  const dst = ensureGray(opts.out, width, height)
  checkValid(opts.valid, width, height)
  const { sx, sy, ok } = rowScratch(width)
  const d = dst.data
  for (let y = 0; y < height; y++) {
    rowCoords(dstToSrc, y, width, src.width, src.height, sx, sy, ok)
    const row = y * width
    for (let x = 0; x < width; x++) {
      d[row + x] = ok[x] === 1 ? sample(src.data, src.width, src.height, sx[x], sy[x]) : 0
    }
    if (opts.valid !== undefined) opts.valid.data.set(ok.subarray(0, width), row)
  }
  return dst
}

export function warpLab(
  src: LabImage,
  dstToSrc: Homography,
  width: number,
  height: number,
  opts: WarpOptions<LabImage> = {},
): LabImage {
  const dst = ensureLab(opts.out, width, height)
  checkValid(opts.valid, width, height)
  const { sx, sy, ok } = rowScratch(width)
  const { width: w, height: h } = src
  for (let y = 0; y < height; y++) {
    rowCoords(dstToSrc, y, width, w, h, sx, sy, ok)
    const row = y * width
    for (let x = 0; x < width; x++) {
      const i = row + x
      if (ok[x] === 1) {
        dst.L[i] = sample(src.L, w, h, sx[x], sy[x])
        dst.a[i] = sample(src.a, w, h, sx[x], sy[x])
        dst.b[i] = sample(src.b, w, h, sx[x], sy[x])
      } else {
        dst.L[i] = 0
        dst.a[i] = 0
        dst.b[i] = 0
      }
    }
    if (opts.valid !== undefined) opts.valid.data.set(ok.subarray(0, width), row)
  }
  return dst
}

/** All four channels are interpolated; invalid pixels become transparent black. */
export function warpRGBA(
  src: RGBAImage,
  dstToSrc: Homography,
  width: number,
  height: number,
  opts: WarpOptions<RGBAImage> = {},
): RGBAImage {
  const dst = ensureRGBA(opts.out, width, height)
  checkValid(opts.valid, width, height)
  const { sx, sy, ok } = rowScratch(width)
  const { width: w, height: h, data: s } = src
  const d = dst.data
  for (let y = 0; y < height; y++) {
    rowCoords(dstToSrc, y, width, w, h, sx, sy, ok)
    const row = y * width
    for (let x = 0; x < width; x++) {
      const o = (row + x) * 4
      if (ok[x] === 0) {
        d[o] = d[o + 1] = d[o + 2] = d[o + 3] = 0
        continue
      }
      const u = sx[x]
      const v = sy[x]
      let x0 = Math.floor(u)
      let y0 = Math.floor(v)
      if (x0 > w - 2) x0 = w > 1 ? w - 2 : 0
      if (y0 > h - 2) y0 = h > 1 ? h - 2 : 0
      const fx = u - x0
      const fy = v - y0
      const x1 = x0 + 1 < w ? x0 + 1 : x0
      const y1 = y0 + 1 < h ? y0 + 1 : y0
      const p00 = (y0 * w + x0) * 4
      const p01 = (y0 * w + x1) * 4
      const p10 = (y1 * w + x0) * 4
      const p11 = (y1 * w + x1) * 4
      for (let c = 0; c < 4; c++) {
        const top = s[p00 + c] + (s[p01 + c] - s[p00 + c]) * fx
        const bottom = s[p10 + c] + (s[p11 + c] - s[p10 + c]) * fx
        d[o + c] = top + (bottom - top) * fy
      }
    }
    if (opts.valid !== undefined) opts.valid.data.set(ok.subarray(0, width), row)
  }
  return dst
}

/** Nonzero source pixels count as 1; the bilinear coverage is thresholded at 0.5. Outside → 0. */
export function warpMask(src: Mask, dstToSrc: Homography, width: number, height: number, opts: WarpOptions<Mask> = {}): Mask {
  const dst = ensureMask(opts.out, width, height)
  checkValid(opts.valid, width, height)
  const { sx, sy, ok } = rowScratch(width)
  const { width: w, height: h, data: s } = src
  const d = dst.data
  for (let y = 0; y < height; y++) {
    rowCoords(dstToSrc, y, width, w, h, sx, sy, ok)
    const row = y * width
    for (let x = 0; x < width; x++) {
      if (ok[x] === 0) {
        d[row + x] = 0
        continue
      }
      const u = sx[x]
      const v = sy[x]
      let x0 = Math.floor(u)
      let y0 = Math.floor(v)
      if (x0 > w - 2) x0 = w > 1 ? w - 2 : 0
      if (y0 > h - 2) y0 = h > 1 ? h - 2 : 0
      const fx = u - x0
      const fy = v - y0
      const x1 = x0 + 1 < w ? x0 + 1 : x0
      const r0 = y0 * w
      const r1 = (y0 + 1 < h ? y0 + 1 : y0) * w
      const top = (s[r0 + x0] !== 0 ? 1 - fx : 0) + (s[r0 + x1] !== 0 ? fx : 0)
      const bottom = (s[r1 + x0] !== 0 ? 1 - fx : 0) + (s[r1 + x1] !== 0 ? fx : 0)
      d[row + x] = top * (1 - fy) + bottom * fy >= 0.5 ? 1 : 0
    }
    if (opts.valid !== undefined) opts.valid.data.set(ok.subarray(0, width), row)
  }
  return dst
}
