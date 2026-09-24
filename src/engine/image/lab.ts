/**
 * Colour conversion: sRGB8 → CIE Lab (D65) and sRGB8 → luma.
 *
 * Lab is the engine's working colour space: L carries texture for the ZNCC cue, a/b carry chroma for the
 * colour cue, and distances in Lab roughly follow perceived difference. The conversion must agree with
 * OpenCV's float path (cv2.cvtColor on float32 RGB in [0,1] → L in [0,100]) because the Python dataset
 * tools and any debugging notebooks use that, so descriptors computed on either side are comparable.
 *
 * Pipeline per pixel (OpenCV's constants): sRGB gamma expansion → linear RGB → XYZ with the sRGB/D65
 * matrix → divide by the D65 white point → f(t) = cbrt(t) above 0.008856, else 7.787 t + 16/116 →
 * L = 116 f(Y) − 16, a = 500 (f(X) − f(Y)), b = 200 (f(Y) − f(Z)).
 *
 * This is the exact formula. OpenCV's own float path approximates it for sRGB input with a 33³
 * trilinear lookup table, which deviates from the formula by up to 0.19 in L and 0.44 in a/b (measured
 * on 200 000 random colours; mean 0.08). So "matching OpenCV" means agreeing within 0.5 per channel,
 * and the remaining difference is OpenCV's approximation, not ours.
 *
 * Speed: the gamma step is an exact 256-entry table (inputs are 8-bit). f(t) is a 4096-interval table
 * with linear interpolation over t ∈ [0, 1] (X, Y, Z are normalised to that range by the white point);
 * the interpolation error is < 5e-6 in f, i.e. < 1e-3 in L (measured in lab.test.ts against Math.cbrt).
 * That avoids three cube roots per pixel, the dominant cost on a 1920×1080 frame.
 *
 * Luma uses Rec.601 weights (0.299, 0.587, 0.114), the same as cv2.COLOR_RGB2GRAY, on the 0..255 scale
 * and without rounding, so GrayImage values from rgbaToGray are directly comparable to OpenCV grey.
 */

import type { GrayImage, LabImage, RGBAImage } from '../types.ts'
import { ensureGray, ensureLab } from './create.ts'

// OpenCV's sRGB → XYZ (D65) matrix, rows pre-divided by the D65 white point (Xn = 0.950456, Zn = 1.088754).
const XN = 0.950456
const ZN = 1.088754
const MX0 = 0.412453 / XN
const MX1 = 0.35758 / XN
const MX2 = 0.180423 / XN
const MY0 = 0.212671
const MY1 = 0.71516
const MY2 = 0.072169
const MZ0 = 0.019334 / ZN
const MZ1 = 0.119193 / ZN
const MZ2 = 0.950227 / ZN

/** CIE threshold between the cube-root and the linear branch of f(t) (the standard constant OpenCV uses). */
const LAB_EPSILON = 0.008856

/** sRGB8 → linear [0,1], exact per code value. */
const GAMMA_LUT = new Float32Array(256)
for (let i = 0; i < 256; i++) {
  const c = i / 255
  GAMMA_LUT[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
}

/** Intervals of the f(t) table over [0, 1]; 4096 keeps the L error below 1e-3 (measured). */
const F_LUT_SIZE = 4096
/** One guard entry past t = 1 absorbs rounding just above 1, so lookups need no clamping. */
const F_LUT = new Float32Array(F_LUT_SIZE + 2)
for (let i = 0; i < F_LUT.length; i++) F_LUT[i] = labF(i / F_LUT_SIZE)

function labF(t: number): number {
  return t > LAB_EPSILON ? Math.cbrt(t) : 7.787 * t + 16 / 116
}

/**
 * Table-interpolated f(t). Only called with t = a white-point-normalised X, Y or Z of an 8-bit colour:
 * every matrix row above sums to exactly 1 and the gamma table lies in [0, 1], so t ∈ [0, 1] up to
 * rounding and the index never leaves the table.
 */
function labFFast(t: number): number {
  const s = t * F_LUT_SIZE
  const i = s | 0
  const f0 = F_LUT[i]
  return f0 + (F_LUT[i + 1] - f0) * (s - i)
}

/** One sRGB8 colour → Lab, exact (no tables). Returns [L, a, b]. */
export function srgbToLab(r: number, g: number, b: number): [number, number, number] {
  const rl = GAMMA_LUT[r & 255]
  const gl = GAMMA_LUT[g & 255]
  const bl = GAMMA_LUT[b & 255]
  const fx = labF(MX0 * rl + MX1 * gl + MX2 * bl)
  const fy = labF(MY0 * rl + MY1 * gl + MY2 * bl)
  const fz = labF(MZ0 * rl + MZ1 * gl + MZ2 * bl)
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]
}

/** RGBA8 → planar Lab. Alpha is ignored. */
export function rgbaToLab(src: RGBAImage, out?: LabImage): LabImage {
  const { width, height, data } = src
  const dst = ensureLab(out, width, height)
  const L = dst.L
  const A = dst.a
  const B = dst.b
  const n = width * height
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const rl = GAMMA_LUT[data[p]]
    const gl = GAMMA_LUT[data[p + 1]]
    const bl = GAMMA_LUT[data[p + 2]]
    const fx = labFFast(MX0 * rl + MX1 * gl + MX2 * bl)
    const fy = labFFast(MY0 * rl + MY1 * gl + MY2 * bl)
    const fz = labFFast(MZ0 * rl + MZ1 * gl + MZ2 * bl)
    L[i] = 116 * fy - 16
    A[i] = 500 * (fx - fy)
    B[i] = 200 * (fy - fz)
  }
  return dst
}

/** RGBA8 → Rec.601 luma on the 0..255 scale (unrounded). Alpha is ignored. */
export function rgbaToGray(src: RGBAImage, out?: GrayImage): GrayImage {
  const { width, height, data } = src
  const dst = ensureGray(out, width, height)
  const g = dst.data
  const n = width * height
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    g[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]
  }
  return dst
}
