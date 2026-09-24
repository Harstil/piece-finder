/**
 * Typed access to the OpenCV golden fixtures written by make_goldens.py.
 *
 * Test-only: the Vitest suites of src/engine/image and src/engine/geom import these helpers to decode
 * the JSON (masks as '0'/'1' row strings, images as flat row-major lists) into engine types, so each
 * suite compares the TypeScript primitive against what opencv-python produced for the same input.
 */

import type { GrayImage, Mask, Point, RGBAImage } from '../types.ts'
import approxJson from './approx.json'
import componentsJson from './components.json'
import contoursJson from './contours.json'
import distanceJson from './distance.json'
import filtersJson from './filters.json'
import homographyJson from './homography.json'
import labJson from './lab.json'
import masksJson from './masks.json'
import morphJson from './morph.json'
import resizeJson from './resize.json'
import warpJson from './warp.json'

export type MaskName = 'blobs' | 'holes' | 'random' | 'piece' | 'border'
export const MASK_NAMES: readonly MaskName[] = ['blobs', 'holes', 'random', 'piece', 'border']

interface MaskJson {
  width: number
  height: number
  rows: string[]
}

export function maskFromRows(rows: readonly string[]): Mask {
  const height = rows.length
  const width = height > 0 ? rows[0].length : 0
  const data = new Uint8Array(width * height)
  for (let y = 0; y < height; y++) {
    const row = rows[y]
    for (let x = 0; x < width; x++) data[y * width + x] = row.charCodeAt(x) === 49 ? 1 : 0
  }
  return { width, height, data }
}

export function maskToRows(mask: Mask): string[] {
  const rows: string[] = []
  for (let y = 0; y < mask.height; y++) {
    let row = ''
    for (let x = 0; x < mask.width; x++) row += mask.data[y * mask.width + x] !== 0 ? '1' : '0'
    rows.push(row)
  }
  return rows
}

export function goldenMask(name: MaskName): Mask {
  return maskFromRows((masksJson as Record<MaskName, MaskJson>)[name].rows)
}

export function grayFrom(width: number, height: number, data: readonly number[]): GrayImage {
  return { width, height, data: Float32Array.from(data) }
}

/** Interleaved RGB list → RGBA image with opaque alpha. */
export function rgbaFromRgb(width: number, height: number, rgb: readonly number[]): RGBAImage {
  const data = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = rgb[i * 3]
    data[i * 4 + 1] = rgb[i * 3 + 1]
    data[i * 4 + 2] = rgb[i * 3 + 2]
    data[i * 4 + 3] = 255
  }
  return { width, height, data }
}

/** Flat [x0, y0, x1, y1, ...] → Point[]. */
export function pointsFrom(flat: readonly number[]): Point[] {
  const pts: Point[] = []
  for (let i = 0; i + 1 < flat.length; i += 2) pts.push({ x: flat[i], y: flat[i + 1] })
  return pts
}

export interface ContourGolden {
  external: number[][]
  allOuter: number[][]
  externalOrientedArea: number[]
}

export interface ApproxCase {
  mask: string
  epsilon: number
  input: number[]
  output: number[]
}

export interface ComponentsGolden {
  count: number
  labels: number[]
  stats: number[][]
  centroids: number[][]
}

export interface MorphCase {
  mask: MaskName
  shape: 'rect' | 'ellipse'
  width: number
  height: number
  kernel: string[]
  erode: string[]
  dilate: string[]
  open: string[]
  close: string[]
}

interface SizedData {
  width: number
  height: number
  data: number[]
}

export interface WarpCase {
  dstToSrc: number[]
  width: number
  height: number
  out8: number[]
  outF: number[] | null
}

export const golden = {
  contours: contoursJson as Record<MaskName, ContourGolden>,
  approx: approxJson as { cases: ApproxCase[] },
  components: componentsJson as Record<MaskName, ComponentsGolden>,
  distance: distanceJson as Partial<Record<MaskName, number[]>>,
  morph: morphJson as { cases: MorphCase[] },
  lab: labJson as { rgb: number[]; lab: number[]; gray8: number[] },
  resize: resizeJson as {
    gray: SizedData
    area: SizedData[]
    rgb: SizedData
    areaRgb: SizedData[]
    linear: SizedData[]
  },
  filters: filtersJson as {
    image: SizedData
    blur: { sigma: number; radius: number; data: number[] }[]
    sobelDx: number[]
    sobelDy: number[]
  },
  homography: homographyJson as {
    quads: { src: number[]; dst: number[]; H: number[] }[]
    pointSets: { noise: number; src: number[]; dst: number[]; H: number[] }[]
    fit: { src: number[]; dst: number[]; affine: number[]; similarity: number[] }
  },
  warp: warpJson as { image: SizedData; warps: WarpCase[] },
}
