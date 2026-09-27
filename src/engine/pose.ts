/**
 * Piece pose: which way a piece's motif-up points in the camera frame, given its 4 core corners and a
 * matched rotation. This is what the rotate-arrow in the app shows.
 *
 * Definitions (types.ts): canonical corner i = corners[i] maps to the canonical square's TL, TR, BR, BL.
 * Rotation r means turning the canonical patch clockwise by r × 90° makes it upright in the motif, so in
 * canonical coordinates the motif-up vector is (0, −1) turned COUNTER-clockwise by r × 90°. It is mapped
 * into the frame at the core centre through the canonical → frame homography, and the result is the
 * angle clockwise from image-up (0 = upright, 90 = the piece's top points right).
 */

import { applyHomography, homographyFromQuad, type Homography } from './geom/index.ts'
import type { Point, Quad, Rotation } from './types.ts'

const UNIT: Quad = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
]

/** Step along the up vector, in core units: small enough that perspective is locally linear. */
const STEP = 0.01

/** Homography from the unit core square (TL, TR, BR, BL = (0,0) … (0,1)) to the frame corners. */
export function unitToFrame(corners: Quad): Homography | null {
  return homographyFromQuad(UNIT, corners)
}

/** Motif-up in canonical (unit square) coordinates for rotation r: (0,−1) turned CCW r quarter turns. */
export function canonicalUp(r: Rotation): Point {
  switch (r) {
    case 1:
      return { x: -1, y: 0 }
    case 2:
      return { x: 0, y: 1 }
    case 3:
      return { x: 1, y: 0 }
    default:
      return { x: 0, y: -1 }
  }
}

/** Direction of the piece's motif-up in the frame, degrees clockwise from image-up, in [0, 360). */
export function upAngleDeg(corners: Quad, rotation: Rotation): number {
  const H = unitToFrame(corners)
  if (H === null) return NaN
  const u = canonicalUp(rotation)
  const p0 = applyHomography(H, { x: 0.5, y: 0.5 })
  const p1 = applyHomography(H, { x: 0.5 + STEP * u.x, y: 0.5 + STEP * u.y })
  const deg = (Math.atan2(p1.x - p0.x, -(p1.y - p0.y)) * 180) / Math.PI
  return (deg + 360) % 360
}
