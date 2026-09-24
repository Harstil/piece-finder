/**
 * Public API of the geometry primitives: homographies and affine fits, perspective warps, contour
 * tracing, polygon utilities and quarter-turn rotation. Orientation conventions are in each module's
 * header and in ../types.ts (image coordinates, y down, contours clockwise on screen).
 */

export { findContours, traceComponent, traceOuterBoundary } from './contour.ts'
export type { ContourOptions } from './contour.ts'
export {
  affineToHomography,
  applyAffine,
  applyHomography,
  composeHomography,
  estimateAffine,
  estimateSimilarity,
  homographyFromPoints,
  homographyFromQuad,
  identityHomography,
  invertHomography,
  transformPoints,
} from './homography.ts'
export type { Affine, Homography, Similarity } from './homography.ts'
export {
  approxPolyDP,
  convexHull,
  isClockwise,
  orientClockwise,
  perimeter,
  pointInPolygon,
  pointSegmentDistance,
  rasterizePolygon,
  resampleClosed,
  signedArea,
  turningAngles,
} from './polygon.ts'
export { rotatedSize, rotateGray, rotateLab, rotateMask, rotatePoint } from './rotate.ts'
export { rectToQuadHomography, warpGray, warpLab, warpMask, warpRGBA } from './warp.ts'
export type { WarpOptions } from './warp.ts'
