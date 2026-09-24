/**
 * Public API of the image primitives (pure TypeScript on typed arrays; runs in the worker and in Node).
 * Everything takes an optional preallocated output so per-frame code can run without allocating.
 */

export { gaussianBlur, gaussianKernel } from './blur.ts'
export { componentMask, labelComponents, markOuterBackground } from './components.ts'
export type { ComponentStats, Components } from './components.ts'
export { createGray, createLab, createMask, createRGBA } from './create.ts'
export { distanceTransform } from './distance.ts'
export { gradientMagnitude, gradientOrientation, sobel } from './gradient.ts'
export type { Gradients } from './gradient.ts'
export { integralImage, windowArea, windowMean, windowSum, windowVariance } from './integral.ts'
export type { IntegralImage } from './integral.ts'
export { rgbaToGray, rgbaToLab, srgbToLab } from './lab.ts'
export { close, dilate, ellipseKernel, ellipseRuns, erode, fillHoles, open, rectKernel } from './morph.ts'
export type { KernelShape, StructuringElement } from './morph.ts'
export {
  fitSize,
  resizeAreaGray,
  resizeAreaLab,
  resizeAreaRGBA,
  resizeBilinearGray,
  resizeBilinearLab,
  resizeBilinearRGBA,
  resizeMaskNearest,
} from './resize.ts'
