/**
 * Maps camera-frame pixels to screen pixels for a video shown with `object-fit: cover`.
 *
 * The live view fills the phone screen, so the frame is scaled up until it covers the element and
 * the overflow is cropped equally on both sides (the default `object-position: 50% 50%`). Every
 * overlay (outlines, labels, the reticle on the Camera check screen) is drawn in frame
 * coordinates — the coordinates the worker reports, see src/engine/types.ts — and goes through
 * this one transform, so it lines up with the pixels underneath.
 */

export interface CoverTransform {
  /** Screen px per frame px (the same on both axes: cover never distorts). */
  scale: number
  /**
   * Screen position of the frame's top-left *edge*; negative on the axis that is cropped. Engine
   * points use pixel-centre coordinates (types.ts: pixel (0, 0) is centred at (0, 0), so the frame
   * edge is at −0.5), so an engine point p maps to ((p.x + 0.5) · scale + offsetX, (p.y + 0.5) · scale + offsetY).
   */
  offsetX: number
  offsetY: number
}

export function coverTransform(
  frameWidth: number,
  frameHeight: number,
  viewWidth: number,
  viewHeight: number,
): CoverTransform {
  const scale = Math.max(viewWidth / frameWidth, viewHeight / frameHeight)
  return {
    scale,
    offsetX: (viewWidth - frameWidth * scale) / 2,
    offsetY: (viewHeight - frameHeight * scale) / 2,
  }
}
