/**
 * Overlay canvas over the live camera view, drawn in camera-frame coordinates.
 *
 * The canvas covers the same box as the <video> (which uses object-fit: cover) and draws through
 * coverTransform, the same frame → screen mapping later overlays (piece outlines, "Col 12 · Row 7"
 * labels) will use. Its backing store is sized in device pixels so lines stay sharp on a
 * 3× iPhone screen.
 *
 * For now it draws one reticle: a box exactly RETICLE_FRAME_PX camera pixels wide at the frame
 * centre. That makes the mapping visible (the box must be square and centred) and shows how large
 * something at the current distance is in camera pixels.
 */

import { useEffect, useRef, useState } from 'react'
import { coverTransform } from '../../camera/coverTransform.ts'

/**
 * Reticle size in camera pixels (design choice): the size of the full-resolution crop the plan
 * gives each piece for its precise outline.
 */
const RETICLE_FRAME_PX = 256

/** Same as --color-accent in src/index.css (canvas cannot read Tailwind theme values). */
const ACCENT = '#f5a524'

interface ViewSize {
  /** CSS px. */
  width: number
  height: number
  devicePixelRatio: number
}

interface FrameOverlayProps {
  frameWidth: number
  frameHeight: number
}

export function FrameOverlay({ frameWidth, frameHeight }: FrameOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [view, setView] = useState<ViewSize | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas === null) return
    const observer = new ResizeObserver(([entry]) => {
      setView({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
        devicePixelRatio: window.devicePixelRatio || 1,
      })
    })
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas === null || view === null) return
    canvas.width = Math.round(view.width * view.devicePixelRatio)
    canvas.height = Math.round(view.height * view.devicePixelRatio)
    const context = canvas.getContext('2d')
    if (context === null) return
    drawReticle(context, view, frameWidth, frameHeight)
  }, [view, frameWidth, frameHeight])

  return (
    <canvas ref={canvasRef} aria-hidden="true" className="pointer-events-none absolute inset-0 size-full" />
  )
}

function drawReticle(
  context: CanvasRenderingContext2D,
  view: ViewSize,
  frameWidth: number,
  frameHeight: number,
): void {
  const { scale, offsetX, offsetY } = coverTransform(frameWidth, frameHeight, view.width, view.height)
  // Draw in CSS px so line widths do not depend on the frame's scale.
  context.setTransform(view.devicePixelRatio, 0, 0, view.devicePixelRatio, 0, 0)
  context.clearRect(0, 0, view.width, view.height)

  const size = RETICLE_FRAME_PX * scale
  const left = (frameWidth - RETICLE_FRAME_PX) / 2 * scale + offsetX
  const top = (frameHeight - RETICLE_FRAME_PX) / 2 * scale + offsetY
  const arm = size * 0.2

  context.strokeStyle = ACCENT
  context.lineWidth = 2
  context.lineCap = 'round'
  context.beginPath()
  for (const [cx, cy, dx, dy] of [
    [left, top, 1, 1],
    [left + size, top, -1, 1],
    [left + size, top + size, -1, -1],
    [left, top + size, 1, -1],
  ]) {
    context.moveTo(cx + dx * arm, cy)
    context.lineTo(cx, cy)
    context.lineTo(cx, cy + dy * arm)
  }
  context.stroke()

  context.fillStyle = ACCENT
  context.font = '600 12px system-ui, sans-serif'
  context.textAlign = 'center'
  context.textBaseline = 'top'
  context.fillText(`${RETICLE_FRAME_PX} camera px`, left + size / 2, top + size + 8)
}
