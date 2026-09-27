/**
 * What the scanner draws over the live camera: each tracked piece's outline, its answer as "C12 · R7",
 * and an arrow pointing to the piece's top as it lies on the table (turn the piece until the arrow
 * points away from you). Colours: green = locked and confident, amber = likely, white = still looking.
 * With a Region-finder area active, pieces that belong there glow blue and the rest fade.
 *
 * Everything is in camera-frame coordinates (FrameResult) mapped through coverTransform, the same
 * object-fit: cover mapping the <video> uses, so outlines sit on the pieces.
 */

import { useCallback } from 'react'
import { coverTransform } from '../../camera/coverTransform.ts'
import type { FrameResult, GridSpec } from '../../engine/types.ts'
import { COLORS, cellLabel } from '../picture/drawPicture.ts'
import { useCanvas } from '../useCanvas.ts'
import { trackLook, type TrackLook } from './trackLook.ts'

const LOOK_COLOR: Record<TrackLook, string> = {
  locked: COLORS.strong,
  likely: COLORS.likely,
  searching: 'rgba(255, 255, 255, 0.85)',
  region: COLORS.regionEdge,
  faded: 'rgba(255, 255, 255, 0.25)',
}

interface ScanOverlayProps {
  result: FrameResult | null
  grid: GridSpec
  regionActive: boolean
}

export function ScanOverlay({ result, grid, regionActive }: ScanOverlayProps) {
  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, size: { width: number; height: number }) => {
      if (result === null) return
      const { scale, offsetX, offsetY } = coverTransform(result.width, result.height, size.width, size.height)
      const sx = (x: number) => (x + 0.5) * scale + offsetX
      const sy = (y: number) => (y + 0.5) * scale + offsetY
      for (const t of result.tracks) {
        const look = trackLook(t, regionActive)
        const color = LOOK_COLOR[look]
        const step = Math.max(1, Math.floor(t.contour.length / 96))
        ctx.beginPath()
        for (let i = 0; i < t.contour.length; i += step) {
          const p = t.contour[i]
          if (i === 0) ctx.moveTo(sx(p.x), sy(p.y))
          else ctx.lineTo(sx(p.x), sy(p.y))
        }
        ctx.closePath()
        ctx.lineJoin = 'round'
        if (look === 'region') {
          ctx.shadowColor = COLORS.regionEdge
          ctx.shadowBlur = 16
        }
        ctx.lineWidth = look === 'locked' || look === 'region' ? 3.5 : 2
        ctx.setLineDash(look === 'searching' ? [6, 5] : [])
        ctx.strokeStyle = color
        ctx.stroke()
        ctx.shadowBlur = 0
        ctx.setLineDash([])
        if (look === 'faded' || t.corners === null) continue

        const cx = sx(t.corners.reduce((s, c) => s + c.x, 0) / 4)
        const cy = sy(t.corners.reduce((s, c) => s + c.y, 0) / 4)
        const side = (Math.hypot(t.corners[1].x - t.corners[0].x, t.corners[1].y - t.corners[0].y) * scale)
        const top = t.candidates[0]
        if (top !== undefined && look !== 'searching') {
          // Arrow to the piece's top (upAngleDeg: clockwise from screen-up).
          const a = (top.upAngleDeg * Math.PI) / 180
          const len = Math.max(18, side * 0.42)
          const ex = cx + Math.sin(a) * len
          const ey = cy - Math.cos(a) * len
          ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)'
          ctx.lineWidth = 6
          arrow(ctx, cx, cy, ex, ey)
          ctx.strokeStyle = color
          ctx.lineWidth = 3
          arrow(ctx, cx, cy, ex, ey)
        }
        const text = top !== undefined && look !== 'searching' ? cellLabel(top.cell, grid) : '…'
        pill(ctx, cx, cy + Math.max(14, side * 0.2), text, color)
      }
    },
    [result, grid, regionActive],
  )
  const ref = useCanvas(draw)
  return <canvas ref={ref} aria-hidden="true" className="pointer-events-none absolute inset-0 size-full" />
}

function arrow(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number): void {
  const a = Math.atan2(y1 - y0, x1 - x0)
  const head = 9
  ctx.beginPath()
  ctx.moveTo(x0, y0)
  ctx.lineTo(x1, y1)
  ctx.moveTo(x1 - head * Math.cos(a - 0.5), y1 - head * Math.sin(a - 0.5))
  ctx.lineTo(x1, y1)
  ctx.lineTo(x1 - head * Math.cos(a + 0.5), y1 - head * Math.sin(a + 0.5))
  ctx.lineCap = 'round'
  ctx.stroke()
}

function pill(ctx: CanvasRenderingContext2D, x: number, y: number, text: string, color: string): void {
  ctx.font = '700 14px system-ui, sans-serif'
  const w = ctx.measureText(text).width + 16
  const h = 24
  ctx.fillStyle = 'rgba(11, 16, 32, 0.82)'
  ctx.beginPath()
  ctx.roundRect(x - w / 2, y - h / 2, w, h, 12)
  ctx.fill()
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.stroke()
  ctx.fillStyle = color
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText(text, x, y + 0.5)
}
