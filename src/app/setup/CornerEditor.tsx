/**
 * The box photo with the picture's 4 corners as draggable handles, and optionally the puzzle grid drawn
 * inside them (in perspective, through the same corner → unit-square homography the engine uses).
 *
 * Dragging is relative — the handle keeps its offset from the finger — so the finger never hides the
 * point being placed, and a loupe in the opposite top corner shows the magnified photo under the handle
 * with a crosshair. The engine needs these corners to within a fraction of a cell, which a 12 MP photo
 * shown on a phone screen can only reach with magnification.
 */

import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { applyHomography } from '../../engine/geom/index.ts'
import { unitToFrame } from '../../engine/pose.ts'
import type { GridSpec, Point, Quad } from '../../engine/types.ts'

/** Screen px within which a touch grabs a handle. Guessed: a fingertip. */
const GRAB_RADIUS = 44
const HANDLE_RADIUS = 11
const LOUPE_SIZE = 132
/** Loupe magnification relative to the photo as displayed. */
const LOUPE_ZOOM = 4

interface CornerEditorProps {
  imageUrl: string
  imageWidth: number
  imageHeight: number
  corners: Quad
  onChange?: (corners: Quad) => void
  grid?: GridSpec | null
}

interface Layout {
  scale: number
  left: number
  top: number
}

export function CornerEditor({ imageUrl, imageWidth, imageHeight, corners, onChange, grid }: CornerEditorProps) {
  const boxRef = useRef<HTMLDivElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const loupeRef = useRef<HTMLCanvasElement>(null)
  const [box, setBox] = useState<{ width: number; height: number } | null>(null)
  const [drag, setDrag] = useState<{ index: number; pointerId: number; dx: number; dy: number } | null>(null)

  useEffect(() => {
    const el = boxRef.current
    if (el === null) return
    const observer = new ResizeObserver(([entry]) => setBox({ width: entry.contentRect.width, height: entry.contentRect.height }))
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const layout: Layout | null = useMemo(() => {
    if (box === null) return null
    const scale = Math.min(box.width / imageWidth, box.height / imageHeight)
    return { scale, left: (box.width - imageWidth * scale) / 2, top: (box.height - imageHeight * scale) / 2 }
  }, [box, imageWidth, imageHeight])

  const gridLines = useMemo(() => {
    if (grid === null || grid === undefined) return []
    const H = unitToFrame(corners)
    if (H === null) return []
    const lines: [Point, Point][] = []
    for (let c = 1; c < grid.cols; c++) lines.push([applyHomography(H, { x: c / grid.cols, y: 0 }), applyHomography(H, { x: c / grid.cols, y: 1 })])
    for (let r = 1; r < grid.rows; r++) lines.push([applyHomography(H, { x: 0, y: r / grid.rows }), applyHomography(H, { x: 1, y: r / grid.rows })])
    return lines
  }, [grid, corners])

  // Loupe: redraw while dragging.
  useEffect(() => {
    const canvas = loupeRef.current
    const img = imgRef.current
    if (drag === null || canvas === null || img === null || layout === null) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = LOUPE_SIZE * dpr
    canvas.height = LOUPE_SIZE * dpr
    const ctx = canvas.getContext('2d')
    if (ctx === null) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const p = corners[drag.index]
    const span = LOUPE_SIZE / (layout.scale * LOUPE_ZOOM)
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, LOUPE_SIZE, LOUPE_SIZE)
    ctx.drawImage(img, p.x + 0.5 - span / 2, p.y + 0.5 - span / 2, span, span, 0, 0, LOUPE_SIZE, LOUPE_SIZE)
    const k = LOUPE_SIZE / span
    ctx.strokeStyle = '#f5a524'
    ctx.lineWidth = 2
    ctx.beginPath()
    for (const q of [corners[(drag.index + 1) % 4], corners[(drag.index + 3) % 4]]) {
      ctx.moveTo(LOUPE_SIZE / 2, LOUPE_SIZE / 2)
      ctx.lineTo(LOUPE_SIZE / 2 + (q.x - p.x) * k, LOUPE_SIZE / 2 + (q.y - p.y) * k)
    }
    ctx.stroke()
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(LOUPE_SIZE / 2, 0)
    ctx.lineTo(LOUPE_SIZE / 2, LOUPE_SIZE)
    ctx.moveTo(0, LOUPE_SIZE / 2)
    ctx.lineTo(LOUPE_SIZE, LOUPE_SIZE / 2)
    ctx.stroke()
  }, [drag, corners, layout])

  const toImage = (e: ReactPointerEvent): Point | null => {
    const el = boxRef.current
    if (el === null || layout === null) return null
    const rect = el.getBoundingClientRect()
    return { x: (e.clientX - rect.left - layout.left) / layout.scale - 0.5, y: (e.clientY - rect.top - layout.top) / layout.scale - 0.5 }
  }

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (onChange === undefined || layout === null) return
    const p = toImage(e)
    if (p === null) return
    let best = -1
    let bestD = GRAB_RADIUS / layout.scale
    corners.forEach((c, i) => {
      const d = Math.hypot(c.x - p.x, c.y - p.y)
      if (d < bestD) {
        bestD = d
        best = i
      }
    })
    if (best < 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    setDrag({ index: best, pointerId: e.pointerId, dx: corners[best].x - p.x, dy: corners[best].y - p.y })
  }

  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (drag === null || e.pointerId !== drag.pointerId || onChange === undefined) return
    const p = toImage(e)
    if (p === null) return
    const next = [...corners] as Quad
    next[drag.index] = {
      x: Math.max(-0.5, Math.min(imageWidth - 0.5, p.x + drag.dx)),
      y: Math.max(-0.5, Math.min(imageHeight - 0.5, p.y + drag.dy)),
    }
    onChange(next)
  }

  const endDrag = () => setDrag(null)
  const loupeLeft = drag !== null && layout !== null && corners[drag.index].x * layout.scale + layout.left < (box?.width ?? 0) / 2

  return (
    <div
      ref={boxRef}
      className="relative size-full touch-none select-none"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      {layout !== null && (
        <>
          <img
            ref={imgRef}
            src={imageUrl}
            alt="Box picture"
            draggable={false}
            className="absolute max-w-none"
            style={{ left: layout.left, top: layout.top, width: imageWidth * layout.scale, height: imageHeight * layout.scale }}
          />
          <svg
            className="pointer-events-none absolute overflow-visible"
            style={{ left: layout.left, top: layout.top, width: imageWidth * layout.scale, height: imageHeight * layout.scale }}
            viewBox={`-0.5 -0.5 ${imageWidth} ${imageHeight}`}
          >
            {gridLines.map(([a, b], i) => (
              <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="rgba(255,255,255,0.75)" strokeWidth={1 / layout.scale} />
            ))}
            <polygon
              points={corners.map((c) => `${c.x},${c.y}`).join(' ')}
              fill={gridLines.length > 0 ? 'none' : 'rgba(245,165,36,0.12)'}
              stroke="#f5a524"
              strokeWidth={2 / layout.scale}
            />
            {onChange !== undefined &&
              corners.map((c, i) => (
                <circle
                  key={i}
                  cx={c.x}
                  cy={c.y}
                  r={HANDLE_RADIUS / layout.scale}
                  fill={drag?.index === i ? 'rgba(245,165,36,0.35)' : 'rgba(0,0,0,0.35)'}
                  stroke="#f5a524"
                  strokeWidth={2.5 / layout.scale}
                />
              ))}
          </svg>
        </>
      )}
      {drag !== null && (
        <canvas
          ref={loupeRef}
          aria-hidden="true"
          className={`pointer-events-none absolute top-2 rounded-full border-2 border-white shadow-xl ${loupeLeft ? 'right-2' : 'left-2'}`}
          style={{ width: LOUPE_SIZE, height: LOUPE_SIZE }}
        />
      )}
    </div>
  )
}
