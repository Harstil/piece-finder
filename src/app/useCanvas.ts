/**
 * A canvas that is always sized to its CSS box in device pixels, redrawn whenever its size or the
 * caller's `draw` function changes. `draw` receives a context already scaled to CSS px, so callers
 * think in layout pixels and lines stay sharp on a 3× iPhone screen.
 */

import { useEffect, useRef, useState, type RefObject } from 'react'

export interface CanvasSize {
  width: number
  height: number
}

export function useCanvas(
  draw: ((ctx: CanvasRenderingContext2D, size: CanvasSize) => void) | null,
): RefObject<HTMLCanvasElement | null> {
  const ref = useRef<HTMLCanvasElement>(null)
  const [size, setSize] = useState<CanvasSize | null>(null)

  useEffect(() => {
    const canvas = ref.current
    if (canvas === null) return
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const canvas = ref.current
    if (canvas === null || size === null || draw === null) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.max(1, Math.round(size.width * dpr))
    canvas.height = Math.max(1, Math.round(size.height * dpr))
    const ctx = canvas.getContext('2d')
    if (ctx === null) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size.width, size.height)
    draw(ctx, size)
  }, [draw, size])

  return ref
}
