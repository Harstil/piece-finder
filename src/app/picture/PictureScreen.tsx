/**
 * The whole box picture with the grid: progress (placed cells shaded green) and the Region finder.
 *
 * Region finder: switch on "Select area" and drag across the picture; the cells under the drag become the
 * area, and back in the scanner every piece whose likely places fall inside it glows — the quickest way
 * to pull "the red boat" out of a pile. Outside selection mode, tapping a placed cell un-places it (for
 * when a piece was marked by mistake).
 */

import { useCallback, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import type { PuzzleControls, PuzzleView } from '../usePuzzle.ts'
import { useCanvas } from '../useCanvas.ts'
import { drawPicture } from './drawPicture.ts'

interface PictureScreenProps {
  view: PuzzleView
  puzzle: PuzzleControls
  onBack: () => void
}

interface Layout {
  x: number
  y: number
  w: number
  h: number
}

export function PictureScreen({ view, puzzle, onBack }: PictureScreenProps) {
  const { picture, record } = view
  const { cols, rows } = record.grid
  const [selecting, setSelecting] = useState(false)
  const [drag, setDrag] = useState<{ c0: number; r0: number; c1: number; r1: number } | null>(null)
  const layoutRef = useRef<Layout | null>(null)

  const dragCells = useCallback((): Set<number> | null => {
    if (drag === null) return null
    const cells = new Set<number>()
    for (let r = Math.min(drag.r0, drag.r1); r <= Math.max(drag.r0, drag.r1); r++) {
      for (let c = Math.min(drag.c0, drag.c1); c <= Math.max(drag.c0, drag.c1); c++) cells.add(r * cols + c)
    }
    return cells
  }, [drag, cols])

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, size: { width: number; height: number }) => {
      const k = Math.min(size.width / picture.width, size.height / picture.height)
      const w = picture.width * k
      const h = picture.height * k
      const layout = { x: (size.width - w) / 2, y: (size.height - h) / 2, w, h }
      layoutRef.current = layout
      drawPicture(ctx, picture, layout.x, layout.y, w, h, {
        grid: record.grid,
        placed: record.placed,
        region: dragCells() ?? puzzle.region,
        gridLines: true,
      })
    },
    [picture, record, puzzle.region, dragCells],
  )
  const canvasRef = useCanvas(draw)

  const cellAt = (e: ReactPointerEvent): { c: number; r: number } | null => {
    const layout = layoutRef.current
    const canvas = canvasRef.current
    if (layout === null || canvas === null) return null
    const rect = canvas.getBoundingClientRect()
    const u = (e.clientX - rect.left - layout.x) / layout.w
    const v = (e.clientY - rect.top - layout.y) / layout.h
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return null
    return { c: Math.floor(u * cols), r: Math.floor(v * rows) }
  }

  const onPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const at = cellAt(e)
    if (at === null) return
    if (!selecting) {
      const cell = at.r * cols + at.c
      if (record.placed.includes(cell)) puzzle.setPlaced(record.placed.filter((x) => x !== cell))
      return
    }
    e.currentTarget.setPointerCapture(e.pointerId)
    setDrag({ c0: at.c, r0: at.r, c1: at.c, r1: at.r })
  }
  const onPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (drag === null) return
    const at = cellAt(e)
    if (at !== null) setDrag({ ...drag, c1: at.c, r1: at.r })
  }
  const onPointerUp = () => {
    const cells = dragCells()
    setDrag(null)
    if (cells !== null) {
      puzzle.setRegion(cells)
      setSelecting(false)
    }
  }

  const total = cols * rows
  return (
    <main className="pt-safe pb-safe flex h-dvh flex-col bg-ink-950">
      <header className="px-safe flex items-center gap-3 pb-3">
        <button type="button" onClick={onBack} className="rounded-full bg-ink-800 px-4 py-2 text-sm font-medium text-white active:bg-ink-700">
          ‹ Scan
        </button>
        <h1 className="text-base font-semibold text-white">Picture</h1>
        <span className="ml-auto text-sm text-slate-400 tabular-nums">
          {record.placed.length} / {total} placed
        </span>
      </header>
      <div className="relative min-h-0 flex-1 bg-black">
        <canvas
          ref={canvasRef}
          className="absolute inset-0 size-full touch-none"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={() => setDrag(null)}
        />
      </div>
      <section className="px-safe space-y-3 pt-4">
        <p className="text-sm text-slate-300">
          {selecting
            ? 'Drag across the part of the picture you are building. Pieces that belong there will glow in the scanner.'
            : 'Green cells are placed (tap one to undo). Use the Region finder to find the pieces for one area.'}
        </p>
        <div className="flex gap-3">
          {puzzle.region !== null && !selecting && (
            <button type="button" onClick={() => puzzle.setRegion(null)} className="flex-1 rounded-2xl bg-ink-800 px-4 py-3 font-medium text-white active:bg-ink-700">
              Clear area
            </button>
          )}
          <button
            type="button"
            onClick={() => setSelecting(!selecting)}
            className={`flex-1 rounded-2xl px-4 py-3 font-semibold ${selecting ? 'bg-ink-800 text-white' : 'bg-sky-500 text-ink-950'}`}
          >
            {selecting ? 'Cancel' : 'Select area'}
          </button>
          <button type="button" onClick={onBack} className="bg-accent text-ink-950 active:bg-accent-strong flex-1 rounded-2xl px-4 py-3 font-semibold">
            Scan
          </button>
        </div>
      </section>
    </main>
  )
}
