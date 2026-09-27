/**
 * The piece sheet: what the user sees after tapping a piece in the scanner.
 *
 * It shows the box picture zoomed around the best cell (outlined in the verdict colour) with the other
 * candidates numbered, how to turn the piece, and the candidate list with probabilities — when the
 * engine is unsure, the right answer is usually among the first few, and the zoomed picture lets the
 * user decide at a glance. "Mark as placed" removes that cell from all future guesses.
 */

import { useCallback, useState, type ReactNode } from 'react'
import type { GridSpec, TrackState } from '../../engine/types.ts'
import { COLORS, cellLabel, drawPicture, viewAround } from '../picture/drawPicture.ts'
import { useCanvas } from '../useCanvas.ts'

/** Cells shown around the target in the zoomed picture (radius). */
const ZOOM_RADIUS = 3

interface PieceSheetProps {
  track: TrackState
  picture: ImageBitmap
  grid: GridSpec
  placed: readonly number[]
  onPlace: (cell: number) => void
  onClose: () => void
}

function turnText(upAngleDeg: number): string {
  let a = upAngleDeg % 360
  if (a > 180) a -= 360
  if (a <= -180) a += 360
  const quarter = Math.round(a / 90)
  if (Math.abs(a) < 25) return 'It already lies the right way up.'
  if (quarter === 2 || quarter === -2) return 'Turn it upside down (half a turn).'
  return quarter > 0 || (quarter === 0 && a > 0)
    ? `Turn it ${Math.round(Math.abs(a) / 5) * 5}° counter-clockwise.`
    : `Turn it ${Math.round(Math.abs(a) / 5) * 5}° clockwise.`
}

export function PieceSheet({ track, picture, grid, placed, onPlace, onClose }: PieceSheetProps) {
  const [selected, setSelected] = useState(0)
  const candidates = track.candidates
  const target = candidates[Math.min(selected, candidates.length - 1)]

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, size: { width: number; height: number }) => {
      if (target === undefined) return
      const view = viewAround(target.cell, grid, ZOOM_RADIUS)
      const vw = (view.col1 - view.col0) * (picture.width / grid.cols)
      const vh = (view.row1 - view.row0) * (picture.height / grid.rows)
      const k = Math.min(size.width / vw, size.height / vh)
      const dw = vw * k
      const dh = vh * k
      drawPicture(ctx, picture, (size.width - dw) / 2, (size.height - dh) / 2, dw, dh, {
        grid,
        view,
        placed,
        gridLines: true,
        highlights: candidates.map((c, i) => ({
          cell: c.cell,
          color: i === selected ? COLORS.accent : 'rgba(255,255,255,0.85)',
          width: i === selected ? 4 : 1.5,
          label: i === selected ? undefined : String(i + 1),
        })),
      })
    },
    [target, candidates, selected, picture, grid, placed],
  )
  const ref = useCanvas(draw)

  if (target === undefined) {
    return (
      <Sheet onClose={onClose}>
        <p className="text-slate-300">Still looking at this piece — hold the phone steady over it for a moment.</p>
      </Sheet>
    )
  }
  return (
    <Sheet onClose={onClose}>
      <div className="flex items-baseline gap-2">
        <h2 className="text-xl font-semibold text-white">
          Column {target.col + 1}, row {target.row + 1}
        </h2>
        <span className="text-sm text-slate-400 tabular-nums">{Math.round(target.prob * 100)}%</span>
        {track.locked && selected === 0 && (
          <span className="ml-auto rounded-full bg-green-500/20 px-2 py-0.5 text-xs font-medium text-green-300">confident</span>
        )}
      </div>
      <p className="text-sm text-slate-300">{turnText(target.upAngleDeg)} The arrow on the piece points to its top.</p>
      <canvas ref={ref} className="h-56 w-full rounded-xl bg-black" aria-label="The picture around the matching spot" />
      <ol className="flex flex-wrap gap-2">
        {candidates.map((c, i) => (
          <li key={c.cell}>
            <button
              type="button"
              onClick={() => setSelected(i)}
              className={`rounded-xl px-3 py-1.5 text-sm tabular-nums ${i === selected ? 'bg-accent text-ink-950 font-semibold' : 'bg-ink-800 text-slate-200'}`}
            >
              {i + 1}. {cellLabel(c.cell, grid)} · {Math.round(c.prob * 100)}%
            </button>
          </li>
        ))}
      </ol>
      <div className="flex gap-3 pt-1">
        <button type="button" onClick={onClose} className="flex-1 rounded-2xl bg-ink-800 px-4 py-3 font-medium text-white active:bg-ink-700">
          Close
        </button>
        <button
          type="button"
          onClick={() => onPlace(target.cell)}
          className="bg-accent text-ink-950 active:bg-accent-strong flex-1 rounded-2xl px-4 py-3 font-semibold"
        >
          Mark as placed
        </button>
      </div>
    </Sheet>
  )
}

function Sheet({ children, onClose }: { children: ReactNode; onClose: () => void }) {
  return (
    <div className="absolute inset-0 z-20 flex items-end bg-black/40" onClick={onClose}>
      <section
        className="pb-safe px-safe max-h-[85dvh] w-full space-y-3 overflow-y-auto rounded-t-3xl bg-ink-900 pt-4 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </section>
    </div>
  )
}
