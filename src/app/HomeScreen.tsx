/**
 * Home: the current puzzle (a thumbnail of its straightened picture, grid and progress) with the way into
 * the scanner, or — before there is one — the way to set one up. The Camera check stays available as a
 * small link: it is how a phone's camera and speed are diagnosed.
 */

import { useCallback } from 'react'
import { BUILD_LABEL } from './buildInfo.ts'
import { drawPicture } from './picture/drawPicture.ts'
import type { PuzzleStatus, PuzzleView } from './usePuzzle.ts'
import { useCanvas } from './useCanvas.ts'

interface HomeScreenProps {
  status: PuzzleStatus
  onScan: () => void
  onNewPuzzle: () => void
  onCameraCheck: () => void
}

export function HomeScreen({ status, onScan, onNewPuzzle, onCameraCheck }: HomeScreenProps) {
  const hasPuzzle = status.kind === 'ready' || status.kind === 'preparing' || (status.kind === 'error' && status.record !== null)
  return (
    <main className="pt-safe pb-safe px-safe flex min-h-dvh flex-col">
      <div className="flex flex-1 flex-col items-center justify-center text-center">
        {status.kind === 'ready' ? (
          <Thumbnail view={status.view} />
        ) : (
          <img src="/icon.svg" alt="" width={96} height={96} className="mb-6 size-24" />
        )}
        <h1 className="text-3xl font-semibold tracking-tight text-white">Piece Finder</h1>
        {status.kind === 'ready' ? (
          <p className="mt-2 text-sm text-slate-300 tabular-nums">
            {status.view.record.grid.cols} × {status.view.record.grid.rows} pieces · {status.view.record.placed.length} placed
            {status.view.summary.cellSizePx < 48 && (
              <span className="mt-1 block text-amber-300">
                The box photo is small for this many pieces ({Math.round(status.view.summary.cellSizePx)} px per piece). A
                sharper or closer photo will find pieces more reliably.
              </span>
            )}
          </p>
        ) : (
          <p className="mt-3 max-w-xs text-base text-slate-300">
            Hold your phone over loose jigsaw pieces and see where each one goes and how to turn it.
          </p>
        )}
        {status.kind === 'preparing' && <p className="mt-3 text-sm text-slate-400">Preparing the picture…</p>}
        {status.kind === 'error' && <p className="mt-3 max-w-xs text-sm text-red-300">{status.message}</p>}

        {hasPuzzle ? (
          <>
            <button
              type="button"
              onClick={onScan}
              className="bg-accent text-ink-950 active:bg-accent-strong mt-8 w-full max-w-xs rounded-2xl px-6 py-4 text-lg font-semibold shadow-lg shadow-black/30"
            >
              Scan pieces
            </button>
            <button type="button" onClick={onNewPuzzle} className="mt-3 w-full max-w-xs rounded-2xl bg-ink-800 px-6 py-3 font-medium text-white active:bg-ink-700">
              New puzzle
            </button>
          </>
        ) : (
          <button
            type="button"
            onClick={onNewPuzzle}
            disabled={status.kind === 'loading'}
            className="bg-accent text-ink-950 active:bg-accent-strong mt-10 w-full max-w-xs rounded-2xl px-6 py-4 text-lg font-semibold shadow-lg shadow-black/30 disabled:opacity-40"
          >
            Set up a puzzle
          </button>
        )}
        <button type="button" onClick={onCameraCheck} className="mt-6 text-sm text-slate-400 underline underline-offset-4">
          Camera check
        </button>
      </div>
      <p className="text-center text-xs text-slate-500 tabular-nums">{BUILD_LABEL}</p>
    </main>
  )
}

function Thumbnail({ view }: { view: PuzzleView }) {
  const { picture, record } = view
  const k = 240 / Math.max(picture.width, picture.height)
  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, size: { width: number; height: number }) =>
      void drawPicture(ctx, picture, 0, 0, size.width, size.height, { grid: record.grid, placed: record.placed }),
    [picture, record],
  )
  const ref = useCanvas(draw)
  return (
    <canvas
      ref={ref}
      style={{ width: Math.round(picture.width * k), height: Math.round(picture.height * k) }}
      className="mb-6 rounded-2xl shadow-xl shadow-black/40"
      aria-label="Your puzzle"
    />
  )
}
