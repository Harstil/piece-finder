/**
 * New puzzle, in three steps:
 *   1. the box picture — photographed now (the phone's full-resolution camera, not a video frame: the
 *      engine wants ≥ 48 photo px per cell) or picked from the library (a sharp image from the maker's
 *      website is best);
 *   2. its 4 corners — the picture's own corners, not the box edge (CornerEditor);
 *   3. the piece count and grid — suggestions from the count and the picture's shape, editable, drawn
 *      over the picture so the user can check it lines up.
 * "Start scanning" hands everything to the app, which builds the engine's reference and opens the scanner.
 */

import { useEffect, useMemo, useState, type ChangeEvent } from 'react'
import type { GridSpec, Point, Quad } from '../../engine/types.ts'
import { CornerEditor } from './CornerEditor.tsx'
import { gridOptions } from './gridOptions.ts'

/** Initial corner inset for a fresh camera photo (the lid never fills the frame exactly). Guessed. */
const PHOTO_INSET = 0.08
const PIECE_PRESETS = [100, 300, 500, 1000, 1500, 2000]

interface Photo {
  blob: Blob
  url: string
  width: number
  height: number
}

interface SetupScreenProps {
  onCancel: () => void
  onStart: (photo: Blob, corners: Quad, grid: GridSpec) => void
}

type Step = 'photo' | 'corners' | 'grid'

function insetCorners(w: number, h: number, inset: number): Quad {
  const x0 = -0.5 + inset * w
  const y0 = -0.5 + inset * h
  const x1 = w - 0.5 - inset * w
  const y1 = h - 0.5 - inset * h
  return [
    { x: x0, y: y0 },
    { x: x1, y: y0 },
    { x: x1, y: y1 },
    { x: x0, y: y1 },
  ]
}

function dist(a: Point, b: Point): number {
  return Math.hypot(b.x - a.x, b.y - a.y)
}

export function SetupScreen({ onCancel, onStart }: SetupScreenProps) {
  const [step, setStep] = useState<Step>('photo')
  const [photo, setPhoto] = useState<Photo | null>(null)
  const [corners, setCorners] = useState<Quad | null>(null)
  const [pieces, setPieces] = useState(500)
  const [grid, setGrid] = useState<GridSpec | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => () => {
    if (photo !== null) URL.revokeObjectURL(photo.url)
  }, [photo])

  const aspect = useMemo(() => {
    if (corners === null) return 1
    const w = (dist(corners[0], corners[1]) + dist(corners[3], corners[2])) / 2
    const h = (dist(corners[0], corners[3]) + dist(corners[1], corners[2])) / 2
    return h > 0 ? w / h : 1
  }, [corners])
  const options = useMemo(() => gridOptions(pieces, aspect), [pieces, aspect])
  const chosen = grid ?? options[0] ?? null

  const onFile = (fromCamera: boolean) => async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (file === undefined) return
    setError(null)
    const url = URL.createObjectURL(file)
    try {
      const img = new Image()
      img.src = url
      await img.decode()
      const next = { blob: file, url, width: img.naturalWidth, height: img.naturalHeight }
      setPhoto(next)
      setCorners(insetCorners(next.width, next.height, fromCamera ? PHOTO_INSET : 0))
      setGrid(null)
      setStep('corners')
    } catch {
      URL.revokeObjectURL(url)
      setError("That file couldn't be opened as a picture. Try a JPEG or PNG.")
    }
  }

  return (
    <main className="pt-safe pb-safe flex h-dvh flex-col bg-ink-950">
      <header className="px-safe flex items-center gap-3 pb-3">
        <button
          type="button"
          onClick={() => (step === 'photo' ? onCancel() : setStep(step === 'grid' ? 'corners' : 'photo'))}
          className="rounded-full bg-ink-800 px-4 py-2 text-sm font-medium text-white active:bg-ink-700"
        >
          ‹ Back
        </button>
        <h1 className="text-base font-semibold text-white">New puzzle</h1>
        <span className="ml-auto text-sm text-slate-400 tabular-nums">
          {step === 'photo' ? '1' : step === 'corners' ? '2' : '3'} / 3
        </span>
      </header>

      {step === 'photo' && (
        <section className="px-safe flex flex-1 flex-col justify-center gap-4 text-center">
          <h2 className="text-2xl font-semibold text-white">The picture on the box</h2>
          <p className="mx-auto max-w-sm text-slate-300">
            Photograph the box lid straight on, in even light, tilting it slightly so there is no glare. Or pick a
            picture you already have — a sharp image from the puzzle maker's website works best.
          </p>
          <label className="bg-accent text-ink-950 active:bg-accent-strong mx-auto mt-4 w-full max-w-xs cursor-pointer rounded-2xl px-6 py-4 text-lg font-semibold">
            Photograph the box
            <input type="file" accept="image/*" capture="environment" className="sr-only" onChange={onFile(true)} />
          </label>
          <label className="mx-auto w-full max-w-xs cursor-pointer rounded-2xl bg-ink-800 px-6 py-4 text-lg font-semibold text-white active:bg-ink-700">
            Choose a picture
            <input type="file" accept="image/*" className="sr-only" onChange={onFile(false)} />
          </label>
          {error !== null && <p className="text-sm text-red-300">{error}</p>}
        </section>
      )}

      {step !== 'photo' && photo !== null && corners !== null && (
        <>
          <div className="relative min-h-0 flex-1 bg-black">
            <CornerEditor
              imageUrl={photo.url}
              imageWidth={photo.width}
              imageHeight={photo.height}
              corners={corners}
              onChange={step === 'corners' ? setCorners : undefined}
              grid={step === 'grid' ? chosen : null}
            />
          </div>
          {step === 'corners' ? (
            <section className="px-safe space-y-3 pt-4">
              <p className="text-sm text-slate-300">
                Drag the four handles onto the corners of the <strong className="text-white">picture</strong> — not the
                edge of the box. If a logo covers a corner, put the handle where the picture's edges would meet. Be
                precise: the magnifier helps.
              </p>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => setCorners(insetCorners(photo.width, photo.height, 0))}
                  className="flex-1 rounded-2xl bg-ink-800 px-4 py-3 font-medium text-white active:bg-ink-700"
                >
                  Whole image
                </button>
                <button
                  type="button"
                  onClick={() => setStep('grid')}
                  className="bg-accent text-ink-950 active:bg-accent-strong flex-1 rounded-2xl px-4 py-3 font-semibold"
                >
                  Next
                </button>
              </div>
            </section>
          ) : (
            <GridStep
              pieces={pieces}
              onPieces={(n) => {
                setPieces(n)
                setGrid(null)
              }}
              options={options}
              chosen={chosen}
              onGrid={setGrid}
              onStart={() => chosen !== null && onStart(photo.blob, corners, { cols: chosen.cols, rows: chosen.rows })}
            />
          )}
        </>
      )}
    </main>
  )
}

interface GridStepProps {
  pieces: number
  onPieces: (n: number) => void
  options: GridSpec[]
  chosen: GridSpec | null
  onGrid: (grid: GridSpec) => void
  onStart: () => void
}

function GridStep({ pieces, onPieces, options, chosen, onGrid, onStart }: GridStepProps) {
  const setCols = (cols: number) => chosen !== null && onGrid({ cols: Math.max(2, cols), rows: chosen.rows })
  const setRows = (rows: number) => chosen !== null && onGrid({ cols: chosen.cols, rows: Math.max(2, rows) })
  return (
    <section className="px-safe space-y-3 pt-4">
      <div className="flex items-center gap-3">
        <label htmlFor="pieces" className="text-sm text-slate-300">
          Pieces on the box
        </label>
        <input
          id="pieces"
          type="number"
          inputMode="numeric"
          min={4}
          value={pieces}
          onChange={(e) => onPieces(Math.max(0, Number(e.target.value) || 0))}
          className="w-24 rounded-xl bg-ink-800 px-3 py-2 text-white tabular-nums"
        />
        <div className="flex flex-1 gap-1 overflow-x-auto">
          {PIECE_PRESETS.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => onPieces(n)}
              className={`rounded-full px-2.5 py-1 text-xs tabular-nums ${n === pieces ? 'bg-accent text-ink-950' : 'bg-ink-800 text-slate-300'}`}
            >
              {n}
            </button>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        {options.map((o) => {
          const on = chosen !== null && o.cols === chosen.cols && o.rows === chosen.rows
          return (
            <button
              key={`${o.cols}x${o.rows}`}
              type="button"
              onClick={() => onGrid(o)}
              className={`rounded-xl px-3 py-2 text-sm tabular-nums ${on ? 'bg-accent text-ink-950 font-semibold' : 'bg-ink-800 text-slate-200'}`}
            >
              {o.cols} × {o.rows}
            </button>
          )
        })}
        {options.length === 0 && <p className="text-sm text-slate-400">Enter the piece count to see grid suggestions.</p>}
      </div>
      {chosen !== null && (
        <div className="flex items-center gap-2 text-sm text-slate-300">
          <span>Columns</span>
          <Stepper value={chosen.cols} onChange={setCols} />
          <span className="ml-2">Rows</span>
          <Stepper value={chosen.rows} onChange={setRows} />
          <span className="ml-auto text-slate-400 tabular-nums">{chosen.cols * chosen.rows} pieces</span>
        </div>
      )}
      <p className="text-xs text-slate-400">
        The grid must match the real puzzle. If the lines don't follow the picture, count the pieces along one edge.
      </p>
      <button
        type="button"
        disabled={chosen === null}
        onClick={onStart}
        className="bg-accent text-ink-950 active:bg-accent-strong w-full rounded-2xl px-6 py-4 text-lg font-semibold disabled:opacity-40"
      >
        Start scanning
      </button>
    </section>
  )
}

function Stepper({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <span className="inline-flex items-center overflow-hidden rounded-xl bg-ink-800">
      <button type="button" onClick={() => onChange(value - 1)} className="px-3 py-1.5 text-white active:bg-ink-700" aria-label="Fewer">
        −
      </button>
      <span className="w-8 text-center text-white tabular-nums">{value}</span>
      <button type="button" onClick={() => onChange(value + 1)} className="px-3 py-1.5 text-white active:bg-ink-700" aria-label="More">
        +
      </button>
    </span>
  )
}
