/**
 * The scanner: hold the phone over loose pieces and every piece in view gets an outline, its place in
 * the picture ("C12 · R7") and an arrow to its top. Tap a piece for the zoomed picture, alternatives and
 * "Mark as placed". The mini-map shows where the confident answers are; with a Region-finder area set,
 * pieces that belong there glow instead.
 *
 * The camera and frame pump are the Camera check's (same session, same one-frame-in-flight pump); the
 * worker runs the whole pipeline per frame (src/engine/pipeline.ts) once the puzzle is prepared.
 */

import { useCallback, useEffect, useRef, useState, type MouseEvent } from 'react'
import { coverTransform } from '../../camera/coverTransform.ts'
import type { CameraSession } from '../../camera/session.ts'
import { useCameraState } from '../../camera/useCameraState.ts'
import { pointInPolygon } from '../../engine/geom/index.ts'
import type { TrackState } from '../../engine/types.ts'
import { CameraMessage } from '../camera-check/CameraMessage.tsx'
import { useFramePump } from '../camera-check/useFramePump.ts'
import { errorMessage } from '../errorMessage.ts'
import { COLORS, drawPicture } from '../picture/drawPicture.ts'
import type { PuzzleControls, PuzzleView } from '../usePuzzle.ts'
import { useCanvas } from '../useCanvas.ts'
import { PieceSheet } from './PieceSheet.tsx'
import { ScanOverlay } from './ScanOverlay.tsx'
import { trackLook } from './trackLook.ts'

interface ScanScreenProps {
  camera: CameraSession
  puzzle: PuzzleControls
  onBack: () => void
  onPicture: () => void
}

export function ScanScreen({ camera, puzzle, onBack, onPicture }: ScanScreenProps) {
  const state = useCameraState(camera)
  const stream = state.status === 'running' ? state.stream : null
  const videoRef = useRef<HTMLVideoElement>(null)
  const [blocked, setBlocked] = useState<MediaStream | null>(null)
  const [selected, setSelected] = useState<TrackState | null>(null)
  const [torchError, setTorchError] = useState<string | null>(null)
  const pump = useFramePump(videoRef, stream)
  const status = puzzle.status
  const view = status.kind === 'ready' ? status.view : null

  useEffect(() => camera.hold(), [camera])

  useEffect(() => {
    const video = videoRef.current
    if (video === null) return
    video.srcObject = stream
    if (stream === null) return
    video.play().catch((error: unknown) => {
      if (error instanceof DOMException && error.name === 'NotAllowedError') setBlocked(stream)
    })
  }, [stream])

  const result = pump.stats?.lastResult ?? null
  const regionActive = puzzle.region !== null

  const onTap = (e: MouseEvent<HTMLDivElement>) => {
    if (result === null) return
    const rect = e.currentTarget.getBoundingClientRect()
    const { scale, offsetX, offsetY } = coverTransform(result.width, result.height, rect.width, rect.height)
    const p = { x: (e.clientX - rect.left - offsetX) / scale - 0.5, y: (e.clientY - rect.top - offsetY) / scale - 0.5 }
    const hit = result.tracks.find((t) => pointInPolygon(p, t.contour))
    if (hit !== undefined) setSelected(hit)
  }

  const tracks = result?.tracks ?? []
  const found = tracks.filter((t) => t.locked).length
  let hint: string
  if (status.kind === 'preparing' || status.kind === 'loading') hint = 'Preparing the picture…'
  else if (status.kind === 'error') hint = `Could not prepare the puzzle: ${status.message}`
  else if (tracks.length === 0) hint = 'Hold the phone 20–40 cm above pieces spread out so they don’t touch.'
  else if (regionActive) hint = `${tracks.filter((t) => trackLook(t, true) === 'region').length} of ${tracks.length} pieces belong to your area`
  else hint = `${tracks.length} piece${tracks.length === 1 ? '' : 's'} in view · ${found} placed on the picture`

  return (
    <main className="relative h-dvh w-full overflow-hidden bg-black">
      <video ref={videoRef} playsInline muted autoPlay className={`absolute inset-0 size-full object-cover ${stream === null ? 'invisible' : ''}`} />
      {view !== null && <ScanOverlay result={result} grid={view.record.grid} regionActive={regionActive} />}
      <div className="absolute inset-0" onClick={onTap} />

      <header className="pt-safe px-safe absolute inset-x-0 top-0 flex items-center gap-2 bg-linear-to-b from-black/75 to-transparent pb-10">
        <button type="button" onClick={onBack} className="rounded-full bg-black/55 px-4 py-2 text-sm font-medium text-white backdrop-blur-md active:bg-black/75">
          ‹ Home
        </button>
        <p className="min-w-0 flex-1 text-sm leading-snug text-white">{hint}</p>
        {state.status === 'running' && state.report.capabilities?.torch === true && (
          <button
            type="button"
            onClick={() => {
              setTorchError(null)
              camera.setTorch(!state.torchOn).catch((error: unknown) => setTorchError(errorMessage(error)))
            }}
            className={`rounded-full px-3 py-2 text-sm font-medium backdrop-blur-md ${state.torchOn ? 'bg-accent text-ink-950' : 'bg-black/55 text-white'}`}
          >
            Light
          </button>
        )}
      </header>
      {torchError !== null && <p className="px-safe absolute inset-x-0 top-24 text-sm text-red-300">{torchError}</p>}

      {state.status === 'running' ? (
        blocked === stream &&
        stream !== null && (
          <div className="absolute inset-0 flex items-center justify-center">
            <button
              type="button"
              onClick={() => void videoRef.current?.play().then(() => setBlocked(null), () => {})}
              className="bg-accent text-ink-950 rounded-2xl px-6 py-4 text-lg font-semibold"
            >
              Tap to show the camera
            </button>
          </div>
        )
      ) : (
        <CameraMessage state={state} onStart={camera.start} />
      )}

      {view !== null && (
        <footer className="pb-safe px-safe absolute inset-x-0 bottom-0 flex items-end gap-3 bg-linear-to-t from-black/75 to-transparent pt-10">
          <MiniMap view={view} tracks={tracks} region={puzzle.region} onClick={onPicture} />
          <div className="flex flex-1 flex-col items-end gap-2">
            {regionActive && (
              <button type="button" onClick={() => puzzle.setRegion(null)} className="rounded-full bg-sky-500/90 px-4 py-2 text-sm font-semibold text-ink-950">
                Clear area
              </button>
            )}
            <button type="button" onClick={onPicture} className="rounded-full bg-black/55 px-4 py-2 text-sm font-medium text-white backdrop-blur-md active:bg-black/75">
              Picture · {view.record.placed.length}/{view.record.grid.cols * view.record.grid.rows}
            </button>
            {pump.stats !== null && (
              <span className="text-[11px] text-slate-400 tabular-nums">
                {pump.stats.processedFps.toFixed(1)} fps · {Math.round(pump.stats.latencyMs)} ms
              </span>
            )}
            {pump.error !== null && <span className="text-xs text-red-300">{pump.error}</span>}
          </div>
        </footer>
      )}

      {selected !== null && view !== null && (
        <PieceSheet
          track={tracks.find((t) => t.trackId === selected.trackId) ?? selected}
          picture={view.picture}
          grid={view.record.grid}
          placed={view.record.placed}
          onClose={() => setSelected(null)}
          onPlace={(cell) => {
            if (!view.record.placed.includes(cell)) puzzle.setPlaced([...view.record.placed, cell])
            setSelected(null)
          }}
        />
      )}
    </main>
  )
}

interface MiniMapProps {
  view: PuzzleView
  tracks: TrackState[]
  region: ReadonlySet<number> | null
  onClick: () => void
}

/** Maximum mini-map size (CSS px). */
const MINIMAP_MAX = 150

function MiniMap({ view, tracks, region, onClick }: MiniMapProps) {
  const { picture, record } = view
  const k = MINIMAP_MAX / Math.max(picture.width, picture.height)
  const width = Math.round(picture.width * k)
  const height = Math.round(picture.height * k)
  const draw = useCallback(
    (ctx: CanvasRenderingContext2D, size: { width: number; height: number }) => {
      drawPicture(ctx, picture, 0, 0, size.width, size.height, {
        grid: record.grid,
        placed: record.placed,
        region,
        highlights: tracks
          .filter((t) => t.locked && t.candidates.length > 0)
          .map((t) => ({ cell: t.candidates[0].cell, color: COLORS.strong, width: 2.5 })),
      })
    },
    [picture, record, tracks, region],
  )
  const ref = useCanvas(draw)
  return (
    <button type="button" onClick={onClick} className="overflow-hidden rounded-xl border border-white/30 shadow-lg" aria-label="Open the picture">
      <canvas ref={ref} style={{ width, height }} className="block" />
    </button>
  )
}
