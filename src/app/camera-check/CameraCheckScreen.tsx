/**
 * Camera check: a full-screen live camera view that proves the whole frame path on the real phone
 * (camera → ImageBitmap → worker → pixels → result) and reports what the phone can do.
 *
 * Layering, bottom to top: the <video> filling the screen with object-fit: cover, the overlay
 * canvas (FrameOverlay, drawn in frame coordinates through the same cover mapping), the top bar,
 * and either the HUD (camera running) or a status/error panel.
 *
 * iOS video rules: the element is playsInline (no forced fullscreen), muted and autoPlay (allowed
 * to play without a tap). If the browser still refuses to play (iOS Low Power Mode can), a
 * "Tap to show the camera" button starts playback from a real tap.
 */

import { useEffect, useRef, useState } from 'react'
import type { CameraSession } from '../../camera/session.ts'
import { useCameraState } from '../../camera/useCameraState.ts'
import { errorMessage } from '../errorMessage.ts'
import { CameraMessage } from './CameraMessage.tsx'
import { FrameOverlay } from './FrameOverlay.tsx'
import { Hud } from './Hud.tsx'
import { useCapabilities } from './useCapabilities.ts'
import { useFramePump } from './useFramePump.ts'

interface CameraCheckScreenProps {
  camera: CameraSession
  onBack: () => void
}

export function CameraCheckScreen({ camera, onBack }: CameraCheckScreenProps) {
  const state = useCameraState(camera)
  const stream = state.status === 'running' ? state.stream : null
  const videoRef = useRef<HTMLVideoElement>(null)
  /** The stream whose playback the browser blocked until a tap (null when none is blocked). */
  const [blockedStream, setBlockedStream] = useState<MediaStream | null>(null)
  const [torchError, setTorchError] = useState<string | null>(null)
  const [detailsOpen, setDetailsOpen] = useState(false)
  const pump = useFramePump(videoRef, stream)
  const capabilities = useCapabilities()

  // The camera runs while this screen is shown; leaving it stops the tracks.
  useEffect(() => camera.hold(), [camera])

  useEffect(() => {
    const video = videoRef.current
    if (video === null) return
    video.srcObject = stream
    if (stream === null) return
    video.play().catch((error: unknown) => {
      // AbortError just means a newer stream replaced this one before it started playing.
      if (error instanceof DOMException && error.name === 'NotAllowedError') setBlockedStream(stream)
    })
  }, [stream])

  const playBlocked = stream !== null && blockedStream === stream
  const result = pump.stats?.lastResult

  return (
    <main className="relative h-dvh w-full overflow-hidden bg-black">
      <video
        ref={videoRef}
        playsInline
        muted
        autoPlay
        className={`absolute inset-0 size-full object-cover ${stream === null ? 'invisible' : ''}`}
      />
      {stream !== null && result !== undefined && (
        <FrameOverlay frameWidth={result.width} frameHeight={result.height} />
      )}

      <header className="pt-safe px-safe absolute inset-x-0 top-0 flex items-center gap-3 bg-linear-to-b from-black/70 to-transparent pb-8">
        <button
          type="button"
          onClick={onBack}
          className="rounded-full bg-black/55 px-4 py-2 text-sm font-medium text-white backdrop-blur-md active:bg-black/75"
        >
          ‹ Back
        </button>
        <h1 className="text-base font-semibold text-white">Camera check</h1>
      </header>

      {state.status === 'running' ? (
        <>
          {playBlocked && (
            <div className="absolute inset-0 flex items-center justify-center">
              <button
                type="button"
                onClick={() => {
                  videoRef.current?.play().then(
                    () => setBlockedStream(null),
                    // Still refused: the button stays for another tap.
                    () => {},
                  )
                }}
                className="bg-accent text-ink-950 rounded-2xl px-6 py-4 text-lg font-semibold"
              >
                Tap to show the camera
              </button>
            </div>
          )}
          <Hud
            camera={state}
            pump={pump}
            capabilities={capabilities}
            torchError={torchError}
            detailsOpen={detailsOpen}
            onToggleDetails={() => setDetailsOpen((open) => !open)}
            onSelectDevice={(deviceId) => {
              setTorchError(null)
              camera.selectDevice(deviceId)
            }}
            onToggleTorch={() => {
              setTorchError(null)
              camera.setTorch(!state.torchOn).catch((error: unknown) => setTorchError(errorMessage(error)))
            }}
          />
        </>
      ) : (
        <CameraMessage state={state} onStart={camera.start} />
      )}
    </main>
  )
}
