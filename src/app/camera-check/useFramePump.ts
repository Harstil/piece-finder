/**
 * Runs the frame pump (src/camera/framePump.ts) against the engine worker while a camera stream is
 * showing, and exposes its latest stats or error for the HUD.
 *
 * A new stream (another lens, or the camera reopening after the app was in the background) starts
 * a fresh pump; stats from the previous stream are not shown for the new one.
 */

import { useEffect, useState, type RefObject } from 'react'
import { startFramePump, type FramePumpStats } from '../../camera/framePump.ts'
import { getEngine } from '../../worker/client.ts'
import { errorMessage } from '../errorMessage.ts'

export interface FramePumpStatus {
  stats: FramePumpStats | null
  error: string | null
}

interface StreamStatus extends FramePumpStatus {
  stream: MediaStream
}

const NOTHING_YET: FramePumpStatus = { stats: null, error: null }

export function useFramePump(
  videoRef: RefObject<HTMLVideoElement | null>,
  stream: MediaStream | null,
): FramePumpStatus {
  const [status, setStatus] = useState<StreamStatus | null>(null)

  useEffect(() => {
    const video = videoRef.current
    if (stream === null || video === null) return
    return startFramePump(video, getEngine().processFrame, {
      onStats: (stats) => setStatus({ stream, stats, error: null }),
      onError: (error) =>
        setStatus((previous) => ({
          stream,
          stats: previous?.stream === stream ? previous.stats : null,
          error: errorMessage(error),
        })),
    })
  }, [videoRef, stream])

  return status !== null && status.stream === stream ? status : NOTHING_YET
}
