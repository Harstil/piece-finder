/**
 * Camera session: owns the one live camera stream and its lifecycle, as a small external store
 * that React screens subscribe to (see useCameraSession.ts).
 *
 * Lifecycle rules, mostly driven by iOS:
 * - The camera opens only when the user asks: `start()` is called from a tap handler (the Home
 *   screen's Camera check button, or Try again), never at page load. In a Home Screen web app iOS
 *   asks for camera permission again on every launch, so an unprompted open would greet the user
 *   with a permission dialog.
 * - While the page is hidden (app switcher, lock screen, another tab) the tracks are stopped, so
 *   the camera light goes off and the battery is spared; when the page is visible again the same
 *   camera is reopened.
 * - A screen that shows the camera holds the session (`hold()`); when the last screen lets go,
 *   the tracks are stopped.
 * - Opening is asynchronous and can overlap with stop, lens switches and visibility changes. Every
 *   open gets a generation number, and a stream that arrives after it was superseded is stopped
 *   at once instead of leaking a running camera.
 *
 * Switching lens, the remembered lens choice and the torch also live here, because each of them
 * reopens or reconfigures the running track.
 */

import {
  applyTorch,
  listVideoInputs,
  openCamera,
  readTrackReport,
  stopStream,
  type CameraDevice,
  type TrackReport,
} from './camera.ts'
import { classifyCameraError, type CameraFailure } from './errors.ts'
import { loadPreferredDeviceId, savePreferredDeviceId } from './preferences.ts'

export interface RunningCamera {
  status: 'running'
  stream: MediaStream
  report: TrackReport
  /** All video inputs; labels are known because permission has been granted. */
  devices: CameraDevice[]
  /** deviceId of the running track, or null when the browser does not report it. */
  activeDeviceId: string | null
  torchOn: boolean
}

export type CameraState =
  | { status: 'idle' }
  | { status: 'starting' }
  /** Stopped because the page is hidden; reopens by itself when the page is visible again. */
  | { status: 'paused' }
  | RunningCamera
  | { status: 'error'; failure: CameraFailure }

export interface CameraSession {
  getState(): CameraState
  subscribe(listener: () => void): () => void
  /** Opens the camera (or retries after an error). Call from a user gesture. */
  start(): void
  /** Stops the tracks; the session can be started again. */
  stop(): void
  /**
   * Keeps the camera running while a screen shows it; returns the release function, meant to be
   * returned from a useEffect. The stop after the last release is deferred by one task, so React
   * StrictMode's development-only unmount + remount does not close the camera the user just opened.
   */
  hold(): () => void
  /** Reopens with another camera and remembers the choice for next time. */
  selectDevice(deviceId: string): void
  /** Switches the torch; rejects when the camera refuses. Only offer it when report says torch. */
  setTorch(on: boolean): Promise<void>
}

export function createCameraSession(): CameraSession {
  let state: CameraState = { status: 'idle' }
  const listeners = new Set<() => void>()
  /** True between start() and stop(): the user wants the camera on. */
  let wanted = false
  /**
   * The lens to open; null means the default rear camera. Kept here as well as in localStorage,
   * so a lens switch still works where storage throws (private browsing).
   */
  let preferredDeviceId = loadPreferredDeviceId()
  let stream: MediaStream | null = null
  let generation = 0
  let holders = 0
  let pendingStop: ReturnType<typeof setTimeout> | null = null

  function setState(next: CameraState): void {
    state = next
    for (const listener of listeners) listener()
  }

  /** Stops the current stream (if any) and invalidates any open still in flight. */
  function closeStream(): void {
    generation += 1
    if (stream !== null) {
      stream.getVideoTracks()[0]?.removeEventListener('ended', onTrackEnded)
      stopStream(stream)
      stream = null
    }
  }

  async function open(): Promise<void> {
    closeStream()
    const openGeneration = generation
    setState({ status: 'starting' })
    try {
      const opened = await openCamera(preferredDeviceId)
      if (openGeneration !== generation) {
        stopStream(opened.stream)
        return
      }
      // The remembered lens is gone (ids change when site data is cleared): stop asking for it.
      if (opened.usedFallback) preferredDeviceId = null
      // Assigned before the next await, so a stop() meanwhile closes this stream.
      stream = opened.stream
      const track = stream.getVideoTracks()[0]
      track.addEventListener('ended', onTrackEnded)
      const devices = await listVideoInputs()
      if (openGeneration !== generation) return
      setState(runningState(stream, track, devices))
    } catch (error) {
      if (openGeneration !== generation) return
      closeStream()
      setState({ status: 'error', failure: classifyCameraError(error) })
    }
  }

  function runningState(
    running: MediaStream,
    track: MediaStreamTrack,
    devices: CameraDevice[],
  ): RunningCamera {
    const report = readTrackReport(track)
    return {
      status: 'running',
      stream: running,
      report,
      devices,
      activeDeviceId: track.getSettings().deviceId ?? null,
      torchOn: report.settings.torch === true,
    }
  }

  /**
   * The track ended without us stopping it: a phone call, or another app took the camera. If the
   * page is hidden at that moment, treat it like the pause below so it reopens on return.
   */
  function onTrackEnded(): void {
    closeStream()
    if (document.hidden) {
      setState({ status: 'paused' })
      return
    }
    setState({
      status: 'error',
      failure: { kind: 'interrupted', detail: 'The camera track ended' },
    })
  }

  function onVisibilityChange(): void {
    if (!wanted) return
    if (document.hidden) {
      if (state.status === 'running' || state.status === 'starting') {
        closeStream()
        setState({ status: 'paused' })
      }
    } else if (state.status === 'paused') {
      void open()
    }
  }

  function cancelPendingStop(): void {
    if (pendingStop !== null) {
      clearTimeout(pendingStop)
      pendingStop = null
    }
  }

  function stop(): void {
    cancelPendingStop()
    if (wanted) {
      wanted = false
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
    closeStream()
    setState({ status: 'idle' })
  }

  return {
    getState: () => state,

    subscribe(listener) {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },

    start() {
      cancelPendingStop()
      if (!wanted) {
        wanted = true
        document.addEventListener('visibilitychange', onVisibilityChange)
      }
      if (state.status === 'running' || state.status === 'starting') return
      void open()
    },

    stop,

    hold() {
      holders += 1
      cancelPendingStop()
      let released = false
      return () => {
        if (released) return
        released = true
        holders -= 1
        if (holders === 0) pendingStop = setTimeout(stop, 0)
      }
    },

    selectDevice(deviceId) {
      preferredDeviceId = deviceId
      savePreferredDeviceId(deviceId)
      if (wanted) void open()
    },

    async setTorch(on) {
      if (state.status !== 'running') return
      const running = state
      const track = running.stream.getVideoTracks()[0]
      try {
        await applyTorch(track, on)
      } finally {
        // Re-read either way: the settings say what the camera actually did.
        if (state === running) setState(runningState(running.stream, track, running.devices))
      }
    },
  }
}
