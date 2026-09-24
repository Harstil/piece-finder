/**
 * Thin, typed wrappers over the browser camera APIs: open the rear camera, list cameras, read what
 * the track actually delivers, and switch the torch.
 *
 * Phones are the target, so a few platform facts shape this file:
 * - `facingMode: environment` picks the rear camera; on iPhone that is the plain 1× lens.
 * - Resolution is requested as 1920×1080 in the sensor's landscape terms. In portrait the browser
 *   rotates the frames, so the <video> reports 1080×1920 while Safari's track settings may still say
 *   1920×1080. The video element's videoWidth/videoHeight (and so the ImageBitmaps made from it) are
 *   the truth for frame coordinates; track settings are shown for diagnostics only.
 * - Device labels are empty until permission is granted, so devices are listed after opening.
 * - Torch, zoom and focusMode are not in every browser's type definitions or implementation; they
 *   are read defensively and reported as unavailable when missing.
 */

import { CameraUnavailableError, isMissingDeviceError } from './errors.ts'

/**
 * Requested frame size in the sensor's landscape orientation. 1080p is what iPhones deliver
 * reliably (researched, see the project plan); the pieces need the detail.
 */
const IDEAL_WIDTH = 1920
const IDEAL_HEIGHT = 1080
/** Requested frame rate. The engine will not process more than this; 60 fps would cost battery. */
const IDEAL_FRAME_RATE = 30

export interface CameraDevice {
  deviceId: string
  /** Browser-provided name, e.g. "Back Ultra Wide Camera" (iPhone labels are localized). */
  label: string
}

export interface NumberRange {
  min: number
  max: number
  step: number | null
}

/** What the running track delivers and what it could do. Null fields are not reported by the browser. */
export interface TrackReport {
  label: string
  settings: {
    width: number | null
    height: number | null
    frameRate: number | null
    facingMode: string | null
    focusMode: string | null
    zoom: number | null
    torch: boolean | null
  }
  /** Null when the browser has no getCapabilities (older Firefox). */
  capabilities: {
    width: NumberRange | null
    height: NumberRange | null
    frameRate: NumberRange | null
    focusModes: string[]
    zoom: NumberRange | null
    torch: boolean
  } | null
}

/** Capabilities the TypeScript DOM types don't list yet (Image Capture extensions). */
type ExtendedCapabilities = MediaTrackCapabilities & {
  focusMode?: string[]
  zoom?: { min?: number; max?: number; step?: number }
  torch?: boolean
}
type ExtendedSettings = MediaTrackSettings & { focusMode?: string }
type TorchConstraintSet = MediaTrackConstraintSet & { torch: boolean }

export interface OpenedCamera {
  stream: MediaStream
  /** True when the requested device was gone and the default rear camera was opened instead. */
  usedFallback: boolean
}

/**
 * Open a camera. Must be called from a user gesture on iOS the first time. When `deviceId` is
 * given but that device no longer exists (ids can change after site data is cleared), falls back
 * to the default rear camera.
 */
export async function openCamera(deviceId: string | null): Promise<OpenedCamera> {
  assertCameraApi()
  if (deviceId !== null) {
    try {
      return { stream: await getVideoStream(deviceId), usedFallback: false }
    } catch (error) {
      if (!isMissingDeviceError(error)) throw error
    }
    return { stream: await getVideoStream(null), usedFallback: true }
  }
  return { stream: await getVideoStream(null), usedFallback: false }
}

function assertCameraApi(): void {
  if (!window.isSecureContext) {
    throw new CameraUnavailableError(
      'insecure-context',
      `${location.origin} is not a secure context, so the browser hides the camera`,
    )
  }
  if (typeof navigator.mediaDevices?.getUserMedia !== 'function') {
    throw new CameraUnavailableError('unsupported', 'navigator.mediaDevices.getUserMedia is missing')
  }
}

function getVideoStream(deviceId: string | null): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      ...(deviceId === null
        ? { facingMode: { ideal: 'environment' } }
        : { deviceId: { exact: deviceId } }),
      width: { ideal: IDEAL_WIDTH },
      height: { ideal: IDEAL_HEIGHT },
      frameRate: { ideal: IDEAL_FRAME_RATE },
    },
  })
}

export function stopStream(stream: MediaStream): void {
  for (const track of stream.getTracks()) track.stop()
}

/** Video inputs, with a readable fallback name when the browser withholds labels. */
export async function listVideoInputs(): Promise<CameraDevice[]> {
  const devices = await navigator.mediaDevices.enumerateDevices()
  return devices
    .filter((device) => device.kind === 'videoinput' && device.deviceId !== '')
    .map((device, index) => ({
      deviceId: device.deviceId,
      label: device.label || `Camera ${index + 1}`,
    }))
}

export function readTrackReport(track: MediaStreamTrack): TrackReport {
  const settings = track.getSettings() as ExtendedSettings
  const capabilities =
    typeof track.getCapabilities === 'function'
      ? (track.getCapabilities() as ExtendedCapabilities)
      : null
  return {
    label: track.label,
    settings: {
      width: settings.width ?? null,
      height: settings.height ?? null,
      frameRate: settings.frameRate ?? null,
      facingMode: settings.facingMode ?? null,
      focusMode: settings.focusMode ?? null,
      zoom: settings.zoom ?? null,
      torch: settings.torch ?? null,
    },
    capabilities:
      capabilities === null
        ? null
        : {
            width: toRange(capabilities.width),
            height: toRange(capabilities.height),
            frameRate: toRange(capabilities.frameRate),
            focusModes: capabilities.focusMode ?? [],
            zoom: toRange(capabilities.zoom),
            torch: capabilities.torch === true,
          },
  }
}

function toRange(range: { min?: number; max?: number; step?: number } | undefined): NumberRange | null {
  if (range?.min === undefined || range.max === undefined) return null
  return { min: range.min, max: range.max, step: range.step ?? null }
}

/** Switch the torch; only call when the track's capabilities report `torch`. */
export async function applyTorch(track: MediaStreamTrack, on: boolean): Promise<void> {
  const torch: TorchConstraintSet = { torch: on }
  await track.applyConstraints({ advanced: [torch] })
}
