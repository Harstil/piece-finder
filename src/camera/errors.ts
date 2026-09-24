/**
 * Why the camera could not start, reduced to the few cases the user can act on.
 *
 * getUserMedia reports failures as DOMException names that differ slightly between Safari, Chrome
 * and Firefox; this maps them onto one small set so the UI can show one clear message per case.
 */

export type CameraFailureKind =
  /** Page is not https (or localhost): browsers hide the camera API entirely. */
  | 'insecure-context'
  /** The browser has no getUserMedia at all. */
  | 'unsupported'
  /** The user (or a browser setting) refused camera access. */
  | 'permission-denied'
  /** No camera, or the requested one no longer exists. */
  | 'no-camera'
  /** The camera exists but another app or tab holds it. */
  | 'camera-busy'
  /** The camera was running and then stopped on its own (a call, another app took it). */
  | 'interrupted'
  | 'unknown'

export interface CameraFailure {
  kind: CameraFailureKind
  /** The browser's own message, for the details line. */
  detail: string
}

/** Thrown by this module for failures detected before getUserMedia is even called. */
export class CameraUnavailableError extends Error {
  readonly kind: CameraFailureKind

  constructor(kind: CameraFailureKind, message: string) {
    super(message)
    this.name = 'CameraUnavailableError'
    this.kind = kind
  }
}

export function classifyCameraError(error: unknown): CameraFailure {
  if (error instanceof CameraUnavailableError) return { kind: error.kind, detail: error.message }
  const name = error instanceof Error ? error.name : ''
  const detail = error instanceof Error ? error.message : String(error)
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return { kind: 'permission-denied', detail }
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return { kind: 'no-camera', detail }
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return { kind: 'camera-busy', detail }
    default:
      return { kind: 'unknown', detail }
  }
}

/** Errors meaning "that particular device is gone", where falling back to the default camera helps. */
export function isMissingDeviceError(error: unknown): boolean {
  return error instanceof Error && (error.name === 'OverconstrainedError' || error.name === 'NotFoundError')
}
