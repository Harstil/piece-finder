/**
 * React binding for the camera session (session.ts): re-renders the screen whenever the camera
 * starts, stops, pauses, fails, switches lens or toggles the torch.
 */

import { useSyncExternalStore } from 'react'
import type { CameraSession, CameraState } from './session.ts'

export function useCameraState(session: CameraSession): CameraState {
  return useSyncExternalStore(session.subscribe, session.getState)
}
