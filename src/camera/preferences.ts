/**
 * Remembers which camera the user picked, so the next session opens the same lens.
 *
 * Only an explicit choice from the device picker is stored, never the browser's default. On
 * iPhone the choice matters: the multi-camera virtual devices ("Back Dual Wide Camera", "Back
 * Triple Camera") switch to the macro lens up close, which may suit close-up pieces better than
 * the plain "Back Camera". localStorage can throw (private mode, storage disabled), in which case
 * the app just forgets the choice.
 */

const STORAGE_KEY = 'piece-finder:camera-device-id'

export function loadPreferredDeviceId(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY)
  } catch {
    return null
  }
}

export function savePreferredDeviceId(deviceId: string): void {
  try {
    localStorage.setItem(STORAGE_KEY, deviceId)
  } catch {
    // Storage unavailable: the choice lasts for this session only.
  }
}
