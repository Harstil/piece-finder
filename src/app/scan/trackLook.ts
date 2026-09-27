/**
 * How the scanner shows a tracked piece: locked (green), likely (amber), still searching (white), or —
 * with a Region-finder area active — belonging to the area (blue glow) or not (faded).
 */

import type { TrackState } from '../../engine/types.ts'

/** Top probability from which an unlocked answer is shown as "likely". Guessed, matches fusion.ts verdicts. */
const LIKELY_PROB = 0.35
/** Region probability from which a piece counts as "belongs to the area". Guessed. */
const REGION_PROB = 0.5

export type TrackLook = 'locked' | 'likely' | 'searching' | 'region' | 'faded'

export function trackLook(t: TrackState, regionActive: boolean): TrackLook {
  if (regionActive) return (t.regionProb ?? 0) >= REGION_PROB ? 'region' : 'faded'
  if (t.locked) return 'locked'
  return (t.candidates[0]?.prob ?? 0) >= LIKELY_PROB ? 'likely' : 'searching'
}
