/**
 * How cue values become one score per hypothesis, probabilities, and a verdict.
 *
 * score = wLuma·t·zncc(L) + wChroma·zncc(a,b) − wChromaMean·Δab/10 + wGradient·gradCorr
 *         − wStats·(|ΔmeanL|/20 + |ΔstdL|/10) − wShapeMismatch·(#sides contradicting the cell's border)
 * where t = min(1, piece L std / lumaTextureRef) fades texture matching out on flat pieces (sky, water),
 * whose ZNCC is dominated by noise. Probabilities are a softmax over the finalists' scores.
 *
 * DEFAULT_FUSION was fitted by eval/run.ts --fit on the v1 val sets (see the note beside it); every other
 * number here is a documented guess.
 */

import type { FusionParams, Verdict } from '../types.ts'

// Fitted 2026-09-27: `node eval/run.ts --sets v1/val100,v1/val300,v1/val500,v1/val1000,v1/val2000
// --shape outline --limit 120 --fit` (563 pieces; coordinate search on top-1 + 0.02 · log-likelihood, then
// the temperature by log-likelihood). Objective 366 → 416: pattern cues (luma, chroma pattern, gradient)
// gained weight and absolute-colour penalties lost most of theirs — box print and camera disagree on colour.
// coarseTemperature (not fitted) is scaled with the ~2.5× larger weights.
export const DEFAULT_FUSION: FusionParams = {
  wLuma: 2.66,
  wChroma: 1.77,
  wChromaMean: 0.084,
  wGradient: 2.13,
  wStats: 0.11,
  wShapeMismatch: 2.0,
  lumaTextureRef: 7.1,
  temperature: 0.13,
  coarseTemperature: 0.25,
  finalistMass: 0.95,
}

export function resolveFusion(override?: Partial<FusionParams>): FusionParams {
  return override === undefined ? DEFAULT_FUSION : { ...DEFAULT_FUSION, ...override }
}

/**
 * Verdict from the top two candidate probabilities. Thresholds guessed from the prior-art verdicts
 * (johnb8005: "strong" needs a clear margin, "likely" a leader) and then checked against eval/run.ts's
 * verdict precision table: strong should be right ≥ 95 % of the time.
 */
const STRONG_P = 0.7
const STRONG_MARGIN = 0.4
const LIKELY_P = 0.4
const LIKELY_MARGIN = 0.15

export function verdictFor(p1: number, p2: number): Verdict {
  if (p1 >= STRONG_P && p1 - p2 >= STRONG_MARGIN) return 'strong'
  if (p1 >= LIKELY_P && p1 - p2 >= LIKELY_MARGIN) return 'likely'
  return 'unsure'
}
