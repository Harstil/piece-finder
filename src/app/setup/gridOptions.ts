/**
 * Grid suggestions: which cols × rows could a puzzle with this piece count and picture shape have?
 *
 * The naive √(N · aspect) guess is often wrong: Ravensburger's "1000" is 36 × 28 = 1008 with non-square
 * 19.4 × 17.8 mm cells, which the naive formula would call 38 × 26 — two columns off at the right edge.
 * So the app lists every grid whose piece count is within MAX_COUNT_ERROR of the box's number and whose
 * cells are not too far from square, best first, and lets the user type cols × rows directly. Getting
 * the grid right matters more than anything else the user enters.
 */

import type { GridSpec } from '../../engine/types.ts'

/** Real piece counts differ from the box number by a few pieces (1008 for "1000"). Guessed tolerance. */
const MAX_COUNT_ERROR = 0.05
/** Cell aspect (width ÷ height) allowed between 1/MAX and MAX. Guessed from commercial cuts (≤ 1.2). */
const MAX_CELL_ASPECT = 1.35
const MAX_OPTIONS = 6

export interface GridOption extends GridSpec {
  count: number
  cellAspect: number
}

/** Plausible grids for `pieces` pieces on a picture of width ÷ height = `aspect`, best first. */
export function gridOptions(pieces: number, aspect: number): GridOption[] {
  if (!(pieces >= 4) || !(aspect > 0)) return []
  const options: (GridOption & { cost: number })[] = []
  const maxRows = Math.ceil(Math.sqrt(pieces * (1 + MAX_COUNT_ERROR) * MAX_CELL_ASPECT / aspect)) + 1
  for (let rows = 2; rows <= maxRows; rows++) {
    const exact = pieces / rows
    for (const cols of new Set([Math.floor(exact), Math.ceil(exact), Math.floor(exact) - 1, Math.ceil(exact) + 1])) {
      if (cols < 2) continue
      const count = cols * rows
      const countError = Math.abs(count - pieces) / pieces
      const cellAspect = (aspect * rows) / cols
      if (countError > MAX_COUNT_ERROR || cellAspect > MAX_CELL_ASPECT || cellAspect < 1 / MAX_CELL_ASPECT) continue
      // Box numbers are usually exact, so a count error costs 4× more than in the synthetic generator's
      // trade-off: 0.5 % count error ≈ 8 % cell-aspect error. Guessed; the user picks from the list anyway.
      const cost = 4 * countError + Math.abs(Math.log(cellAspect)) / 4
      options.push({ cols, rows, count, cellAspect, cost })
    }
  }
  options.sort((a, b) => a.cost - b.cost)
  const seen = new Set<string>()
  return options
    .filter((o) => !seen.has(`${o.cols}x${o.rows}`) && (seen.add(`${o.cols}x${o.rows}`), true))
    .slice(0, MAX_OPTIONS)
    .map(({ cols, rows, count, cellAspect }) => ({ cols, rows, count, cellAspect }))
}
