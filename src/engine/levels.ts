/**
 * The matching resolutions, shared by the reference model, the canonical piece and the matcher.
 *
 * Both the box picture and every piece are resampled so that one grid cell / one piece core is a
 * square of `s` px, at s = 8 (coarse: every cell × 4 rotations), 16 (the best few dozen) and 32 (the
 * finalists). Around the core each keeps a margin of 3/8 · s: a piece's tabs reach up to ~0.3 of a
 * cell into its neighbours, and those tab pixels are extra evidence. 3/8 keeps the margin an integer
 * at every level, so the levels are exact area-downsamples of each other (8 → 3 px, 16 → 6, 32 → 12).
 */

export const LEVEL_SIZES = [8, 16, 32] as const
export const FINEST = LEVEL_SIZES[LEVEL_SIZES.length - 1]

/** Canonical piece work size: warped once at this core size, then area-downsampled to each level. */
export const WORK_SIZE = 64

export function levelMargin(size: number): number {
  return (size * 3) / 8
}

/** Side of a canonical patch / reference window at core size `size` (core + 2 margins). */
export function levelSpan(size: number): number {
  return size + 2 * levelMargin(size)
}
