/**
 * Per-rotation, per-level views of a canonical piece, laid out for the scoring loop.
 *
 * For each level and each quarter turn r (the piece turned clockwise r × 90°, i.e. the hypothesis
 * "rotation r"), this lists the piece's masked pixels as offsets into a reference window (so scoring a
 * hypothesis is one flat loop over those offsets) together with their L, a, b values already
 * zero-meaned and scaled to unit norm, which turns masked ZNCC into a dot product:
 *   zncc = Σ p'·q / sqrt(Σq² − (Σq)²/n)   because Σp' = 0 and Σp'² = 1.
 * Gradient samples use the mask eroded by one more pixel, so no Sobel stencil touches the table around
 * the piece (that edge does not exist in the box picture), and are normalised to unit total energy.
 * Gradients are computed after rotating, so their directions rotate with the piece.
 */

import { rotateLab, rotateMask } from '../geom/index.ts'
import { erode, rectKernel, sobel } from '../image/index.ts'
import type { CanonicalPiece } from '../types.ts'

export interface PieceView {
  /** Masked pixel count. */
  n: number
  /** Offsets (y · refWidth + x) from the window origin, for the n masked pixels. */
  off: Int32Array
  /** Zero-mean, unit-norm values (all zero when the channel is flat). */
  pl: Float32Array
  pa: Float32Array
  pb: Float32Array
  meanL: number
  meanA: number
  meanB: number
  stdL: number
  stdA: number
  stdB: number
  /** Gradient samples (interior of the mask). */
  gn: number
  goff: Int32Array
  /** Gradient vectors scaled so that Σ(gx² + gy²) = 1 (all zero when there is no gradient). */
  gx: Float32Array
  gy: Float32Array
}

/** views[level][rotation]. `refWidths[level]` is the padded width of the reference level. */
export function prepareViews(piece: CanonicalPiece, refWidths: readonly number[]): PieceView[][] {
  return piece.levels.map((level, li) => {
    const views: PieceView[] = []
    for (let r = 0; r < 4; r++) {
      const lab = r === 0 ? level.lab : rotateLab(level.lab, r)
      const mask = r === 0 ? level.mask : rotateMask(level.mask, r)
      views.push(buildView(lab.L, lab.a, lab.b, mask.data, mask.width, refWidths[li], erode(mask, rectKernel(3)).data))
    }
    return views
  })
}

function buildView(
  L: Float32Array,
  A: Float32Array,
  B: Float32Array,
  mask: Uint8Array,
  size: number,
  refWidth: number,
  inner: Uint8Array,
): PieceView {
  let n = 0
  let gn = 0
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] !== 0) n++
    if (inner[i] !== 0) gn++
  }
  const off = new Int32Array(n)
  const pl = new Float32Array(n)
  const pa = new Float32Array(n)
  const pb = new Float32Array(n)
  let k = 0
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x
      if (mask[i] === 0) continue
      off[k] = y * refWidth + x
      pl[k] = L[i]
      pa[k] = A[i]
      pb[k] = B[i]
      k++
    }
  }
  const sl = normalise(pl)
  const sa = normalise(pa)
  const sb = normalise(pb)

  const grad = sobel({ width: size, height: size, data: L })
  const goff = new Int32Array(gn)
  const gx = new Float32Array(gn)
  const gy = new Float32Array(gn)
  let energy = 0
  k = 0
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = y * size + x
      if (inner[i] === 0) continue
      goff[k] = y * refWidth + x
      gx[k] = grad.dx.data[i]
      gy[k] = grad.dy.data[i]
      energy += gx[k] * gx[k] + gy[k] * gy[k]
      k++
    }
  }
  const gs = energy > 0 ? 1 / Math.sqrt(energy) : 0
  for (let j = 0; j < gn; j++) {
    gx[j] *= gs
    gy[j] *= gs
  }
  return { n, off, pl, pa, pb, meanL: sl.mean, meanA: sa.mean, meanB: sb.mean, stdL: sl.std, stdA: sa.std, stdB: sb.std, gn, goff, gx, gy }
}

/** In place: v ← (v − mean) / ‖v − mean‖. Returns the mean and the standard deviation. */
function normalise(v: Float32Array): { mean: number; std: number } {
  const n = v.length
  if (n === 0) return { mean: 0, std: 0 }
  let s = 0
  for (let i = 0; i < n; i++) s += v[i]
  const mean = s / n
  let ss = 0
  for (let i = 0; i < n; i++) {
    const d = v[i] - mean
    v[i] = d
    ss += d * d
  }
  const std = Math.sqrt(ss / n)
  const scale = ss > 1e-9 ? 1 / Math.sqrt(ss) : 0
  for (let i = 0; i < n; i++) v[i] *= scale
  return { mean, std }
}
