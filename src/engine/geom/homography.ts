/**
 * Projective and affine transforms between point sets: homographies (exact from 4 points, least squares
 * from N), their application, inversion and composition, and least-squares affine / similarity fits.
 *
 * Homographies are how the engine removes perspective: box photo → rectified motif, piece corners →
 * canonical square, canonical square → frame for drawing overlays. Similarity fits serve tracking
 * (frame-to-frame motion of an outline) and the canonical warp's sanity checks.
 *
 * Representation: a Homography is a row-major 3×3 Float64Array [h0 h1 h2; h3 h4 h5; h6 h7 h8] mapping
 * (x, y) to ((h0 x + h1 y + h2) / w, (h3 x + h4 y + h5) / w) with w = h6 x + h7 y + h8, normalised so
 * h8 = 1 whenever h8 is not ~0. An Affine is a row-major 2×3 Float64Array [a b tx; c d ty].
 * Composition: compose(A, B) = A·B, i.e. apply B first, then A.
 *
 * Numerics: both solvers first apply Hartley normalisation (centroid to the origin, mean distance √2) so
 * pixel-scale coordinates do not wreck the conditioning. The 4-point solve is exact: an 8×8 linear
 * system (h8 = 1 in the normalised frame) by Gaussian elimination with partial pivoting; it equals
 * cv2.getPerspectiveTransform (golden-tested). The N-point fit minimises the algebraic DLT error: the
 * eigenvector of AᵀA with the smallest eigenvalue, via cyclic Jacobi. OpenCV's findHomography(method=0)
 * additionally refines the reprojection error with Levenberg–Marquardt, so with noisy points the two
 * differ slightly (bounded in the golden test); with exact points they agree.
 */

import type { Point } from '../types.ts'

export type Homography = Float64Array
export type Affine = Float64Array

/**
 * Pivots / relative determinants below this are treated as singular. Dimensionless because it is only
 * applied in the normalised frame. Guessed: ~1e4 × double epsilon, far below any usable configuration.
 */
const SINGULAR_EPS = 1e-12
/**
 * The N-point fit is degenerate (e.g. all points collinear) when the second-smallest eigenvalue of
 * AᵀA is below this fraction of the largest — then the solution is not unique. Guessed; generous
 * because genuine fits sit many orders of magnitude above it.
 */
const DEGENERATE_EIG_RATIO = 1e-10
/**
 * Jacobi stops when the off-diagonal energy is below this fraction of the diagonal energy, i.e. at
 * double precision. A 9×9 matrix converges in < 10 sweeps; the sweep cap only guards against NaN input.
 */
const JACOBI_TOLERANCE = 1e-30
const JACOBI_MAX_SWEEPS = 100

export function identityHomography(): Homography {
  return Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1)
}

/** Scales H so h8 = 1 (when h8 is not ~0); returns H. */
function normalize(H: Homography): Homography {
  const s = H[8]
  if (Math.abs(s) > SINGULAR_EPS) {
    for (let i = 0; i < 9; i++) H[i] /= s
    H[8] = 1
  }
  return H
}

/** Hartley normalisation T (as [s, tx, ty]: x' = s·x + tx) for a point set; null when all points coincide. */
function normalization(pts: readonly Point[]): [number, number, number] | null {
  let cx = 0
  let cy = 0
  for (const p of pts) {
    cx += p.x
    cy += p.y
  }
  cx /= pts.length
  cy /= pts.length
  let d = 0
  for (const p of pts) d += Math.hypot(p.x - cx, p.y - cy)
  d /= pts.length
  if (!(d > 0)) return null
  const s = Math.SQRT2 / d
  return [s, -s * cx, -s * cy]
}

/** H = Td⁻¹ · Hn · Ts for normalisations Ts = [s, tx, ty] (source) and Td (destination). */
function denormalize(Hn: ArrayLike<number>, ts: [number, number, number], td: [number, number, number]): Homography {
  const [ss, sx, sy] = ts
  const [ds, dx, dy] = td
  // Hn · Ts, Ts = [ss 0 sx; 0 ss sy; 0 0 1]
  const m = new Float64Array(9)
  for (let r = 0; r < 3; r++) {
    const a = Hn[r * 3]
    const b = Hn[r * 3 + 1]
    const c = Hn[r * 3 + 2]
    m[r * 3] = a * ss
    m[r * 3 + 1] = b * ss
    m[r * 3 + 2] = a * sx + b * sy + c
  }
  // Td⁻¹ = [1/ds 0 -dx/ds; 0 1/ds -dy/ds; 0 0 1]
  const H = new Float64Array(9)
  for (let c = 0; c < 3; c++) {
    H[c] = (m[c] - dx * m[6 + c]) / ds
    H[3 + c] = (m[3 + c] - dy * m[6 + c]) / ds
    H[6 + c] = m[6 + c]
  }
  return normalize(H)
}

/** Solves the n×n system M·x = b in place (M row-major, destroyed); returns false when singular. */
function solveLinear(M: Float64Array, b: Float64Array, n: number): boolean {
  for (let col = 0; col < n; col++) {
    let piv = col
    let best = Math.abs(M[col * n + col])
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(M[r * n + col])
      if (v > best) {
        best = v
        piv = r
      }
    }
    if (best < SINGULAR_EPS) return false
    if (piv !== col) {
      for (let k = 0; k < n; k++) {
        const t = M[col * n + k]
        M[col * n + k] = M[piv * n + k]
        M[piv * n + k] = t
      }
      const t = b[col]
      b[col] = b[piv]
      b[piv] = t
    }
    const inv = 1 / M[col * n + col]
    for (let r = col + 1; r < n; r++) {
      const f = M[r * n + col] * inv
      if (f === 0) continue
      for (let k = col; k < n; k++) M[r * n + k] -= f * M[col * n + k]
      b[r] -= f * b[col]
    }
  }
  for (let r = n - 1; r >= 0; r--) {
    let acc = b[r]
    for (let k = r + 1; k < n; k++) acc -= M[r * n + k] * b[k]
    b[r] = acc / M[r * n + r]
  }
  return true
}

/**
 * The homography mapping src[i] → dst[i] for exactly 4 correspondences (e.g. quad corners), or null
 * when three of the points are collinear.
 */
export function homographyFromQuad(src: readonly Point[], dst: readonly Point[]): Homography | null {
  if (src.length !== 4 || dst.length !== 4) throw new Error('homographyFromQuad needs exactly 4 point pairs')
  const ts = normalization(src)
  const td = normalization(dst)
  if (ts === null || td === null) return null
  const M = new Float64Array(64)
  const b = new Float64Array(8)
  for (let i = 0; i < 4; i++) {
    const x = ts[0] * src[i].x + ts[1]
    const y = ts[0] * src[i].y + ts[2]
    const u = td[0] * dst[i].x + td[1]
    const v = td[0] * dst[i].y + td[2]
    M.set([x, y, 1, 0, 0, 0, -u * x, -u * y], i * 16)
    M.set([0, 0, 0, x, y, 1, -v * x, -v * y], i * 16 + 8)
    b[i * 2] = u
    b[i * 2 + 1] = v
  }
  if (!solveLinear(M, b, 8)) {
    // h8 = 0 in the normalised frame (the source centroid maps to infinity): use the general solver.
    return homographyFromPoints(src, dst)
  }
  const Hn = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1]
  return isSingular(Hn) ? null : denormalize(Hn, ts, td)
}

function determinant(H: ArrayLike<number>): number {
  return H[0] * (H[4] * H[8] - H[5] * H[7]) - H[1] * (H[3] * H[8] - H[5] * H[6]) + H[2] * (H[3] * H[7] - H[4] * H[6])
}

/**
 * True when |det H| is negligible relative to the scale of its entries, i.e. H collapses the plane
 * onto a line or point. Only meaningful for a homography in the normalised frame, where all entries
 * are O(1); in pixel units the entries mix very different scales.
 */
function isSingular(H: ArrayLike<number>): boolean {
  let scale = 0
  for (let k = 0; k < 9; k++) scale = Math.max(scale, Math.abs(H[k]))
  return !(Math.abs(determinant(H)) > SINGULAR_EPS * scale * scale * scale)
}

/**
 * Eigen-decomposition of a symmetric n×n matrix (row-major, destroyed) by cyclic Jacobi rotations.
 * Eigenvalues end up on the diagonal of A; eigenvector j is column j of V.
 */
function jacobiEigen(A: Float64Array, n: number, V: Float64Array): void {
  V.fill(0)
  for (let i = 0; i < n; i++) V[i * n + i] = 1
  for (let sweep = 0; sweep < JACOBI_MAX_SWEEPS; sweep++) {
    let off = 0
    let diag = 0
    for (let p = 0; p < n; p++) {
      diag += A[p * n + p] * A[p * n + p]
      for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q]
    }
    if (off <= JACOBI_TOLERANCE * diag || off === 0) return
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q]
        if (apq === 0) continue
        const theta = (A[q * n + q] - A[p * n + p]) / (2 * apq)
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        for (let k = 0; k < n; k++) {
          const akp = A[k * n + p]
          const akq = A[k * n + q]
          A[k * n + p] = c * akp - s * akq
          A[k * n + q] = s * akp + c * akq
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p * n + k]
          const aqk = A[q * n + k]
          A[p * n + k] = c * apk - s * aqk
          A[q * n + k] = s * apk + c * aqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k * n + p]
          const vkq = V[k * n + q]
          V[k * n + p] = c * vkp - s * vkq
          V[k * n + q] = s * vkp + c * vkq
        }
      }
    }
  }
}

/**
 * Least-squares homography (normalised DLT) mapping src[i] → dst[i] for N ≥ 4 pairs; null when the
 * configuration does not determine a unique homography (fewer than 4 points in general position).
 */
export function homographyFromPoints(src: readonly Point[], dst: readonly Point[]): Homography | null {
  if (src.length !== dst.length) throw new Error('homographyFromPoints needs equally many src and dst points')
  if (src.length < 4) return null
  const ts = normalization(src)
  const td = normalization(dst)
  if (ts === null || td === null) return null
  const AtA = new Float64Array(81)
  const r1 = new Float64Array(9)
  const r2 = new Float64Array(9)
  for (let i = 0; i < src.length; i++) {
    const x = ts[0] * src[i].x + ts[1]
    const y = ts[0] * src[i].y + ts[2]
    const u = td[0] * dst[i].x + td[1]
    const v = td[0] * dst[i].y + td[2]
    r1.set([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    r2.set([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    for (let a = 0; a < 9; a++) {
      for (let c = a; c < 9; c++) AtA[a * 9 + c] += r1[a] * r1[c] + r2[a] * r2[c]
    }
  }
  for (let a = 0; a < 9; a++) for (let c = 0; c < a; c++) AtA[a * 9 + c] = AtA[c * 9 + a]
  const V = new Float64Array(81)
  jacobiEigen(AtA, 9, V)
  let lo = 0
  let hi = 0
  for (let i = 1; i < 9; i++) {
    if (AtA[i * 10] < AtA[lo * 10]) lo = i
    if (AtA[i * 10] > AtA[hi * 10]) hi = i
  }
  let second = Infinity
  for (let i = 0; i < 9; i++) if (i !== lo) second = Math.min(second, AtA[i * 10])
  if (!(second > DEGENERATE_EIG_RATIO * AtA[hi * 10])) return null
  const h = new Float64Array(9)
  for (let k = 0; k < 9; k++) h[k] = V[k * 9 + lo]
  return isSingular(h) ? null : denormalize(h, ts, td)
}

export function applyHomography(H: Homography, p: Point, out: Point = { x: 0, y: 0 }): Point {
  const w = H[6] * p.x + H[7] * p.y + H[8]
  const x = (H[0] * p.x + H[1] * p.y + H[2]) / w
  out.y = (H[3] * p.x + H[4] * p.y + H[5]) / w
  out.x = x
  return out
}

export function transformPoints(H: Homography, pts: readonly Point[]): Point[] {
  return pts.map((p) => applyHomography(H, p))
}

/** H⁻¹, or null when H is exactly singular (a nearly singular H yields a huge but finite inverse). */
export function invertHomography(H: Homography): Homography | null {
  const [a, b, c, d, e, f, g, h, i] = H
  const A = e * i - f * h
  const B = -(d * i - f * g)
  const C = d * h - e * g
  const det = a * A + b * B + c * C
  if (det === 0 || !Number.isFinite(1 / det)) return null
  const inv = Float64Array.of(
    A, -(b * i - c * h), b * f - c * e,
    B, a * i - c * g, -(a * f - c * d),
    C, -(a * h - b * g), a * e - b * d,
  )
  for (let k = 0; k < 9; k++) inv[k] /= det
  return normalize(inv)
}

/** A·B: the transform that applies B first, then A. */
export function composeHomography(A: Homography, B: Homography): Homography {
  const out = new Float64Array(9)
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] = A[r * 3] * B[c] + A[r * 3 + 1] * B[3 + c] + A[r * 3 + 2] * B[6 + c]
    }
  }
  return normalize(out)
}

export function applyAffine(A: Affine, p: Point, out: Point = { x: 0, y: 0 }): Point {
  const x = A[0] * p.x + A[1] * p.y + A[2]
  out.y = A[3] * p.x + A[4] * p.y + A[5]
  out.x = x
  return out
}

export function affineToHomography(A: Affine): Homography {
  return Float64Array.of(A[0], A[1], A[2], A[3], A[4], A[5], 0, 0, 1)
}

/** Least-squares affine map src[i] → dst[i] (N ≥ 3); null when the source points are collinear. */
export function estimateAffine(src: readonly Point[], dst: readonly Point[]): Affine | null {
  const n = src.length
  if (n !== dst.length) throw new Error('estimateAffine needs equally many src and dst points')
  if (n < 3) return null
  let mx = 0
  let my = 0
  let mu = 0
  let mv = 0
  for (let i = 0; i < n; i++) {
    mx += src[i].x
    my += src[i].y
    mu += dst[i].x
    mv += dst[i].y
  }
  mx /= n
  my /= n
  mu /= n
  mv /= n
  let sxx = 0
  let sxy = 0
  let syy = 0
  let sxu = 0
  let syu = 0
  let sxv = 0
  let syv = 0
  for (let i = 0; i < n; i++) {
    const x = src[i].x - mx
    const y = src[i].y - my
    const u = dst[i].x - mu
    const v = dst[i].y - mv
    sxx += x * x
    sxy += x * y
    syy += y * y
    sxu += x * u
    syu += y * u
    sxv += x * v
    syv += y * v
  }
  const det = sxx * syy - sxy * sxy
  if (!(Math.abs(det) > SINGULAR_EPS * (sxx + syy) * (sxx + syy))) return null
  const a = (sxu * syy - syu * sxy) / det
  const b = (syu * sxx - sxu * sxy) / det
  const c = (sxv * syy - syv * sxy) / det
  const d = (syv * sxx - sxv * sxy) / det
  return Float64Array.of(a, b, mu - a * mx - b * my, c, d, mv - c * mx - d * my)
}

export interface Similarity {
  /** Uniform scale factor. */
  scale: number
  /** Rotation in radians; positive turns +x towards +y, i.e. clockwise on screen (y down). */
  angle: number
  tx: number
  ty: number
  /** The same transform as a 2×3 affine: x' = s·(cos·x − sin·y) + tx, y' = s·(sin·x + cos·y) + ty. */
  matrix: Affine
}

/**
 * Least-squares similarity (rotation, uniform scale, translation; no reflection) mapping src[i] →
 * dst[i] for N ≥ 2 — the 2-D closed form of Umeyama's method. Null when all source points coincide.
 */
export function estimateSimilarity(src: readonly Point[], dst: readonly Point[]): Similarity | null {
  const n = src.length
  if (n !== dst.length) throw new Error('estimateSimilarity needs equally many src and dst points')
  if (n < 2) return null
  let mx = 0
  let my = 0
  let mu = 0
  let mv = 0
  for (let i = 0; i < n; i++) {
    mx += src[i].x
    my += src[i].y
    mu += dst[i].x
    mv += dst[i].y
  }
  mx /= n
  my /= n
  mu /= n
  mv /= n
  let dot = 0
  let cross = 0
  let norm = 0
  for (let i = 0; i < n; i++) {
    const x = src[i].x - mx
    const y = src[i].y - my
    const u = dst[i].x - mu
    const v = dst[i].y - mv
    dot += x * u + y * v
    cross += x * v - y * u
    norm += x * x + y * y
  }
  if (!(norm > 0)) return null
  const sc = dot / norm // s·cos
  const ss = cross / norm // s·sin
  const tx = mu - (sc * mx - ss * my)
  const ty = mv - (ss * mx + sc * my)
  return {
    scale: Math.hypot(sc, ss),
    angle: Math.atan2(ss, sc),
    tx,
    ty,
    matrix: Float64Array.of(sc, -ss, tx, ss, sc, ty),
  }
}
