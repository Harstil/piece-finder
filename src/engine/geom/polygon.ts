/**
 * Polygon and closed-contour utilities used by shape analysis and drawing.
 *
 * Orientation convention (engine-wide, see types.ts): image coordinates, y down, and contours run
 * CLOCKWISE on screen. With y down, the shoelace sum ½ Σ (xᵢ yᵢ₊₁ − xᵢ₊₁ yᵢ) is POSITIVE for such a
 * contour, so signedArea > 0 ⇔ clockwise. Likewise a turn is positive when it bends clockwise on screen
 * (a right turn while walking the contour), so the convex corners of a clockwise contour have positive
 * turning angles and the angles of a simple closed contour sum to +2π.
 *
 * Contents: signed area, perimeter, orientation fix, closed Douglas–Peucker simplification (a port of
 * OpenCV's approxPolyDP, identical output on the golden cases), convex hull (monotone chain),
 * point-in-polygon (even-odd), rasterisation to a Mask, uniform arc-length resampling, turning angle at
 * an arc-length scale, and point-to-segment distance.
 */

import { ensureMask, scratchF64, scratchI32 } from '../image/create.ts'
import type { Mask, Point } from '../types.ts'

/** Shoelace area; positive for clockwise-on-screen polygons (y down). */
export function signedArea(poly: readonly Point[]): number {
  const n = poly.length
  let s = 0
  for (let i = 0, j = n - 1; i < n; j = i++) s += poly[j].x * poly[i].y - poly[i].x * poly[j].y
  return s / 2
}

export function isClockwise(poly: readonly Point[]): boolean {
  return signedArea(poly) > 0
}

/** Closed (default) or open polyline length. */
export function perimeter(poly: readonly Point[], closed = true): number {
  const n = poly.length
  let s = 0
  for (let i = 1; i < n; i++) s += Math.hypot(poly[i].x - poly[i - 1].x, poly[i].y - poly[i - 1].y)
  if (closed && n > 1) s += Math.hypot(poly[0].x - poly[n - 1].x, poly[0].y - poly[n - 1].y)
  return s
}

/**
 * The polygon clockwise on screen: returned as is when it already is (or has zero area), otherwise a
 * reversed copy that keeps the same first point.
 */
export function orientClockwise<T extends Point>(poly: readonly T[]): T[] {
  if (signedArea(poly) >= 0) return poly as T[]
  return poly.length <= 1 ? poly.slice() : [poly[0], ...poly.slice(1).reverse()]
}

/** Distance from p to the segment ab (to the point a when a = b). */
export function pointSegmentDistance(p: Point, a: Point, b: Point): number {
  const vx = b.x - a.x
  const vy = b.y - a.y
  const len2 = vx * vx + vy * vy
  let t = len2 > 0 ? ((p.x - a.x) * vx + (p.y - a.y) * vy) / len2 : 0
  t = t < 0 ? 0 : t > 1 ? 1 : t
  return Math.hypot(p.x - (a.x + t * vx), p.y - (a.y + t * vy))
}

/**
 * Simplifies a closed contour with the Douglas–Peucker algorithm: every dropped point lies within
 * `epsilon` of the simplified outline. Port of OpenCV's approxPolyDP(closed = true) as of 4.x/5.x,
 * including its choice of the initial split (approximate diameter by three farthest-point passes), its
 * distance-to-segment (not to-line) split criterion and its final pass that drops near-collinear
 * vertices, so the output equals cv2.approxPolyDP point for point (golden-tested on 88 contours).
 * The result keeps the input's orientation and contains copies of input points.
 */
export function approxPolyDP(contour: readonly Point[], epsilon: number): Point[] {
  const count = contour.length
  if (count === 0) return []
  const eps = epsilon * epsilon
  const dstIdx: number[] = []
  const stack: number[] = []

  // 1. Approximately the two farthest points of the contour.
  let rightStart = 0
  let pos = 0
  let startIdx = 0
  let leEps = false
  for (let iter = 0; iter < 3; iter++) {
    pos = (pos + rightStart) % count
    startIdx = pos
    const sp = contour[pos]
    pos = (pos + 1) % count
    let maxDist = 0
    for (let j = 1; j < count; j++) {
      const p = contour[pos]
      pos = (pos + 1) % count
      const dx = p.x - sp.x
      const dy = p.y - sp.y
      const dist = dx * dx + dy * dy
      if (dist > maxDist) {
        maxDist = dist
        rightStart = j
      }
    }
    leEps = maxDist <= eps
  }

  // 2. Seed the stack with the two halves (the first half is processed first).
  if (!leEps) {
    const first = pos % count
    const far = (rightStart + first) % count
    stack.push(far, first) // right slice: far → first
    stack.push(first, far) // slice: first → far
  } else {
    dstIdx.push(startIdx)
  }

  // 3. Recursive splitting.
  while (stack.length > 0) {
    const sliceEnd = stack.pop()!
    const sliceStart = stack.pop()!
    const end = contour[sliceEnd]
    const start = contour[sliceStart]
    let p = (sliceStart + 1) % count
    let split = -1
    if (p !== sliceEnd) {
      const dx = end.x - start.x
      const dy = end.y - start.y
      const len2 = dx * dx + dy * dy
      // Squared distance to the SEGMENT (not the infinite line), times len2 to stay division-free —
      // what current OpenCV computes. A degenerate chord (start = end, possible on arbitrary input,
      // OpenCV asserts) falls back to the plain squared distance from the start point.
      const scale = len2 > 0 ? len2 : 1
      let maxDist = 0
      while (p !== sliceEnd) {
        const q = contour[p]
        p = (p + 1) % count
        const proj = (q.x - start.x) * dx + (q.y - start.y) * dy
        let dist: number
        if (len2 === 0 || proj < 0) {
          dist = ((q.x - start.x) * (q.x - start.x) + (q.y - start.y) * (q.y - start.y)) * scale
        } else if (proj > len2) {
          dist = ((q.x - end.x) * (q.x - end.x) + (q.y - end.y) * (q.y - end.y)) * len2
        } else {
          const cross = (q.y - start.y) * dx - (q.x - start.x) * dy
          dist = cross * cross
        }
        if (dist > maxDist) {
          maxDist = dist
          split = (p + count - 1) % count
        }
      }
      leEps = maxDist <= eps * scale
    } else {
      leEps = true
    }
    if (leEps) {
      dstIdx.push(sliceStart)
    } else {
      stack.push(split, sliceEnd)
      stack.push(sliceStart, split)
    }
  }

  // 4. Drop vertices lying on (almost) straight runs, in place, exactly as OpenCV does.
  const dst: Point[] = dstIdx.map((i) => ({ x: contour[i].x, y: contour[i].y }))
  const n = dst.length
  let newCount = n
  let rd = n - 1
  let startPt = dst[rd]
  rd = (rd + 1) % n
  let wpos = rd
  let pt = dst[rd]
  rd = (rd + 1) % n
  for (let i = 0; i < n && newCount > 2; i++) {
    const endPt = dst[rd]
    rd = (rd + 1) % n
    const dx = endPt.x - startPt.x
    const dy = endPt.y - startPt.y
    const dist = Math.abs((pt.x - startPt.x) * dy - (pt.y - startPt.y) * dx)
    const inner = (pt.x - startPt.x) * (endPt.x - pt.x) + (pt.y - startPt.y) * (endPt.y - pt.y)
    if (dist * dist <= 0.5 * eps * (dx * dx + dy * dy) && dx !== 0 && dy !== 0 && inner >= 0) {
      newCount--
      dst[wpos] = startPt = endPt
      wpos = (wpos + 1) % n
      pt = dst[rd]
      rd = (rd + 1) % n
      i++
      continue
    }
    dst[wpos] = startPt = pt
    wpos = (wpos + 1) % n
    pt = endPt
  }
  return dst.slice(0, newCount)
}

/**
 * Convex hull by Andrew's monotone chain: clockwise on screen, starting at the leftmost (then topmost)
 * point, without collinear points. Fewer than three distinct points are returned as they are (deduped).
 */
export function convexHull(points: readonly Point[]): Point[] {
  const pts = points.slice().sort((a, b) => a.x - b.x || a.y - b.y)
  const uniq: Point[] = []
  for (const p of pts) {
    const last = uniq[uniq.length - 1]
    if (last === undefined || last.x !== p.x || last.y !== p.y) uniq.push(p)
  }
  if (uniq.length < 3) return uniq.map((p) => ({ x: p.x, y: p.y }))
  // cross > 0 ⇔ o→a→b turns clockwise on screen (y down).
  const cross = (o: Point, a: Point, b: Point) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
  const hull: Point[] = []
  // Upper chain (smaller y on screen) left → right, then lower chain right → left: clockwise on screen.
  for (const p of uniq) {
    while (hull.length >= 2 && cross(hull[hull.length - 2], hull[hull.length - 1], p) <= 0) hull.pop()
    hull.push(p)
  }
  const upperLen = hull.length + 1
  for (let i = uniq.length - 2; i >= 0; i--) {
    const p = uniq[i]
    while (hull.length >= upperLen && cross(hull[hull.length - 2], hull[hull.length - 1], p) <= 0) hull.pop()
    hull.push(p)
  }
  hull.pop()
  return hull.map((p) => ({ x: p.x, y: p.y }))
}

/** Even-odd rule; points exactly on an edge may land on either side. */
export function pointInPolygon(p: Point, poly: readonly Point[]): boolean {
  let inside = false
  const n = poly.length
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const a = poly[i]
    const b = poly[j]
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside
  }
  return inside
}

/**
 * Rasterises a polygon (even-odd rule) into a 0/1 mask: a pixel is set when its centre lies inside.
 * Half-open at the boundary (a centre exactly on a left/top edge is in, on a right/bottom edge is out),
 * so polygons that share an edge never both claim a pixel. Scanline with per-row crossing lists.
 */
export function rasterizePolygon(poly: readonly Point[], width: number, height: number, out?: Mask): Mask {
  const dst = ensureMask(out, width, height)
  const d = dst.data
  d.fill(0)
  const n = poly.length
  if (n < 3) return dst
  // Count crossings per row: edge (a, b) crosses row y when min(ay, by) ≤ y < max(ay, by).
  const rowCount = scratchI32('polygon.rowCount', height + 1)
  rowCount.fill(0, 0, height + 1)
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const y0 = Math.max(0, Math.ceil(Math.min(poly[i].y, poly[j].y)))
    const y1 = Math.min(height, Math.ceil(Math.max(poly[i].y, poly[j].y)))
    for (let y = y0; y < y1; y++) rowCount[y + 1]++
  }
  for (let y = 0; y < height; y++) rowCount[y + 1] += rowCount[y]
  const total = rowCount[height]
  const xs = scratchF64('polygon.xs', total)
  const fill = scratchI32('polygon.fill', height)
  fill.fill(0, 0, height)
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const a = poly[i]
    const b = poly[j]
    if (a.y === b.y) continue
    const y0 = Math.max(0, Math.ceil(Math.min(a.y, b.y)))
    const y1 = Math.min(height, Math.ceil(Math.max(a.y, b.y)))
    const slope = (b.x - a.x) / (b.y - a.y)
    for (let y = y0; y < y1; y++) xs[rowCount[y] + fill[y]++] = a.x + (y - a.y) * slope
  }
  for (let y = 0; y < height; y++) {
    const s = rowCount[y]
    const e = rowCount[y + 1]
    if (e - s < 2) continue
    const row = xs.subarray(s, e).sort()
    const off = y * width
    for (let k = 0; k + 1 < row.length; k += 2) {
      const x0 = Math.max(0, Math.ceil(row[k]))
      const x1 = Math.min(width, Math.ceil(row[k + 1]))
      for (let x = x0; x < x1; x++) d[off + x] = 1
    }
  }
  return dst
}

/** Cumulative arc length at each vertex of a closed contour; entry n is the full perimeter. */
function arcLengths(contour: readonly Point[]): Float64Array {
  const n = contour.length
  const L = scratchF64('polygon.arc', n + 1)
  L[0] = 0
  for (let i = 1; i <= n; i++) {
    const a = contour[i - 1]
    const b = contour[i % n]
    L[i] = L[i - 1] + Math.hypot(b.x - a.x, b.y - a.y)
  }
  return L
}

/** Point at arc length t ∈ [0, total) of a closed contour; `L` from arcLengths. */
function pointAtArc(contour: readonly Point[], L: Float64Array, t: number, out: Point): Point {
  const n = contour.length
  // Last vertex index with L[i] ≤ t.
  let lo = 0
  let hi = n
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (L[mid] <= t) lo = mid
    else hi = mid
  }
  const a = contour[lo]
  const b = contour[(lo + 1) % n]
  const seg = L[lo + 1] - L[lo]
  const f = seg > 0 ? (t - L[lo]) / seg : 0
  out.x = a.x + (b.x - a.x) * f
  out.y = a.y + (b.y - a.y) * f
  return out
}

/** `count` points evenly spaced by arc length along a closed contour, starting at contour[0]. */
export function resampleClosed(contour: readonly Point[], count: number): Point[] {
  const n = contour.length
  if (n === 0 || count <= 0) return []
  const L = arcLengths(contour)
  const total = L[n]
  const out: Point[] = []
  for (let k = 0; k < count; k++) out.push(pointAtArc(contour, L, total > 0 ? (k * total) / count : 0, { x: 0, y: 0 }))
  return out
}

/**
 * Discrete turning angle at every vertex of a closed contour, measured at arc-length `scale`: the
 * signed angle between the chord arriving from the point `scale` behind the vertex and the chord
 * leaving to the point `scale` ahead, in radians in (−π, π]. Positive = the contour bends clockwise on
 * screen there (a convex corner of a clockwise contour). Curvature ≈ angle / scale (for a circle of
 * radius R and small scale the angle is scale / R). `scale` should stay well below half the perimeter.
 */
export function turningAngles(contour: readonly Point[], scale: number, out?: Float64Array): Float64Array {
  const n = contour.length
  if (out !== undefined && out.length !== n) throw new Error(`output has ${out.length} entries, expected ${n}`)
  const angles = out ?? new Float64Array(n)
  if (n === 0) return angles
  const L = arcLengths(contour)
  const total = L[n]
  const back: Point = { x: 0, y: 0 }
  const ahead: Point = { x: 0, y: 0 }
  for (let i = 0; i < n; i++) {
    if (!(total > 0)) {
      angles[i] = 0
      continue
    }
    let tb = (L[i] - scale) % total
    if (tb < 0) tb += total
    const ta = (L[i] + scale) % total
    pointAtArc(contour, L, tb, back)
    pointAtArc(contour, L, ta, ahead)
    const p = contour[i]
    const ux = p.x - back.x
    const uy = p.y - back.y
    const vx = ahead.x - p.x
    const vy = ahead.y - p.y
    angles[i] = Math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
  }
  return angles
}
