/**
 * Connected components of binary masks: 8-connected labelling with per-component statistics, and the
 * complementary 4-connected "outer background" flood fill.
 *
 * Segmentation produces a foreground mask; its 8-connected components are the candidate pieces, and
 * their area, bounding box and centroid drive filtering and tracking. 8-connectivity for the foreground
 * pairs with 4-connectivity for the background (the digital-topology duality OpenCV's contour tracing
 * also assumes), which is why the outer-background fill below is 4-connected.
 *
 * Labelling is the classic two-pass scan with a union–find over provisional labels (decision tree over
 * the already-visited W, NW, N, NE neighbours, so at most one union per pixel). Unions keep the smaller
 * root, which makes final labels follow the raster order of each component's first pixel (its topmost,
 * then leftmost pixel). Statistics and label sets equal cv2.connectedComponentsWithStats(connectivity=8)
 * (golden-tested); only the numbering could differ from OpenCV's block-based scan on some inputs, so
 * callers must not rely on OpenCV's numbering.
 */

import type { Mask, Point, Rect } from '../types.ts'
import { ensureMask, scratchF64, scratchI32 } from './create.ts'

export interface ComponentStats {
  /** 1-based label in `labels`. */
  label: number
  /** Pixel count. */
  area: number
  bbox: Rect
  centroid: Point
  /** Row-major index of the component's first pixel in raster order (its topmost, then leftmost pixel). */
  start: number
}

export interface Components {
  width: number
  height: number
  /** 0 = background, 1..count = component. */
  labels: Int32Array
  count: number
  /** stats[k - 1] describes label k. */
  stats: ComponentStats[]
}

function find(parent: Int32Array, p: number): number {
  while (parent[p] !== p) {
    parent[p] = parent[parent[p]]
    p = parent[p]
  }
  return p
}

/** Merges the sets of a and b, keeping the smaller root; returns that root. */
function union(parent: Int32Array, a: number, b: number): number {
  const ra = find(parent, a)
  const rb = find(parent, b)
  if (ra < rb) {
    parent[rb] = ra
    return ra
  }
  parent[ra] = rb
  return rb
}

/** Labels the 8-connected components of the nonzero pixels of `mask`. */
export function labelComponents(mask: Mask, outLabels?: Int32Array): Components {
  const { width: w, height: h, data: m } = mask
  const n = w * h
  if (outLabels !== undefined && outLabels.length !== n) {
    throw new Error(`labels output has ${outLabels.length} entries, expected ${n}`)
  }
  const lab = outLabels ?? new Int32Array(n)
  // New provisional labels are pairwise non-adjacent pixels, so at most ceil(w/2)·ceil(h/2) of them.
  const maxProvisional = ((w + 1) >> 1) * ((h + 1) >> 1) + 1
  const parent = scratchI32('components.parent', maxProvisional)
  let next = 1

  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) {
      const i = row + x
      if (m[i] === 0) {
        lab[i] = 0
        continue
      }
      const hasUp = y > 0
      const n0 = hasUp ? lab[i - w] : 0
      if (n0 !== 0) {
        // N touches W, NW and NE, which are therefore already in N's set.
        lab[i] = n0
        continue
      }
      const ne = hasUp && x + 1 < w ? lab[i - w + 1] : 0
      const west = x > 0 ? lab[i - 1] : 0
      if (west !== 0) {
        lab[i] = ne !== 0 ? union(parent, west, ne) : west
        continue
      }
      const nw = hasUp && x > 0 ? lab[i - w - 1] : 0
      if (nw !== 0) {
        lab[i] = ne !== 0 ? union(parent, nw, ne) : nw
        continue
      }
      if (ne !== 0) {
        lab[i] = ne
        continue
      }
      parent[next] = next
      lab[i] = next++
    }
  }

  // Flatten: parents always have smaller indices, so one ascending sweep resolves every chain.
  const final = scratchI32('components.final', next)
  let count = 0
  for (let p = 1; p < next; p++) final[p] = parent[p] === p ? ++count : final[parent[p]]

  const acc = scratchF64('components.acc', (count + 1) * 7)
  acc.fill(0, 0, (count + 1) * 7)
  // Per label: start, minX, minY, maxX, maxY, sumX, sumY.
  for (let k = 1; k <= count; k++) {
    acc[k * 7 + 1] = w
    acc[k * 7 + 2] = h
    acc[k * 7 + 3] = -1
    acc[k * 7 + 4] = -1
  }
  const area = scratchI32('components.area', count + 1)
  area.fill(0, 0, count + 1)
  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) {
      const i = row + x
      if (lab[i] === 0) continue
      const k = final[lab[i]]
      lab[i] = k
      const o = k * 7
      if (area[k]++ === 0) acc[o] = i
      if (x < acc[o + 1]) acc[o + 1] = x
      if (y < acc[o + 2]) acc[o + 2] = y
      if (x > acc[o + 3]) acc[o + 3] = x
      if (y > acc[o + 4]) acc[o + 4] = y
      acc[o + 5] += x
      acc[o + 6] += y
    }
  }

  const stats: ComponentStats[] = []
  for (let k = 1; k <= count; k++) {
    const o = k * 7
    stats.push({
      label: k,
      area: area[k],
      bbox: { x: acc[o + 1], y: acc[o + 2], width: acc[o + 3] - acc[o + 1] + 1, height: acc[o + 4] - acc[o + 2] + 1 },
      centroid: { x: acc[o + 5] / area[k], y: acc[o + 6] / area[k] },
      start: acc[o],
    })
  }
  return { width: w, height: h, labels: lab, count, stats }
}

/** Binary mask (0/1) of one labelled component. */
export function componentMask(components: Components, label: number, out?: Mask): Mask {
  const { width: w, height: h, labels } = components
  const dst = ensureMask(out, w, h)
  const d = dst.data
  for (let i = 0; i < d.length; i++) d[i] = labels[i] === label ? 1 : 0
  return dst
}

/**
 * Marks (1) every background pixel that is 4-connected to the image frame; 0 elsewhere. Background
 * pixels left unmarked are holes enclosed by foreground.
 */
export function markOuterBackground(mask: Mask, out?: Uint8Array): Uint8Array {
  const { width: w, height: h, data: m } = mask
  const n = w * h
  if (out !== undefined && out.length !== n) throw new Error(`output has ${out.length} entries, expected ${n}`)
  const mark = out ?? new Uint8Array(n)
  mark.fill(0)
  // Scanline (span) flood fill: fill a whole horizontal run at once, then queue one seed per run of
  // fillable pixels directly above and below it. Far fewer stack operations than pixel-by-pixel.
  let stack = scratchI32('components.stack', 1024)
  const fillFrom = (seed: number) => {
    let top = 0
    stack[top++] = seed
    while (top > 0) {
      const i = stack[--top]
      if (m[i] !== 0 || mark[i] !== 0) continue
      const y = (i / w) | 0
      const row = y * w
      let l = i - row
      let r = l
      while (l > 0 && m[row + l - 1] === 0 && mark[row + l - 1] === 0) l--
      while (r + 1 < w && m[row + r + 1] === 0 && mark[row + r + 1] === 0) r++
      mark.fill(1, row + l, row + r + 1)
      for (let ny = y - 1; ny <= y + 1; ny += 2) {
        if (ny < 0 || ny >= h) continue
        const nrow = ny * w
        let open = false
        for (let x = l; x <= r; x++) {
          const j = nrow + x
          if (m[j] === 0 && mark[j] === 0) {
            if (!open) {
              if (top === stack.length) {
                const grown = scratchI32('components.stack', stack.length * 2)
                grown.set(stack.subarray(0, top))
                stack = grown
              }
              stack[top++] = j
              open = true
            }
          } else {
            open = false
          }
        }
      }
    }
  }
  for (let x = 0; x < w; x++) {
    if (m[x] === 0 && mark[x] === 0) fillFrom(x)
    const b = (h - 1) * w + x
    if (m[b] === 0 && mark[b] === 0) fillFrom(b)
  }
  for (let y = 0; y < h; y++) {
    if (m[y * w] === 0 && mark[y * w] === 0) fillFrom(y * w)
    const e = y * w + w - 1
    if (m[e] === 0 && mark[e] === 0) fillFrom(e)
  }
  return mark
}
