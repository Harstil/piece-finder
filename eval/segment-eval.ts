/**
 * Segmentation evaluation: `node eval/segment-eval.ts v1/val500 [--limit 20] [--match]`
 *
 * Runs the live-frame segmenter (src/engine/segment.ts) on dataset scenes and compares its pieces with
 * the ground-truth instances: a detection is a hit when it overlaps one matchable ground-truth piece
 * with IoU ≥ HIT_IOU (both rasterised at frame resolution; the ground truth is the instance mask, which
 * includes the side wall like a real segmenter's). Reports recall over matchable pieces, precision over
 * detections, per table background, plus timing. With --match it also runs the full chain on every hit
 * (canonicalize + matchPiece against the reference built from the GT corners) and reports top-1 —
 * the end-to-end number the app will actually deliver.
 */

import { rasterizePolygon } from '../src/engine/geom/index.ts'
import { buildReference, canonicalize, matchPiece, type ReferenceModel } from '../src/engine/index.ts'
import { segmentPieces } from '../src/engine/segment.ts'
import { isMatchable, loadPuzzle, loadScene, openDataset, scenesByPuzzle, toQuad } from './lib/dataset.ts'

/** Pieces at least this visible (not occluded, not cut by the frame) are "fully visible". */
const FULLY_VISIBLE = 0.98
/** Overlap needed to count a detection as finding a piece. Guessed: corners must be usable. */
const HIT_IOU = 0.7

interface Tally {
  gt: number
  found: number
  det: number
  hits: number
  matched: number
  right: number
  ms: number[]
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2)
  const li = argv.indexOf('--limit')
  const limit = li >= 0 ? Number(argv[li + 1]) : Infinity
  const doMatch = argv.includes('--match')
  const names = argv.filter((a, i) => !a.startsWith('--') && argv[i - 1] !== '--limit')
  for (const name of names) {
    const ds = await openDataset(name)
    const byBg = new Map<string, Tally>()
    const confs: { c: number; hit: boolean; full: boolean }[] = []
    let fullGt = 0
    const total: Tally = { gt: 0, found: 0, det: 0, hits: 0, matched: 0, right: 0, ms: [] }
    const byPuzzle = await scenesByPuzzle(ds)
    let done = 0
    for (const puzzleId of ds.meta.puzzles) {
      let ref: ReferenceModel | null = null
      if (doMatch) {
        const pz = await loadPuzzle(ds, puzzleId)
        ref = buildReference(pz.reference, toQuad(pz.ref.referenceCorners), { cols: pz.ref.cols, rows: pz.ref.rows })
      }
      for (const sceneId of byPuzzle.get(puzzleId) ?? []) {
        if (done >= limit) break
        done++
        const scene = await loadScene(ds, sceneId)
        const t0 = performance.now()
        const seg = segmentPieces(scene.frame, { minConfidence: 0 })
        const ms = performance.now() - t0
        const bg = scene.json.background
        const t = byBg.get(bg) ?? { gt: 0, found: 0, det: 0, hits: 0, matched: 0, right: 0, ms: [] }
        byBg.set(bg, t)
        const { width: w, height: h } = scene.frame
        const labels = scene.labels.data
        const gtArea = new Map<number, number>()
        for (let i = 0; i < labels.length; i++) if (labels[i] > 0) gtArea.set(labels[i], (gtArea.get(labels[i]) ?? 0) + 1)
        const matchable = new Map(scene.json.pieces.filter(isMatchable).map((p) => [p.index, p]))
        fullGt += scene.json.pieces.filter((p) => isMatchable(p) && p.visibleFraction >= FULLY_VISIBLE).length
        const foundGt = new Set<number>()
        let hits = 0
        let matched = 0
        let right = 0
        for (const d of seg.pieces) {
          const m = rasterizePolygon(d.contour, w, h)
          // Overlap with each ground-truth instance.
          const inter = new Map<number, number>()
          let dArea = 0
          for (let i = 0; i < m.data.length; i++) {
            if (m.data[i] === 0) continue
            dArea++
            if (labels[i] > 0) inter.set(labels[i], (inter.get(labels[i]) ?? 0) + 1)
          }
          let bestIdx = 0
          let bestIou = 0
          for (const [idx, ov] of inter) {
            const iou = ov / (dArea + (gtArea.get(idx) ?? 0) - ov)
            if (iou > bestIou) {
              bestIou = iou
              bestIdx = idx
            }
          }
          const gt = matchable.get(bestIdx)
          const isHit = bestIou >= HIT_IOU && gt !== undefined && !foundGt.has(bestIdx)
          confs.push({ c: d.score, hit: isHit, full: isHit && (gt?.visibleFraction ?? 0) >= FULLY_VISIBLE })
          if (isHit) {
            hits++
            foundGt.add(bestIdx)
            if (ref !== null) {
              matched++
              const piece = canonicalize(scene.frame, d.contour, d.shape)
              const r = matchPiece(ref, piece)
              if (r.candidates[0]?.cell === gt.cell) right++
            }
          }
        }
        for (const x of [t, total]) {
          x.gt += matchable.size
          x.found += foundGt.size
          x.det += seg.pieces.length
          x.hits += hits
          x.matched += matched
          x.right += right
          x.ms.push(ms)
        }
      }
    }
    const pct = (a: number, b: number): string => (b === 0 ? '—' : `${((100 * a) / b).toFixed(1)}%`)
    const med = (xs: number[]): string => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)]?.toFixed(0) ?? '—'
    const line = (label: string, x: Tally): string =>
      `${label.padEnd(24)} recall ${pct(x.found, x.gt).padStart(6)} (${x.found}/${x.gt})  precision ${pct(x.hits, x.det).padStart(6)} (${x.hits}/${x.det})` +
      (doMatch ? `  end-to-end top-1 ${pct(x.right, x.gt).padStart(6)} (of found: ${pct(x.right, x.matched)})` : '') +
      `  ${med(x.ms)} ms/frame`
    console.log(line(`${name} ALL`, total))
    for (const [bg, x] of [...byBg.entries()].sort()) console.log(line(`  ${bg}`, x))
    for (const th of [0, 0.0001, 0.001, 0.003, 0.01, 0.02, 0.05]) {
      const kept = confs.filter((x) => x.c >= th)
      const hits = kept.filter((x) => x.hit).length
      const full = kept.filter((x) => x.full).length
      console.log(`  conf ≥ ${String(th).padEnd(6)} precision ${pct(hits, kept.length).padStart(6)}  recall(all matchable) ${pct(hits, total.gt).padStart(6)}  recall(fully visible) ${pct(full, fullGt).padStart(6)} (${full}/${fullGt})`)
    }
  }
}

await main()
