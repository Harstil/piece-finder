/**
 * Live-pipeline check: `node eval/pipeline-bench.ts v1/val500 [scenes]`
 *
 * Feeds dataset scenes, shrunk to the app's processing size, through stepFrame several times each (as if
 * the user hovered) and prints per-stage timings plus, for each tracked piece, whether its locked/leading
 * cell is the ground truth. This is the closest Node gets to the phone's live loop.
 */

import { fitSize, resizeAreaRGBA } from '../src/engine/image/index.ts'
import { buildReference } from '../src/engine/index.ts'
import { createSession, stepFrame } from '../src/engine/pipeline.ts'
import { isMatchable, loadPuzzle, loadScene, openDataset, scenesByPuzzle, toQuad } from './lib/dataset.ts'

/** The app's processing size (camera frames are shrunk to this long side before the worker). */
const PROCESS_LONG_SIDE = 1280
const FRAMES_PER_SCENE = 3

const name = process.argv[2] ?? 'v1/val500'
const sceneLimit = Number(process.argv[3] ?? 6)
const ds = await openDataset(name)
const byPuzzle = await scenesByPuzzle(ds)
let right = 0
let wrong = 0
let missing = 0
const seg: number[] = []
const match: number[] = []
let done = 0
for (const puzzleId of ds.meta.puzzles) {
  if (done >= sceneLimit) break
  const pz = await loadPuzzle(ds, puzzleId)
  const t0 = performance.now()
  const ref = buildReference(pz.reference, toQuad(pz.ref.referenceCorners), { cols: pz.ref.cols, rows: pz.ref.rows })
  console.log(`${puzzleId}: reference ${pz.ref.cols}x${pz.ref.rows} built in ${(performance.now() - t0).toFixed(0)} ms`)
  for (const sceneId of byPuzzle.get(puzzleId) ?? []) {
    if (done >= sceneLimit) break
    done++
    const scene = await loadScene(ds, sceneId)
    const size = fitSize(scene.frame.width, scene.frame.height, PROCESS_LONG_SIDE)
    const frame = resizeAreaRGBA(scene.frame, size.width, size.height)
    const k = size.width / scene.frame.width
    const session = createSession(ref)
    let result = stepFrame(session, frame, 0)
    for (let f = 1; f < FRAMES_PER_SCENE; f++) result = stepFrame(session, frame, f)
    seg.push(result.timingsMs.segment)
    match.push(result.timingsMs.match)
    const gt = scene.json.pieces.filter((p) => isMatchable(p) && p.visibleFraction >= 0.98)
    for (const p of gt) {
      const cx = (p.corners.reduce((s, q) => s + q[0], 0) / 4) * k
      const cy = (p.corners.reduce((s, q) => s + q[1], 0) / 4) * k
      const t = result.tracks.find((tr) => {
        const xs = tr.contour.map((q) => q.x)
        const ys = tr.contour.map((q) => q.y)
        return cx > Math.min(...xs) && cx < Math.max(...xs) && cy > Math.min(...ys) && cy < Math.max(...ys)
      })
      if (t === undefined || t.candidates.length === 0) missing++
      else if (t.candidates[0].cell === p.cell) right++
      else wrong++
    }
    console.log(`  ${sceneId} ${scene.json.background}: ${result.tracks.length} tracked, segment ${result.timingsMs.segment.toFixed(0)} ms, match ${result.timingsMs.match.toFixed(0)} ms (${result.timingsMs.matched} matched), locked ${result.tracks.filter((t) => t.locked).length}`)
  }
}
const med = (xs: number[]): string => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)].toFixed(0)
console.log(`fully visible pieces: right ${right}, wrong ${wrong}, not detected ${missing}; median segment ${med(seg)} ms, match ${med(match)} ms`)
