/**
 * Dataset loading for the evaluation scripts (docs/DATASET.md), shared by eval/run.ts and
 * eval/shape-eval.ts.
 *
 * Images are decoded with sharp into the engine's own image types, so the scripts exercise exactly
 * the code the phone runs. Decoding a 3000–4000 px reference or a 1080×1920 scene takes tens of ms,
 * so callers load one puzzle at a time and let it go before the next.
 */

import { readFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import sharp from 'sharp'
import type { Mask, Point, Quad, RGBAImage, SideKind } from '../../src/engine/types.ts'

export type XY = [number, number]

export interface MetaJson {
  seed: number
  split: string
  puzzles: string[]
  scenes: string[]
}

export interface ReferenceJson {
  cols: number
  rows: number
  motifSize: XY
  referenceCorners: [XY, XY, XY, XY]
  cut: 'grid' | 'irregular'
  referenceKind?: string
  motifStyle?: string
  piecesNominal?: number
}

export interface ScenePieceJson {
  index: number
  pieceId: number
  col: number
  row: number
  cell: number
  upAngleDeg: number
  corners: [XY, XY, XY, XY]
  outline: XY[]
  visibleFraction: number
  faceUp: boolean
}

export interface SceneJson {
  puzzleId: string
  width: number
  height: number
  background: string
  pieces: ScenePieceJson[]
}

export interface PieceJson {
  id: number
  col: number
  row: number
  cell: number
  sides: [SideKind, SideKind, SideKind, SideKind]
}

export const toPoint = ([x, y]: XY): Point => ({ x, y })
export const toQuad = (c: readonly XY[]): Quad => [toPoint(c[0]), toPoint(c[1]), toPoint(c[2]), toPoint(c[3])]

export async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(await readFile(path, 'utf8')) as T
}

export async function loadRGBA(path: string): Promise<RGBAImage> {
  const { data, info } = await sharp(path).ensureAlpha().raw().toBuffer({ resolveWithObject: true })
  return { width: info.width, height: info.height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) }
}

/** The 8-bit instance map: 0 = background, k = scene piece index k. */
export async function loadLabels(path: string): Promise<Mask> {
  const { data, info } = await sharp(path).extractChannel(0).raw().toBuffer({ resolveWithObject: true })
  return { width: info.width, height: info.height, data: new Uint8Array(data.buffer, data.byteOffset, data.length) }
}

export interface DatasetHandle {
  root: string
  name: string
  meta: MetaJson
}

/** Opens `datasets/<name>`; throws when the set is incomplete (meta.json is written last). */
export async function openDataset(name: string, base = 'datasets'): Promise<DatasetHandle> {
  const root = join(base, name)
  const metaPath = join(root, 'meta.json')
  if (!existsSync(metaPath)) throw new Error(`${root}: no meta.json (missing or still being generated)`)
  return { root, name, meta: await readJson<MetaJson>(metaPath) }
}

export interface PuzzleData {
  id: string
  ref: ReferenceJson
  pieces: PieceJson[]
  reference: RGBAImage
}

export async function loadPuzzle(ds: DatasetHandle, id: string): Promise<PuzzleData> {
  const dir = join(ds.root, 'puzzles', id)
  return {
    id,
    ref: await readJson<ReferenceJson>(join(dir, 'reference.json')),
    pieces: await readJson<PieceJson[]>(join(dir, 'pieces.json')),
    reference: await loadRGBA(join(dir, 'reference.jpg')),
  }
}

export interface SceneData {
  id: string
  json: SceneJson
  frame: RGBAImage
  labels: Mask
}

export async function loadScene(ds: DatasetHandle, id: string): Promise<SceneData> {
  const base = join(ds.root, 'scenes', id)
  return {
    id,
    json: await readJson<SceneJson>(`${base}.json`),
    frame: await loadRGBA(`${base}.jpg`),
    labels: await loadLabels(`${base}_mask.png`),
  }
}

/** Scene ids grouped by puzzle id, in meta order. */
export async function scenesByPuzzle(ds: DatasetHandle): Promise<Map<string, string[]>> {
  const out = new Map<string, string[]>()
  for (const id of ds.meta.scenes) {
    const json = await readJson<SceneJson>(join(ds.root, 'scenes', `${id}.json`))
    const list = out.get(json.puzzleId) ?? []
    list.push(id)
    out.set(json.puzzleId, list)
  }
  return out
}

/** Pieces that the metrics count (DATASET.md): face-up and at least 60 % visible. */
export function isMatchable(p: ScenePieceJson): boolean {
  return p.faceUp && p.visibleFraction >= 0.6
}

/** Signed angle difference a − b wrapped to (−180, 180]. */
export function angleDiff(a: number, b: number): number {
  let d = (a - b) % 360
  if (d <= -180) d += 360
  if (d > 180) d -= 360
  return d
}
