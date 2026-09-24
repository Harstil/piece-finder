/**
 * App icon generator: `node scripts/make-icons.ts` (or `npm run icons`).
 *
 * This script is the single source of truth for the Piece Finder icon. It draws the artwork as SVG
 * (a jigsaw piece with a crosshair: "this is where it goes"), writes `public/icon.svg` (the favicon),
 * and renders the PNGs the PWA manifest and iOS need with sharp. The PNG outputs are committed, so
 * the build does not depend on sharp; re-run this only after changing the artwork.
 *
 * The piece outline is built from the same conventions as the engine (src/engine/types.ts): the
 * core's corners run clockwise TL, TR, BR, BL in image coordinates (y down) and sides are ordered
 * top, right, bottom, left, each 'flat' | 'tab' | 'blank'.
 */

import { mkdir, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import sharp from 'sharp'

type Side = 'flat' | 'tab' | 'blank'
interface Vec {
  x: number
  y: number
}

/** Flat palette: 3 colours. The background doubles as the app's theme colour (see vite.config.ts). */
const COLORS = {
  background: '#0b1020',
  piece: '#f5a524',
  mark: '#0b1020',
} as const

/** Artwork is drawn on a 512 × 512 canvas and scaled per output. */
const CANVAS = 512

/** Side length of the piece's square core, in canvas px (design choice). */
const CORE = 250

/**
 * Tab size as a fraction of the side length (design choice). The knob control points follow the
 * common parametric jigsaw-edge construction; with t = 0.1 a tab protrudes 0.25 × side.
 */
const TAB_T = 0.1
const TAB_PROTRUSION = 2.5 * TAB_T

const PIECE_SIDES: [Side, Side, Side, Side] = ['tab', 'tab', 'blank', 'blank']

/** One side from corner p to corner q (clockwise), with a knob pushed out (tab) or in (blank). */
function sidePath(p: Vec, q: Vec, kind: Side): string {
  if (kind === 'flat') return `L ${fmt(q)}`
  const len = Math.hypot(q.x - p.x, q.y - p.y)
  const d = { x: (q.x - p.x) / len, y: (q.y - p.y) / len }
  // Outward normal for a clockwise outline in y-down image coordinates.
  const n = { x: d.y, y: -d.x }
  const sign = kind === 'tab' ? 1 : -1
  const at = (u: number, v: number): Vec => ({
    x: p.x + d.x * u * len + n.x * v * len * sign,
    y: p.y + d.y * u * len + n.y * v * len * sign,
  })
  const t = TAB_T
  const pts = [
    at(0.2, 0),
    at(0.5, -t),
    at(0.5 - t, t),
    at(0.5 - 2 * t, 3 * t),
    at(0.5 + 2 * t, 3 * t),
    at(0.5 + t, t),
    at(0.5, -t),
    at(0.8, 0),
    q,
  ]
  return [
    `C ${fmt(pts[0])} ${fmt(pts[1])} ${fmt(pts[2])}`,
    `C ${fmt(pts[3])} ${fmt(pts[4])} ${fmt(pts[5])}`,
    `C ${fmt(pts[6])} ${fmt(pts[7])} ${fmt(pts[8])}`,
  ].join(' ')
}

function fmt(v: Vec): string {
  return `${v.x.toFixed(2)} ${v.y.toFixed(2)}`
}

/** Closed outline of a piece whose core's top-left corner is (x, y). */
function piecePath(x: number, y: number, size: number, sides: [Side, Side, Side, Side]): string {
  const corners: Vec[] = [
    { x, y },
    { x: x + size, y },
    { x: x + size, y: y + size },
    { x, y: y + size },
  ]
  const parts = [`M ${fmt(corners[0])}`]
  for (let i = 0; i < 4; i++) parts.push(sidePath(corners[i], corners[(i + 1) % 4], sides[i]))
  parts.push('Z')
  return parts.join(' ')
}

/** Piece + crosshair on the 512 canvas, centred on the piece's bounding box (tabs included). */
function artwork(): string {
  const bboxSize = CORE * (1 + TAB_PROTRUSION)
  const x0 = (CANVAS - bboxSize) / 2
  const y0 = (CANVAS - bboxSize) / 2 + CORE * TAB_PROTRUSION
  const cx = x0 + CORE / 2
  const cy = y0 + CORE / 2
  // Sized so the ticks (plus round caps) end before the blanks' knob tips, 0.25 × CORE from the centre.
  const ring = CORE * 0.15
  const stroke = CORE * 0.06
  const tick = CORE * 0.05
  const ticks = [
    [cx, cy - ring - tick, cx, cy - ring + tick],
    [cx + ring - tick, cy, cx + ring + tick, cy],
    [cx, cy + ring - tick, cx, cy + ring + tick],
    [cx - ring - tick, cy, cx - ring + tick, cy],
  ]
    .map(([ax, ay, bx, by]) => `<line x1="${ax}" y1="${ay}" x2="${bx}" y2="${by}"/>`)
    .join('')
  return [
    `<path d="${piecePath(x0, y0, CORE, PIECE_SIDES)}" fill="${COLORS.piece}"/>`,
    `<g stroke="${COLORS.mark}" stroke-width="${stroke}" stroke-linecap="round" fill="none">`,
    `<circle cx="${cx}" cy="${cy}" r="${ring}"/>`,
    ticks,
    '</g>',
    `<circle cx="${cx}" cy="${cy}" r="${stroke * 0.9}" fill="${COLORS.mark}"/>`,
  ].join('')
}

interface IconVariant {
  /** Rounded tile for "any" icons and the favicon; a full-bleed square where the OS applies its own mask. */
  shape: 'rounded' | 'square'
  /** Artwork scale about the centre. Maskable icons keep everything inside the central 80 % circle. */
  artScale: number
}

function iconSvg(variant: IconVariant, pixelSize?: number): string {
  const sizeAttrs = pixelSize === undefined ? '' : ` width="${pixelSize}" height="${pixelSize}"`
  const radius = variant.shape === 'rounded' ? CANVAS * 0.22 : 0
  const offset = (CANVAS * (1 - variant.artScale)) / 2
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${CANVAS} ${CANVAS}"${sizeAttrs}>`,
    `<rect width="${CANVAS}" height="${CANVAS}" rx="${radius}" fill="${COLORS.background}"/>`,
    `<g transform="translate(${offset} ${offset}) scale(${variant.artScale})">${artwork()}</g>`,
    '</svg>',
  ].join('')
}

const ROUNDED: IconVariant = { shape: 'rounded', artScale: 1 }
// 0.78: the artwork's farthest point (the bottom-left core corner) then sits ~0.34 × size from the
// centre, inside the 0.4 × size safe radius that every maskable-icon shape keeps.
const MASKABLE: IconVariant = { shape: 'square', artScale: 0.78 }
// iOS rounds the corners itself and shows transparency as black, so the tile is full-bleed.
const APPLE: IconVariant = { shape: 'square', artScale: 0.9 }

const OUTPUTS: { file: string; size: number; variant: IconVariant }[] = [
  { file: 'pwa-192x192.png', size: 192, variant: ROUNDED },
  { file: 'pwa-512x512.png', size: 512, variant: ROUNDED },
  { file: 'maskable-512x512.png', size: 512, variant: MASKABLE },
  { file: 'apple-touch-icon.png', size: 180, variant: APPLE },
]

const publicDir = fileURLToPath(new URL('../public/', import.meta.url))
await mkdir(publicDir, { recursive: true })
await writeFile(`${publicDir}icon.svg`, `${iconSvg(ROUNDED)}\n`)
console.log('wrote public/icon.svg')
for (const { file, size, variant } of OUTPUTS) {
  await sharp(Buffer.from(iconSvg(variant, size)))
    .png({ compressionLevel: 9 })
    .toFile(`${publicDir}${file}`)
  console.log(`wrote public/${file} (${size}×${size})`)
}
