/**
 * Draws the straightened box picture with the puzzle grid on a canvas: placed cells, the Region-finder
 * area, and highlighted candidate cells. Shared by the mini-map, the piece sheet (zoomed on the target)
 * and the Picture screen, so all three show cells in exactly the same place.
 *
 * Cells are drawn on the picture's own pixel grid: cell (col, row) covers
 * [col·W/cols, (col+1)·W/cols) × [row·H/rows, (row+1)·H/rows) of the picture, then scaled to the
 * destination rectangle. `view` picks a sub-rectangle of cells to zoom into (the whole picture by default).
 */

import type { GridSpec } from '../../engine/types.ts'

export const COLORS = {
  accent: '#f5a524',
  strong: '#22c55e',
  likely: '#f5a524',
  unsure: '#94a3b8',
  placed: 'rgba(34, 197, 94, 0.45)',
  region: 'rgba(56, 189, 248, 0.35)',
  regionEdge: '#38bdf8',
}

export interface CellHighlight {
  cell: number
  color: string
  /** Line width in CSS px. */
  width: number
  label?: string
}

export interface PictureView {
  col0: number
  row0: number
  col1: number
  row1: number
}

export interface DrawPictureOptions {
  grid: GridSpec
  placed?: readonly number[]
  region?: ReadonlySet<number> | null
  highlights?: readonly CellHighlight[]
  gridLines?: boolean
  view?: PictureView
}

/** A view of `radius` cells around `cell`, clamped to the puzzle. */
export function viewAround(cell: number, grid: GridSpec, radius: number): PictureView {
  const col = cell % grid.cols
  const row = Math.floor(cell / grid.cols)
  const span = 2 * radius + 1
  const col0 = Math.max(0, Math.min(grid.cols - span, col - radius))
  const row0 = Math.max(0, Math.min(grid.rows - span, row - radius))
  return { col0: Math.max(0, col0), row0: Math.max(0, row0), col1: Math.min(grid.cols, col0 + span), row1: Math.min(grid.rows, row0 + span) }
}

/**
 * Draws into `ctx` within the destination rectangle (dx, dy, dw, dh) in the context's current
 * coordinate system (CSS px when the caller has applied the device-pixel-ratio transform).
 */
export function drawPicture(
  ctx: CanvasRenderingContext2D,
  picture: ImageBitmap,
  dx: number,
  dy: number,
  dw: number,
  dh: number,
  opts: DrawPictureOptions,
): { cellRect: (cell: number) => { x: number; y: number; w: number; h: number } } {
  const { cols, rows } = opts.grid
  const view = opts.view ?? { col0: 0, row0: 0, col1: cols, row1: rows }
  const cw = picture.width / cols
  const ch = picture.height / rows
  const sx = view.col0 * cw
  const sy = view.row0 * ch
  const sw = (view.col1 - view.col0) * cw
  const sh = (view.row1 - view.row0) * ch
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(picture, sx, sy, sw, sh, dx, dy, dw, dh)
  const kx = dw / (view.col1 - view.col0)
  const ky = dh / (view.row1 - view.row0)
  const cellRect = (cell: number) => {
    const col = cell % cols
    const row = Math.floor(cell / cols)
    return { x: dx + (col - view.col0) * kx, y: dy + (row - view.row0) * ky, w: kx, h: ky }
  }
  const visible = (cell: number): boolean => {
    const col = cell % cols
    const row = Math.floor(cell / cols)
    return col >= view.col0 && col < view.col1 && row >= view.row0 && row < view.row1
  }

  if (opts.gridLines === true && kx >= 4) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)'
    ctx.lineWidth = 1
    ctx.beginPath()
    for (let c = 1; c < view.col1 - view.col0; c++) {
      ctx.moveTo(dx + c * kx, dy)
      ctx.lineTo(dx + c * kx, dy + dh)
    }
    for (let r = 1; r < view.row1 - view.row0; r++) {
      ctx.moveTo(dx, dy + r * ky)
      ctx.lineTo(dx + dw, dy + r * ky)
    }
    ctx.stroke()
  }
  if (opts.placed !== undefined && opts.placed.length > 0) {
    ctx.fillStyle = COLORS.placed
    for (const cell of opts.placed) {
      if (!visible(cell)) continue
      const r = cellRect(cell)
      ctx.fillRect(r.x, r.y, r.w, r.h)
    }
  }
  if (opts.region !== undefined && opts.region !== null) {
    ctx.fillStyle = COLORS.region
    for (const cell of opts.region) {
      if (!visible(cell)) continue
      const r = cellRect(cell)
      ctx.fillRect(r.x, r.y, r.w, r.h)
    }
  }
  for (const h of opts.highlights ?? []) {
    if (!visible(h.cell)) continue
    const r = cellRect(h.cell)
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.65)'
    ctx.lineWidth = h.width + 2
    ctx.strokeRect(r.x, r.y, r.w, r.h)
    ctx.strokeStyle = h.color
    ctx.lineWidth = h.width
    ctx.strokeRect(r.x, r.y, r.w, r.h)
    if (h.label !== undefined) {
      const size = Math.max(10, Math.min(16, r.h * 0.5))
      ctx.font = `700 ${size}px system-ui, sans-serif`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.lineWidth = 3
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)'
      ctx.strokeText(h.label, r.x + r.w / 2, r.y + r.h / 2)
      ctx.fillStyle = h.color
      ctx.fillText(h.label, r.x + r.w / 2, r.y + r.h / 2)
    }
  }
  return { cellRect }
}

/** "C12 · R7": 1-based column and row, as the user counts them on the table. */
export function cellLabel(cell: number, grid: GridSpec): string {
  return `C${(cell % grid.cols) + 1} · R${Math.floor(cell / grid.cols) + 1}`
}
