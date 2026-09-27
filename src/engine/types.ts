/**
 * Engine contract — the shapes every part of Piece Finder agrees on.
 *
 * The engine is pure TypeScript on typed arrays. It runs unchanged in two places:
 * the browser Web Worker (live camera) and Node (the evaluation harness in `eval/`).
 * So nothing under `src/engine/` may touch the DOM, `window`, `document`, canvases or fetch.
 * Imports inside the engine use explicit `.ts` extensions and erasable-only syntax
 * (no enums, no namespaces, no parameter properties) so Node 24 can run it directly.
 *
 * Conventions (all of these are load-bearing — the dataset generator uses the same ones):
 * - Image coordinates: x to the right, y down, pixel centres at integer coordinates.
 * - Grid: `col` 0..cols-1 from the left, `row` 0..rows-1 from the top, `cell = row * cols + col`.
 * - Piece corners: the 4 sharp corners of a piece's square-ish core (not tab tips).
 * - A piece's *motif orientation* is how it sits in the finished puzzle. In that orientation its
 *   corners are TL, TR, BR, BL and its sides are 0 = top, 1 = right, 2 = bottom, 3 = left.
 * - Angles are in degrees, clockwise, measured from "up" in the image (0 = up, 90 = right).
 * - `upAngleDeg` of a piece in a frame = the direction its motif-up vector points in that frame.
 *   0 means the piece already lies the right way up; 90 means its top points to the frame's right,
 *   so the user must turn it 90° counter-clockwise. Under perspective the direction varies slightly
 *   across the piece; it is taken at the core centre (the mean of the 4 corners in motif coordinates).
 *   For a non-rectangular core (irregular cut) this is NOT the bottom-edge → top-edge midpoint direction.
 */

export interface Point {
  x: number
  y: number
}

/** Four points in order. For a piece in the frame: clockwise as seen in the image. */
export type Quad = [Point, Point, Point, Point]

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

/** ImageData-compatible RGBA8 image (row-major, 4 bytes per pixel). */
export interface RGBAImage {
  width: number
  height: number
  data: Uint8ClampedArray
}

/** Single-channel float image (row-major). */
export interface GrayImage {
  width: number
  height: number
  data: Float32Array
}

/** Planar CIE Lab image: L in [0,100], a/b roughly [-128,127]. Each plane is width*height. */
export interface LabImage {
  width: number
  height: number
  L: Float32Array
  a: Float32Array
  b: Float32Array
}

/** Binary / label mask (row-major). 0 = background. */
export interface Mask {
  width: number
  height: number
  data: Uint8Array
}

export type SideKind = 'flat' | 'tab' | 'blank'
export type PieceType = 'corner' | 'edge' | 'interior'
/** Quarter turns clockwise. */
export type Rotation = 0 | 1 | 2 | 3

export interface GridSpec {
  cols: number
  rows: number
}

/** One segmented piece in a frame, before shape analysis. */
export interface PieceInstance {
  /** Outer contour in frame coordinates, clockwise, no repeated closing point. */
  contour: Point[]
  bbox: Rect
  area: number
  /** Segmentation confidence 0..1. */
  score: number
}

/** Result of analysing a piece outline. */
export interface PieceShape {
  /**
   * The 4 core corners in frame coordinates, clockwise as seen in the image, starting at an
   * arbitrary corner. Side i runs from corners[i] to corners[(i + 1) % 4].
   */
  corners: Quad
  sides: [SideKind, SideKind, SideKind, SideKind]
  pieceType: PieceType
  /** 0..1 — how rectangle-like the chosen corner quadruple is; low means "don't trust". */
  confidence: number
}

/**
 * Canonical piece: the core warped to a square of `size` px with `margin` px of padding on every
 * side so tabs survive. Canonical corner i = shape.corners[i] mapped to TL, TR, BR, BL.
 */
export interface CanonicalPiece {
  size: number
  margin: number
  lab: LabImage
  /** 1 where the pixel belongs to the piece (tabs included, blanks excluded). */
  mask: Mask
  shape: PieceShape
  /**
   * The same piece at every matching resolution, coarsest first (core `size` 8, 16, 32 px). The
   * top-level size/margin/lab/mask are the finest of these.
   */
  levels: CanonicalLevel[]
}

export interface CanonicalLevel {
  size: number
  margin: number
  lab: LabImage
  mask: Mask
}

/**
 * One location hypothesis.
 * `rotation` r: turning the canonical patch clockwise by r × 90° makes it upright in the motif.
 */
export interface Candidate {
  col: number
  row: number
  cell: number
  rotation: Rotation
  /** Where the piece's motif-up points in the frame (see file header). */
  upAngleDeg: number
  /** Raw fused score (higher is better). */
  score: number
  /** Calibrated probability over all hypotheses considered, 0..1. */
  prob: number
}

export interface MatchOptions {
  /** Cells to skip (already placed). */
  excludedCells?: ReadonlySet<number>
  /** How many candidates to return. Default 5. */
  topK?: number
  /** Which cues to use — lets the eval harness run ablations. Default: all enabled. */
  cues?: Partial<Record<CueName, boolean>>
  /** Override of the fused-score weights (eval harness fitting). Default: match/fusion.ts. */
  fusion?: Partial<FusionParams>
}

/** Weights and temperatures that turn cue values into scores and probabilities (match/fusion.ts). */
export interface FusionParams {
  /** Masked ZNCC on L (texture). */
  wLuma: number
  /** Variance-weighted masked ZNCC on a and b (colour pattern). */
  wChroma: number
  /** Penalty per unit of mean-colour distance in ab (colour drift between box print and camera). */
  wChromaMean: number
  /** Normalised correlation of gradient vectors (structure, lighting-robust). */
  wGradient: number
  /** Penalty on mean/std luminance mismatch (the cheap histogram stand-in). */
  wStats: number
  /** Penalty per side whose flat/non-flat kind contradicts the cell's border (soft shape prior). */
  wShapeMismatch: number
  /** Piece L std (Lab units) at which luma ZNCC gets its full weight; weaker texture scales it down. */
  lumaTextureRef: number
  /** Softmax temperature over final scores. */
  temperature: number
  /** Softmax temperature for the coarse per-cell heatmap. */
  coarseTemperature: number
  /** Share of probability mass given to the finalists (the true answer can be pruned early). */
  finalistMass: number
}

export type Verdict = 'strong' | 'likely' | 'unsure'

export type CueName = 'lumaZncc' | 'chroma' | 'gradient' | 'histogram' | 'shapePrior'

export interface MatchResult {
  candidates: Candidate[]
  pieceType: PieceType
  timingsMs: Record<string, number>
  verdict: Verdict
  /** Coarse probability per cell (length cols × rows, sums to ~1) for the candidate heatmap. */
  heat: Float32Array
}

/** Summary the UI shows after a reference is built. */
export interface ReferenceSummary {
  grid: GridSpec
  cellSizePx: number
  /** Rectified motif size in px. */
  width: number
  height: number
}

/** A piece being followed across frames while the user hovers. */
export interface TrackState {
  trackId: number
  /** Decimated outline for drawing, frame coordinates. */
  contour: Point[]
  corners: Quad | null
  pieceType: PieceType | null
  /** Accumulated over frames, best first. */
  candidates: Candidate[]
  /** True once the top candidate is confident enough to show as the answer. */
  locked: boolean
  /** Probability mass inside the active region (Region finder), or null when no region is set. */
  regionProb: number | null
  framesSeen: number
}

export interface FrameResult {
  frameId: number
  width: number
  height: number
  tracks: TrackState[]
  timingsMs: Record<string, number>
}
