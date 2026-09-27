/**
 * Public API of the Piece Finder engine (pure TypeScript; runs in the Web Worker and in Node).
 *
 *   buildReference(photo, corners, grid) → ReferenceModel      once per puzzle
 *   analyzeShape(contour)                → PieceShape | null    per piece outline
 *   canonicalize(frame, contour, shape)  → CanonicalPiece       per piece
 *   matchPiece(reference, piece, opts)   → MatchResult          per piece
 */

export { canonicalize } from './canonical.ts'
export type { CanonicalOptions } from './canonical.ts'
export { DEFAULT_FUSION, resolveFusion, verdictFor } from './match/fusion.ts'
export { matchPiece } from './match/index.ts'
export { canonicalUp, unitToFrame, upAngleDeg } from './pose.ts'
export { buildReference, photoCellPx, referenceSummary } from './reference.ts'
export type { BuildReferenceOptions, ReferenceLevel, ReferenceModel } from './reference.ts'
export { analyzeShape } from './shape.ts'
export type { ShapeOptions } from './shape.ts'
export type * from './types.ts'
