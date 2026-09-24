/**
 * Turns anything a promise can reject with into one readable line for the UI.
 */

export function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message || error.name
  return String(error)
}
