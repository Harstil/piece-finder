import { describe, expect, it } from 'vitest'
import { gridOptions } from './gridOptions.ts'

describe('gridOptions', () => {
  it('suggests the classic grids first', () => {
    expect(gridOptions(500, 4 / 3)[0]).toMatchObject({ cols: 25, rows: 20, count: 500 })
    expect(gridOptions(1000, 1.5)[0]).toMatchObject({ cols: 40, rows: 25 })
  })

  it("includes Ravensburger's non-square 36 × 28 for a 70 × 50 cm 1000-piece puzzle", () => {
    const options = gridOptions(1000, 70 / 50)
    expect(options.some((o) => o.cols === 36 && o.rows === 28)).toBe(true)
  })

  it('handles portrait pictures and rejects nonsense', () => {
    expect(gridOptions(300, 2 / 3)[0]).toMatchObject({ cols: 15, rows: 20 })
    expect(gridOptions(0, 1)).toEqual([])
    expect(gridOptions(100, 0)).toEqual([])
  })
})
