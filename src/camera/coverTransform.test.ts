/**
 * Checks the object-fit: cover mapping against hand-computed cases, including the portrait phone
 * case (a 1080×1920 frame on a 390×844 CSS-px iPhone screen) that every overlay depends on.
 */

import { describe, expect, it } from 'vitest'
import { coverTransform } from './coverTransform.ts'

describe('coverTransform', () => {
  it('is the identity when the frame already matches the view', () => {
    expect(coverTransform(640, 480, 640, 480)).toEqual({ scale: 1, offsetX: 0, offsetY: 0 })
  })

  it('crops the sides of a frame that is wider than the view', () => {
    // 1920×1080 into 1080×1080: height fits at scale 1, 840 px of width is cropped, 420 each side.
    expect(coverTransform(1920, 1080, 1080, 1080)).toEqual({ scale: 1, offsetX: -420, offsetY: 0 })
  })

  it('crops top and bottom of a frame that is taller than the view', () => {
    // 1000×2000 into 500×500: width fits at scale 0.5 → 1000 px tall, 250 cropped top and bottom.
    expect(coverTransform(1000, 2000, 500, 500)).toEqual({ scale: 0.5, offsetX: 0, offsetY: -250 })
  })

  it('maps the frame centre to the view centre for a portrait phone', () => {
    const t = coverTransform(1080, 1920, 390, 844)
    // The taller screen aspect (844/390 > 1920/1080) means height decides the scale.
    expect(t.scale).toBeCloseTo(844 / 1920, 12)
    expect(540 * t.scale + t.offsetX).toBeCloseTo(195, 9)
    expect(960 * t.scale + t.offsetY).toBeCloseTo(422, 9)
    expect(t.offsetY).toBeCloseTo(0, 9)
    expect(t.offsetX).toBeLessThan(0)
  })
})
