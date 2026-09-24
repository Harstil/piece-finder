/**
 * Asks the engine worker what this phone and browser can do (cross-origin isolation, WASM SIMD,
 * WebGPU, …; see src/worker/probe.ts), once per visit to the Camera check screen.
 */

import { useEffect, useState } from 'react'
import type { CapabilityReport } from '../../worker/api.ts'
import { getEngine } from '../../worker/client.ts'
import { errorMessage } from '../errorMessage.ts'

export type CapabilitiesStatus =
  | { status: 'probing' }
  | { status: 'done'; report: CapabilityReport }
  | { status: 'failed'; error: string }

export function useCapabilities(): CapabilitiesStatus {
  const [result, setResult] = useState<CapabilitiesStatus>({ status: 'probing' })

  useEffect(() => {
    let cancelled = false
    getEngine()
      .probe()
      .then(
        (report) => {
          if (!cancelled) setResult({ status: 'done', report })
        },
        (error: unknown) => {
          if (!cancelled) setResult({ status: 'failed', error: errorMessage(error) })
        },
      )
    return () => {
      cancelled = true
    }
  }, [])

  return result
}
