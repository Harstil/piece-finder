/**
 * Home screen: what the app is, the way into the Camera check, and which build is running.
 *
 * The Camera check comes first because everything else depends on it: it shows, on the actual
 * phone, whether the camera, the worker frame path and the fast-inference features (cross-origin
 * isolation, WASM SIMD, WebGPU) work before any puzzle is set up.
 */

import { BUILD_LABEL } from './buildInfo.ts'

interface HomeScreenProps {
  onCameraCheck: () => void
}

export function HomeScreen({ onCameraCheck }: HomeScreenProps) {
  return (
    <main className="pt-safe pb-safe px-safe flex min-h-dvh flex-col">
      <div className="flex flex-1 flex-col items-center justify-center text-center">
        <img src="/icon.svg" alt="" width={96} height={96} className="mb-6 size-24" />
        <h1 className="text-3xl font-semibold tracking-tight text-white">Piece Finder</h1>
        <p className="mt-3 max-w-xs text-base text-slate-300">
          Hold your phone over loose jigsaw pieces and see where each one goes and how to turn it.
        </p>
        <button
          type="button"
          onClick={onCameraCheck}
          className="bg-accent text-ink-950 active:bg-accent-strong mt-10 w-full max-w-xs rounded-2xl px-6 py-4 text-lg font-semibold shadow-lg shadow-black/30"
        >
          Camera check
        </button>
        <p className="mt-3 max-w-xs text-sm text-slate-400">
          Tests the camera and how fast this phone can process what it sees.
        </p>
      </div>
      <p className="text-center text-xs text-slate-500 tabular-nums">{BUILD_LABEL}</p>
    </main>
  )
}
