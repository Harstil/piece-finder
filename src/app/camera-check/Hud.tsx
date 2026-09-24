/**
 * The Camera check HUD: the few numbers that decide whether this phone can run Piece Finder,
 * plus the camera controls.
 *
 * Always visible: the frame size the worker receives, processed FPS, capture→result latency and
 * the three capability flags (cross-origin isolation for threaded WASM, WASM SIMD, WebGPU).
 * Controls: lens picker, torch (only when the camera reports one) and a Details panel with the
 * full track settings/capabilities and the worker's capability report, for bug reports.
 */

import type { RunningCamera } from '../../camera/session.ts'
import type { WebGpuReport } from '../../worker/api.ts'
import type { CapabilitiesStatus } from './useCapabilities.ts'
import type { FramePumpStatus } from './useFramePump.ts'

interface HudProps {
  camera: RunningCamera
  pump: FramePumpStatus
  capabilities: CapabilitiesStatus
  torchError: string | null
  /** Owned by the screen, so the panel stays open while the camera reopens on a lens switch. */
  detailsOpen: boolean
  onSelectDevice: (deviceId: string) => void
  onToggleTorch: () => void
  onToggleDetails: () => void
}

export function Hud({
  camera,
  pump,
  capabilities,
  torchError,
  detailsOpen,
  onSelectDevice,
  onToggleTorch,
  onToggleDetails,
}: HudProps) {
  const result = pump.stats?.lastResult
  const report = capabilities.status === 'done' ? capabilities.report : null
  const torchSupported = camera.report.capabilities?.torch === true

  return (
    <section className="pb-safe px-safe absolute inset-x-0 bottom-0">
      <div className="rounded-2xl bg-black/65 p-3 text-sm text-slate-100 backdrop-blur-md">
        <dl className="grid grid-cols-3 gap-2 text-center">
          <Stat label="Frame" value={result ? `${result.width}×${result.height}` : '—'} />
          <Stat label="Processed" value={pump.stats ? `${pump.stats.processedFps.toFixed(1)} fps` : '—'} />
          <Stat label="Latency" value={pump.stats ? formatMs(pump.stats.latencyMs) : '—'} />
        </dl>

        <div className="mt-3 flex flex-wrap justify-center gap-2">
          <Flag label="Isolated" value={report?.crossOriginIsolated ?? null} />
          <Flag label="SIMD" value={report?.wasmSimd ?? null} />
          <Flag label="WebGPU" value={report === null ? null : report.webgpu.available} />
        </div>

        {pump.error !== null && <Problem text={`Frame processing stopped: ${pump.error}`} />}
        {capabilities.status === 'failed' && (
          <Problem text={`Capability probe failed: ${capabilities.error}`} />
        )}
        {torchError !== null && <Problem text={`Torch: ${torchError}`} />}

        <div className="mt-3 flex gap-2">
          {camera.devices.length > 0 && (
            <label className="min-w-0 flex-1">
              <span className="sr-only">Camera</span>
              <select
                value={camera.activeDeviceId ?? ''}
                onChange={(event) => onSelectDevice(event.target.value)}
                className="bg-ink-800 w-full truncate rounded-xl px-3 py-2.5 text-slate-100"
              >
                {camera.activeDeviceId === null && (
                  <option value="" disabled>
                    {camera.report.label || 'Current camera'}
                  </option>
                )}
                {camera.devices.map((device) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          {torchSupported && (
            <button
              type="button"
              aria-pressed={camera.torchOn}
              onClick={onToggleTorch}
              className={`rounded-xl px-4 py-2.5 font-medium ${
                camera.torchOn ? 'bg-accent text-ink-950' : 'bg-ink-800 text-slate-100'
              }`}
            >
              Torch
            </button>
          )}
          <button
            type="button"
            aria-expanded={detailsOpen}
            onClick={onToggleDetails}
            className="bg-ink-800 rounded-xl px-4 py-2.5 font-medium text-slate-100"
          >
            Details
          </button>
        </div>

        {detailsOpen && <Details camera={camera} pump={pump} capabilities={capabilities} />}
      </div>
    </section>
  )
}

function Details({
  camera,
  pump,
  capabilities,
}: Pick<HudProps, 'camera' | 'pump' | 'capabilities'>) {
  const { settings, capabilities: caps } = camera.report
  const report = capabilities.status === 'done' ? capabilities.report : null
  const rows: [string, string][] = [
    ['Camera', camera.report.label || '—'],
    // Browsers may report these in the sensor's landscape terms; Frame above is what the worker gets.
    ['Track settings', `${dims(settings.width, settings.height)} @ ${num(settings.frameRate)} fps`],
    ['Frame rate range', range(caps?.frameRate ?? null)],
    ['Size range', `${range(caps?.width ?? null)} × ${range(caps?.height ?? null)}`],
    ['Focus mode', `${settings.focusMode ?? 'not reported'} (supports: ${list(caps?.focusModes)})`],
    ['Zoom', `${num(settings.zoom)} (range ${range(caps?.zoom ?? null)})`],
    ['Torch', caps?.torch ? (camera.torchOn ? 'on' : 'off') : 'not available'],
    ['Pixel read in worker', pump.stats ? formatMs(pump.stats.lastResult.timingsMs.readPixels ?? NaN) : '—'],
    ['Frames processed', pump.stats ? String(pump.stats.framesProcessed) : '—'],
  ]
  if (report !== null) {
    rows.push(
      ['SharedArrayBuffer', yesNo(report.sharedArrayBuffer)],
      ['OffscreenCanvas 2D', yesNo(report.offscreenCanvas2d)],
      ['CPU cores reported', String(report.hardwareConcurrency)],
      ['Device memory', report.deviceMemoryGb === null ? 'not exposed' : `${report.deviceMemoryGb} GB`],
      ['WebGPU', webGpuText(report.webgpu)],
      ['User agent', report.userAgent],
    )
  }
  return (
    <dl className="mt-3 max-h-[40dvh] space-y-1.5 overflow-y-auto border-t border-white/10 pt-3 text-xs">
      {rows.map(([label, value]) => (
        <div key={label} className="flex gap-3">
          <dt className="w-32 shrink-0 text-slate-400">{label}</dt>
          <dd className="min-w-0 break-words text-slate-100 tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs text-slate-400">{label}</dt>
      <dd className="font-semibold text-white tabular-nums">{value}</dd>
    </div>
  )
}

/** A capability flag; null while the worker probe is still running. */
function Flag({ label, value }: { label: string; value: boolean | null }) {
  const tone =
    value === null
      ? 'bg-white/10 text-slate-400'
      : value
        ? 'bg-emerald-500/20 text-emerald-300'
        : 'bg-rose-500/20 text-rose-300'
  const state = value === null ? 'checking' : value ? 'yes' : 'no'
  return (
    <span className={`rounded-full px-3 py-1 text-xs font-medium ${tone}`}>
      {label}: {state}
    </span>
  )
}

function Problem({ text }: { text: string }) {
  return <p className="mt-2 text-xs break-words text-rose-300">{text}</p>
}

function webGpuText(webgpu: WebGpuReport): string {
  if (!webgpu.available) return `no (${webgpu.reason})`
  if (webgpu.adapter === null) return 'yes (adapter info not exposed)'
  const { vendor, architecture, device, description } = webgpu.adapter
  const known = [vendor, architecture, device, description].filter((part) => part !== '')
  return `yes (${known.length > 0 ? known.join(' · ') : 'adapter info redacted'})`
}

function formatMs(ms: number): string {
  if (!Number.isFinite(ms)) return '—'
  return `${ms.toFixed(ms < 10 ? 1 : 0)} ms`
}

function num(value: number | null): string {
  if (value === null) return '—'
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}

function dims(width: number | null, height: number | null): string {
  return `${num(width)}×${num(height)}`
}

function range(value: { min: number; max: number } | null): string {
  return value === null ? '—' : `${num(value.min)}–${num(value.max)}`
}

function list(values: string[] | undefined): string {
  return values !== undefined && values.length > 0 ? values.join(', ') : 'not reported'
}

function yesNo(value: boolean): string {
  return value ? 'yes' : 'no'
}
