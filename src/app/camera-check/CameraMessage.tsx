/**
 * What the Camera check screen shows while there is no live picture: opening, paused, off, or
 * one friendly message per failure the user can act on (src/camera/errors.ts). The browser's own
 * error text is kept as a small detail line, because it is what a bug report needs.
 */

import type { CameraFailureKind } from '../../camera/errors.ts'
import type { CameraState } from '../../camera/session.ts'

const FAILURE_TEXT: Record<CameraFailureKind, { title: string; body: string }> = {
  'insecure-context': {
    title: 'The camera needs a secure connection',
    body: 'Browsers only allow the camera on https:// pages. Open the app over HTTPS; for testing on the local network, use npm run dev:lan.',
  },
  unsupported: {
    title: 'This browser cannot use the camera',
    body: 'Open the app in Safari on iPhone, or in Chrome on Android.',
  },
  'permission-denied': {
    title: 'Camera access is blocked',
    body: 'Allow the camera for this site in your browser settings (on iPhone: Settings → Apps → Safari → Camera), then try again. An app added to the Home Screen asks again each time it starts.',
  },
  'no-camera': {
    title: 'No camera found',
    body: 'This device has no camera the browser can use, or the chosen camera is no longer available.',
  },
  'camera-busy': {
    title: 'The camera is in use',
    body: 'Another app or browser tab is using the camera. Close it, then try again.',
  },
  interrupted: {
    title: 'The camera stopped',
    body: 'Something else took over the camera, such as a call or another app.',
  },
  unknown: {
    title: 'The camera could not start',
    body: 'Try again. If it keeps failing, reload the page.',
  },
}

interface CameraMessageProps {
  state: Exclude<CameraState, { status: 'running' }>
  onStart: () => void
}

export function CameraMessage({ state, onStart }: CameraMessageProps) {
  switch (state.status) {
    case 'starting':
      return (
        <Panel title="Opening the camera…" body="If your phone asks, allow camera access." />
      )
    case 'paused':
      return <Panel title="Camera paused" body="It restarts when you return to the app." />
    case 'idle':
      return <Panel title="The camera is off" action={{ label: 'Start camera', onClick: onStart }} />
    case 'error': {
      const text = FAILURE_TEXT[state.failure.kind]
      return (
        <Panel
          title={text.title}
          body={text.body}
          detail={state.failure.detail}
          action={{ label: 'Try again', onClick: onStart }}
        />
      )
    }
  }
}

interface PanelProps {
  title: string
  body?: string
  detail?: string
  action?: { label: string; onClick: () => void }
}

function Panel({ title, body, detail, action }: PanelProps) {
  return (
    <div className="px-safe absolute inset-0 flex items-center justify-center">
      <div
        role="status"
        className="bg-ink-900/90 w-full max-w-sm rounded-2xl p-6 text-center shadow-xl shadow-black/40"
      >
        <h2 className="text-lg font-semibold text-white">{title}</h2>
        {body !== undefined && <p className="mt-2 text-sm text-slate-300">{body}</p>}
        {detail !== undefined && (
          <p className="mt-3 text-xs break-words text-slate-500">{detail}</p>
        )}
        {action !== undefined && (
          <button
            type="button"
            onClick={action.onClick}
            className="bg-accent text-ink-950 active:bg-accent-strong mt-5 w-full rounded-xl px-4 py-3 font-semibold"
          >
            {action.label}
          </button>
        )}
      </div>
    </div>
  )
}
