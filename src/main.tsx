/**
 * Entry point: mounts the React app into index.html's #root.
 *
 * StrictMode stays on: its development-only double mount is what proves the camera session and
 * the frame pump clean up after themselves (a leaked camera stream keeps the phone's camera light
 * on). The service worker is registered by vite-plugin-pwa's injected script, not from here.
 */

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './app/App.tsx'
import './index.css'

const container = document.getElementById('root')
if (container === null) throw new Error('index.html is missing the #root element')

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
