/**
 * Vite build and dev-server config for Piece Finder.
 *
 * Three things here are load-bearing for the phone:
 * - Cross-origin isolation. COOP same-origin + COEP require-corp make the page `crossOriginIsolated`,
 *   which unlocks SharedArrayBuffer and so multithreaded WASM (onnxruntime-web). Safari has no
 *   `COEP: credentialless`, so it has to be `require-corp` and every asset must be self-hosted.
 *   vercel.json sends the same headers in production.
 * - HTTPS on the LAN. Browsers only expose the camera in a secure context, so `npm run dev:lan`
 *   (`vite --mode lan --host`) adds a self-signed certificate so the iPhone can open the dev server.
 * - Offline PWA. The service worker precaches everything the app needs, including the large
 *   onnxruntime `.wasm` and the segmentation model, so the precache size limit is raised.
 */

import { execSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import tailwindcss from '@tailwindcss/vite'
import basicSsl from '@vitejs/plugin-basic-ssl'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { VitePWA } from 'vite-plugin-pwa'

const CROSS_ORIGIN_ISOLATION_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

/**
 * Largest single file the service worker will precache. onnxruntime-web's plain WASM build is
 * ~14 MB (measured on 1.30); the rest is headroom for the segmentation model and a threaded build.
 */
const MAX_PRECACHE_FILE_BYTES = 40 * 1024 * 1024

/** Matches the icon background (scripts/make-icons.ts) and the app's page colour (src/index.css). */
const THEME_COLOR = '#0b1020'

const pkg = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8')) as {
  version: string
  description: string
}

/** Short commit hash for the build label: Vercel's env var in CI, local git otherwise. */
function commitHash(): string {
  const fromVercel = process.env.VERCEL_GIT_COMMIT_SHA
  if (fromVercel) return fromVercel.slice(0, 7)
  try {
    return execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] })
      .toString()
      .trim()
  } catch {
    return 'unknown'
  }
}

export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    tailwindcss(),
    ...(mode === 'lan' ? [basicSsl()] : []),
    VitePWA({
      registerType: 'autoUpdate',
      // public/ files (icons included) are already matched by globPatterns below; the plugin's own
      // icon/asset lists would only add duplicate precache entries.
      includeManifestIcons: false,
      manifest: {
        name: 'Piece Finder',
        short_name: 'Piece Finder',
        description: pkg.description,
        display: 'standalone',
        orientation: 'portrait',
        start_url: '/',
        scope: '/',
        theme_color: THEME_COLOR,
        background_color: THEME_COLOR,
        icons: [
          { src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png' },
          { src: 'maskable-512x512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        // The web manifest is added by the plugin itself; sw.js and workbox-*.js are never precached.
        globPatterns: ['**/*.{js,mjs,css,html,png,svg,wasm,onnx}'],
        maximumFileSizeToCacheInBytes: MAX_PRECACHE_FILE_BYTES,
      },
    }),
  ],
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
    __APP_COMMIT__: JSON.stringify(commitHash()),
    __APP_BUILD_DATE__: JSON.stringify(new Date().toISOString().slice(0, 10)),
  },
  worker: {
    format: 'es',
  },
  // onnxruntime-web finds its .wasm/.mjs files relative to its own module URL; pre-bundling it into
  // node_modules/.vite would break that lookup in dev.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: CROSS_ORIGIN_ISOLATION_HEADERS,
  },
  preview: {
    headers: CROSS_ORIGIN_ISOLATION_HEADERS,
  },
}))
