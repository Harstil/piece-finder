/**
 * The build label shown on the Home screen, so a tester on the phone can tell which deploy they
 * are running (the service worker can keep an older version around until the next reload).
 *
 * The three constants are replaced at build time by Vite's `define` (vite.config.ts): package
 * version, short commit hash (Vercel's VERCEL_GIT_COMMIT_SHA, or local git) and build date.
 */

declare const __APP_VERSION__: string
declare const __APP_COMMIT__: string
declare const __APP_BUILD_DATE__: string

export const BUILD_LABEL = `v${__APP_VERSION__} · ${__APP_COMMIT__} · ${__APP_BUILD_DATE__}`
