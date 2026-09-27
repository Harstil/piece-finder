# Piece Finder

A phone web app for jigsaw puzzles. Hold the phone camera over loose pieces on the table, and it shows
where each piece goes in the puzzle picture (column and row) and how to turn it.

- **Everything runs on the phone.** It is a static site with no server and no API keys, and it works offline as
  an installable PWA. Camera frames never leave the device.
- **iPhone first.** It is built for Safari on iPhone (Chrome on iOS uses the same WebKit engine), and it also
  runs in Chrome on Android.
- **The reference is the box picture.** You photograph the lid, or load a digital image of the motif.
- **Tables:** works best on a plain or lightly textured surface with pieces spread out; a learned segmentation
  model for any table is the next step.

## Using it

1. **Set up a puzzle.** Photograph the picture on the box lid (straight on, no glare), or pick a sharp image
   of the motif. Drag the four handles onto the corners of the *picture* — precisely; the magnifier helps. Enter
   the piece count and pick the grid (columns × rows) whose lines follow the picture.
2. **Scan pieces.** Spread loose pieces face-up so they don't touch, and hold the phone 20–40 cm above them.
   Each piece gets an outline and a label such as **C12 · R7** (column 12, row 7, counted from the top-left).
   The arrow on a piece points to its top. Green = confident, amber = likely, dashed white = still looking.
3. **Tap a piece** for the zoomed picture around its spot, the other likely spots, and how far to turn it.
   **Mark as placed** removes that spot from all future answers, so the app gets more accurate as you go.
4. **Picture** shows progress. **Select area** is the Region finder: drag across the part of the picture you
   are building, and in the scanner the pieces that belong there glow blue.

## How the engine works

`src/engine/` is pure TypeScript (no OpenCV, no ML runtime — the worker bundle is ~45 KB) and runs unchanged in
the phone's Web Worker and in Node for evaluation:

- **Reference** (`reference.ts`): the box photo is straightened from its four corners and split into the grid,
  at 8, 16 and 32 px per cell.
- **Segmentation** (`segment.ts`): the table's colours are learned per frame (k-means on the frame edges,
  shadow-tolerant); anything else is a candidate piece, re-outlined at full resolution.
- **Shape** (`shape.ts`): the piece outline gives its 4 core corners and each side's kind (flat / tab / blank).
  Flat sides restrict edge and corner pieces to the puzzle's border — and fix their rotation.
- **Canonical piece** (`canonical.ts`): one homography from the corners removes position, size, rotation and
  camera perspective at once; tabs are kept (they reach into neighbouring cells and add evidence).
- **Matching** (`match/`): every (cell, rotation) is scored coarse-to-fine with masked ZNCC on lightness,
  colour-pattern correlation, gradient correlation and colour statistics; weights fitted on validation data.
- **Tracking** (`tracker.ts`, `pipeline.ts`): the same piece is matched across frames while you hover; its
  answer locks when it is stable.

Accuracy is measured, not guessed: `tools/synth/` renders realistic synthetic puzzles (CC0 paintings, bezier
cuts, cardboard, tables, camera noise, box photos with glare and logos) and `eval/` runs the real engine on them.
Top-1 = the right cell is the first answer (outline-based corners, fitted weights, validation sets):

| pieces | 100 | 300 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| top-1 | 92 % | 82 % | 89 % | 68 % | 65 % |
| rotation (right cell) | 100 % | 100 % | 100 % | 100 % | 100 % |
| answers marked confident that are right | 96 % | 94 % | 95 % | 94 % | 96 % |

On the held-out test sets (artwork never used for tuning; `eval/reports/p2-test.md`): top-1 **86 %** at 100
pieces, **78 %** at 300, **85 %** at 500; rotation 100 %; answers marked confident are right 93–97 % of the time.

**Known limits (next steps):** detection is classical, so pieces must lie apart, and pieces whose colours match
the table are missed — the planned learned segmentation model fixes both. Large puzzles need a sharp box photo
(≥ 48 photo px per piece; the app warns below that).

## Quick start

Requires Node 22 or newer (developed on Node 24, which runs the `.ts` scripts directly).

```sh
npm install
npm run dev          # http://localhost:5173 (localhost counts as secure, so the desktop webcam works)
```

### Test on the iPhone over the local network

Browsers only allow the camera on HTTPS pages, and a phone cannot reach `localhost`. Use:

```sh
npm run dev:lan      # vite --mode lan --host: HTTPS with a self-signed certificate, reachable on the LAN
```

Vite prints a `Network:` address such as `https://192.168.1.23:5173/`. Open that address on the phone. The
phone and the PC must be on the same Wi-Fi, and Windows may ask you to allow Node through the firewall (allow it
on private networks).

The certificate is self-signed, so the browser warns you the first time:

- **Safari (iPhone):** "This Connection Is Not Private" → **Show Details** → **visit this website** →
  **Visit Website**.
- **Chrome (Android):** "Your connection is not private" → **Advanced** → **Proceed to …**.

The warning appears because the certificate was generated on your PC rather than issued by an authority the
phone trusts. The connection is still encrypted, and it is only your own dev server. After you accept, the page is
a secure context: the camera works, and the COOP/COEP headers make it `crossOriginIsolated`.

### Other commands

```sh
npm test             # unit tests (Vitest, Node environment)
npm run lint         # ESLint over the whole repo
npm run typecheck    # tsc -b for the app (src/) and the Node side (configs, scripts/, eval/)
npm run build        # typecheck + production build into dist/ (includes the service worker)
npm run preview      # serve dist/ locally with the production headers
npm run eval -- --sets v1/val500              # matching accuracy (see eval/run.ts for options)
node eval/shape-eval.ts v1/val500               # corner / side-kind accuracy
node eval/segment-eval.ts v1/val500 --match     # piece detection + end-to-end accuracy
node eval/pipeline-bench.ts v1/val500           # the live pipeline over frames, with timings
python -m tools.synth.make_eval_sets            # regenerate the frozen val/test datasets
npm run icons        # re-render the app icons into public/ (only needed after changing the artwork)
```

## Project layout

```
index.html            the single page; iOS home-screen meta tags (viewport-fit=cover, status bar, icon)
vite.config.ts        React, Tailwind, PWA (manifest + offline precache), COOP/COEP headers, `lan` HTTPS mode
vercel.json           production headers, caching and SPA rewrite for Vercel
eslint.config.js      flat config: recommended JS/TS rules, React hooks + fast-refresh for the UI
public/               icons (generated by scripts/make-icons.ts); later also models/ for the ONNX model
scripts/make-icons.ts renders the SVG icon and the PWA/iOS PNG icons with sharp
src/
  main.tsx            React entry
  index.css           Tailwind v4 theme and safe-area utilities
  app/                React UI: Home, Setup (photo, corners, grid), Scan (overlay, piece sheet), Picture, Camera check
  camera/             getUserMedia session (lens, torch, pause when hidden), frame pump, cover mapping
  store/              the active puzzle in IndexedDB (photo, corners, grid, placed cells)
  worker/             the engine Web Worker (Comlink API), capability probe, UI-side client
  engine/             pure-TypeScript vision engine; runs in the worker and in Node (types.ts is the contract)
eval/                 Node harness that runs the engine on datasets and reports accuracy
tools/                Python: synthetic dataset generator (later also segmentation training)
docs/DATASET.md       dataset format contract shared by tools/, eval/ and the tests
```

The UI thread never runs vision code. The frame pump takes one camera frame at a time as an `ImageBitmap`,
transfers it to the worker (no copy) and waits for the result before it takes the next frame. When the worker is
slower than the camera, the extra frames are skipped instead of queued, so the overlay is never more than one
frame behind.

## Deploy to Vercel

1. Push the repo to a **private** GitHub repository.
2. In Vercel, click **Add New… → Project**, then import that repository. The Vite framework preset is detected,
   and `vercel.json` already sets the build command (`npm run build`) and the output directory (`dist`).
3. Deploy. Every push to `main` then deploys to production, and every other branch gets a preview URL.

`vercel.json` also sets the response headers:

- `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` on every route.
  These make the page `crossOriginIsolated`, which enables `SharedArrayBuffer` and so multithreaded WASM.
  Safari does not support `credentialless`, so every asset must be self-hosted: no CDN scripts, fonts or
  images.
- Long-lived `immutable` caching for `/assets/*` (content-hashed build output) and `/models/*`.
- `no-cache` for `sw.js` and `index.html`, so a new deploy is picked up on the next visit. The service worker
  then updates itself (`registerType: 'autoUpdate'`).
- An SPA rewrite to `index.html` for extensionless paths, except `/assets/` and `/models/`.

After deploying, open the production URL on the iPhone and run **Camera check**. The Isolated flag should read
*yes*. To install the app, choose **Share → Add to Home Screen**. A Home Screen app asks for camera permission
each time it starts; this is how iOS works, not a bug.
