# jutul-agent web UI

The browser UI for `jutul-agent web`: a React + TypeScript app built with Vite. It
talks to the FastAPI server in `../` over REST and one WebSocket per session (the
wire protocol is `../protocol.py`, mirrored in `src/protocol.ts`).

## How it ships

The app is **built ahead of time** into `../web_dist`, which is committed and
shipped in the package, so end users install with `pip`/`uv` and never need Node.
The server serves that build directly (`app.py:_ui_dir`).

**After changing anything in `src/`, rebuild and commit `../web_dist`:**

```sh
npm install        # first time only
npm run build      # tsc typecheck + vite build -> ../web_dist
```

## Develop

Use Node 22.22.2+, 24.15.0+, or 26+ for the current build and test dependencies.

```sh
npm run dev          # Vite dev server with HMR, proxying the API/WebSocket
npm test             # vitest (unit + component tests)
npm run typecheck    # tsc --noEmit
```

`npm run dev` proxies `/sessions`, `/models`, `/simulators` (and the WebSocket) to a
backend at `http://127.0.0.1:8181` — run `jutul-agent web --port 8181` (or set
`JA_DEV_BACKEND`) alongside it to drive a real session.

## Layout

- `src/protocol.ts` — the typed wire contract (mirror of `protocol.py`).
- `src/store.ts` — the render model: a pure, unit-tested store that turns wire
  messages into thread items, canvas views, approvals, and status. No React/DOM.
- `src/transport.ts` — the WebSocket client; coalesces streaming deltas per frame.
- `src/controller.ts` — imperative glue (REST + transport + store); the commands
  components call (send, resume, slash commands, upload, …).
- `src/api.ts` — typed REST calls.
- `src/markdown.tsx` — assistant prose via `react-markdown` + `rehype-sanitize`
  (no raw HTML, so model/tool output can't inject markup).
- `src/ansi.ts`, `src/julia.ts`, `src/toolPolicy.ts` — terminal-output rendering,
  Julia highlighting, and per-tool card policy (all pure, all tested).
- `src/components/` — the UI (Thread, Composer, Canvas, Sidebar, Topbar, …).
- `src/canvas/registry.tsx` — **the extension seam.** A pinned view has a `kind`;
  the canvas looks up a panel for it. Built-ins render in an iframe or an image.
  A new surface (e.g. a MapLibre map to place wells) registers a panel:
  `registerPanel("map", MapPanel)` — no change to the canvas core.
