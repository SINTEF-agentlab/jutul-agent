import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// This config is intentionally outside the app's `tsc` scope (tsconfig includes
// only `src`): Vite loads it through esbuild at runtime, so the cross-package
// Vite/Vitest plugin-type friction never blocks the build.
//
// The app is served by FastAPI mounted at "/", from ../web_dist. `base: "./"`
// emits relative asset URLs so the bundle works regardless of the mount path.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    outDir: "../web_dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1500,
  },
  server: {
    // `vite dev` proxies API + WebSocket calls to a locally running FastAPI server
    // so the dev UI drives a real backend. Override the target with JA_DEV_BACKEND.
    // The server only answers its own pages, and this one is Vite's, so the proxy
    // drops the page's Origin and calls the server as a plain client.
    proxy: {
      "/sessions": { target: backend(), changeOrigin: true, ws: true, configure: dropOrigin },
      "/models": { target: backend(), changeOrigin: true, configure: dropOrigin },
      "/simulators": { target: backend(), changeOrigin: true, configure: dropOrigin },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
    clearMocks: true,
  },
});

// The proxy's request events, typed loosely: the http-proxy types aren't a dependency.
interface ProxyEvents {
  on(event: "proxyReq" | "proxyReqWs", listener: (req: { removeHeader(name: string): void }) => void): void;
}

function dropOrigin(proxy: ProxyEvents): void {
  proxy.on("proxyReq", (req) => req.removeHeader("origin"));
  proxy.on("proxyReqWs", (req) => req.removeHeader("origin"));
}

function backend(): string {
  return process.env.JA_DEV_BACKEND || "http://127.0.0.1:8181";
}
