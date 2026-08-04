// The canvas renderer registry: the extension seam. A pinned view has a `kind`
// ("plot", "report", "image", …); the canvas looks up a panel component for that
// kind and mounts it. Built-in kinds render in an <iframe> (live/HTML views) or an
// <img> (static images, and a resumed plot's poster). An extension adds a new
// surface — e.g. a MapLibre map to place geothermal wells — by calling
// `registerPanel("map", MapPanel)`; no change to the canvas core is needed.

import { useEffect, useRef, useState } from "react";

import type { View } from "../store";

export interface PanelProps {
  view: View;
  active: boolean;
  /** Changes when the view should reload (a same-slot refresh, or "back"). */
  reloadToken: number;
  /** Call when the panel's content has finished loading (clears the spinner). */
  onLoaded: () => void;
}

export type Panel = (props: PanelProps) => React.ReactElement;

const registry: Record<string, Panel> = {};

export function registerPanel(kind: string, panel: Panel): void {
  registry[kind] = panel;
}

/** A view renders as an image when it is a true image OR its URL points at one —
 *  on resume a live plot falls back to its PNG poster, which must not sit in an
 *  iframe (that shows the browser's bare, mis-sized image viewer). */
export function isImageView(view: View): boolean {
  return view.kind === "image" || /\.(png|jpe?g|gif|svg|webp|bmp)(?:[?#]|$)/i.test(view.url || "");
}

function withToken(url: string, token: number): string {
  if (token <= 0) return url;
  return url + (url.includes("?") ? "&" : "?") + "_=" + token;
}

export function ImagePanel({ view, active, reloadToken, onLoaded }: PanelProps) {
  return (
    <img
      className={active ? "active" : ""}
      src={withToken(view.url, reloadToken)}
      alt={view.title}
      onLoad={onLoaded}
      onError={onLoaded}
    />
  );
}

// A fresh live plot's Bonito route can take a couple of seconds to start
// answering (the Julia kernel only pumps its event loop once something else
// runs it). That doesn't show up as an iframe `error` event: a failed
// navigation (refused/unreachable) still fires `load` (the browser loads its
// own error page as the frame's "content"), and browsers don't fire `error`
// for iframe navigation failures at all. So detect readiness by pinging the
// URL directly and retrying with backoff, instead of mounting the iframe and
// hoping a DOM event tells us it failed.
const LIVE_RETRY_DELAYS_MS = [400, 800, 1200, 2000, 3000];

function useLiveReady(url: string, enabled: boolean, resetKey: number): boolean {
  const [ready, setReady] = useState(!enabled);
  useEffect(() => {
    if (!enabled) {
      setReady(true);
      return;
    }
    setReady(false);
    let cancelled = false;
    let attempt = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tryOnce = () => {
      // HEAD + no-cors: enough to learn whether *something* is listening and
      // responding on this origin, without invoking the actual plot route or
      // reading a response we're not allowed to see cross-origin.
      fetch(url, { method: "HEAD", mode: "no-cors", cache: "no-store" })
        .then(() => {
          if (!cancelled) setReady(true);
        })
        .catch(() => {
          if (cancelled) return;
          if (attempt < LIVE_RETRY_DELAYS_MS.length) {
            const delay = LIVE_RETRY_DELAYS_MS[attempt];
            attempt += 1;
            timer = setTimeout(tryOnce, delay);
          } else {
            setReady(true); // give up; mount anyway so the loader clears
          }
        });
    };
    tryOnce();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [url, enabled, resetKey]);
  return ready;
}

export function IframePanel({ view, active, reloadToken, onLoaded }: PanelProps) {
  // A live WebGL figure reflows once right after `load` (WGLMakie sizes to its
  // parent only after mounting), so clearing the loader on `load` would flash a
  // mis-sized first frame; hold briefly for plots. Reports don't reflow.
  const hold = view.kind === "plot" ? 450 : 0;
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  // Clear a pending hold-timer if the panel unmounts first (tab/canvas closed),
  // so a stale onLoaded can't fire against a view that is already gone.
  useEffect(() => () => clearTimeout(timer.current), []);
  const ready = useLiveReady(view.url, view.kind === "plot", reloadToken);
  const handleLoad = () => {
    if (hold) timer.current = setTimeout(onLoaded, hold);
    else onLoaded();
  };
  if (!ready) return null; // keep the canvas's own loading spinner up while probing
  return (
    <iframe
      className={active ? "active" : ""}
      title={view.title}
      loading="lazy"
      src={withToken(view.url, reloadToken)}
      onLoad={handleLoad}
      onError={onLoaded}
    />
  );
}

export function panelFor(view: View): Panel {
  if (isImageView(view)) return ImagePanel;
  return registry[view.kind] ?? IframePanel;
}
