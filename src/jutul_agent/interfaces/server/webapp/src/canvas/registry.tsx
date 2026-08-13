// The canvas renderer registry: the extension seam. A pinned view has a `kind`
// ("plot", "report", "image", …); the canvas looks up a panel component for that
// kind and mounts it. Built-in kinds render in an <iframe> (live/HTML views) or an
// <img> (static images, and a resumed plot's poster). An extension adds a new
// surface — e.g. a MapLibre map to place geothermal wells — by calling
// `registerPanel("map", MapPanel)`; no change to the canvas core is needed.

import { useEffect, useRef, useState } from "react";

import { useSel } from "../context";
import type { View, ViewMode } from "../store";
import { framePool } from "./framePool";

export interface PanelProps {
  view: View;
  active: boolean;
  /** Changes when the view should reload (a same-slot refresh, or "back"). */
  reloadToken: number;
  /** Call when the panel's content has finished loading (clears the spinner). */
  onLoaded: () => void;
}

export type Panel = (props: PanelProps) => React.ReactElement | null;

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
  // An image the browser has already decoded fires `load` as soon as `src` is set,
  // which can be before React has attached `onLoad` — so the handler never runs and
  // the canvas loader, which only clears from it, stays over a picture that is
  // sitting right there. Seen on a resumed session: the poster complete and 3200px
  // wide behind a permanent "Loading view…". Settling on mount when the element is
  // already `complete` closes that gap; it can only ever clear a loader for content
  // the browser has finished, so an image still arriving is untouched.
  const settle = (img: HTMLImageElement | null) => {
    if (img?.complete) onLoaded();
  };
  return (
    <img
      className={active ? "active" : ""}
      ref={settle}
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

/** The stage rectangle for a figure of design size ``w x h`` in a ``bw x bh``
 *  panel. "scale" fits the figure at its own layout, shrunk as needed (never
 *  upscaled: a canvas grown by CSS blurs) and centered; "fill" hands the figure
 *  the whole panel to reflow into. A stage grown past a *live* figure is not
 *  handled here but by a kernel-side re-fit (see the refit effect below), which
 *  resizes the served figure itself, and this geometry then sees matching sizes. */
export function stageGeometry(
  mode: ViewMode,
  w: number,
  h: number,
  bw: number,
  bh: number,
): { left: number; top: number; width: number; height: number; scale: number } {
  if (mode === "fill" || bw <= 0 || bh <= 0 || w <= 0 || h <= 0) {
    return { left: 0, top: 0, width: bw, height: bh, scale: 1 };
  }
  const scale = Math.min(bw / w, bh / h, 1);
  const width = Math.round(w * scale);
  const height = Math.round(h * scale);
  return {
    left: Math.round((bw - width) / 2),
    top: Math.round((bh - height) / 2),
    width,
    height,
    scale,
  };
}

export interface StagedFrameProps {
  url: string;
  title: string;
  /** The figure's design pixel size; without it the frame just fills. */
  width?: number | null;
  height?: number | null;
  mode: ViewMode;
  active: boolean;
  /** Cache-busting token appended to the URL (a same-slot refresh). */
  token: number;
  /** Probe the URL before mounting (live routes take seconds to answer). */
  probe: boolean;
  /** Hold ms after `load` before reporting loaded (WebGL reflow flash). */
  hold: number;
  /** Put the frame on white instead of the theme colour, for a document
   *  whose page is a centered column with transparent margins, so the frame is
   *  the paper it sits on. Figures want the theme colour: their page is fully
   *  transparent, so the frame is what shows while a resize clears the canvas
   *  (see layout.css) and white there flashes on every re-fit. */
  paper?: boolean;
  onLoaded: () => void;
  /** Called (debounced) when the stage and the figure disagree on size (the
   *  panel grew, shrank, or changed shape) with the stage's settled size; the
   *  caller re-fits the live figure for it. */
  onMismatch?: (width: number, height: number) => void;
}

/** One figure frame inside its stage: the sizing core shared by static plot
 *  views and the pooled live frames. In "scale" mode the frame's inner viewport
 *  is the figure's own design size and a CSS transform shrinks it to fit — the
 *  served layout never reflows, so a panel resize only changes the transform.
 *  In "fill" the figure reflows to the whole stage. */
export function StagedFrame({
  url,
  title,
  width,
  height,
  mode,
  active,
  token,
  probe,
  hold,
  paper,
  onLoaded,
  onMismatch,
}: StagedFrameProps) {
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  // Clear a pending hold-timer if the panel unmounts first (tab/canvas closed),
  // so a stale onLoaded can't fire against a view that is already gone.
  useEffect(() => () => clearTimeout(timer.current), []);
  const ready = useLiveReady(url, probe, token);

  // The stage's own size, tracked while visible (hidden it measures 0x0, so the
  // activation re-measure is what sizes a tab the user switches to).
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    // A hidden stage measures 0x0, so keep the last real size instead of
    // adopting it. The frame's geometry follows this box, and collapsing a
    // hidden frame to zero is not cosmetic: the served figure tracks its
    // parent's size, so a zeroed frame resizes the live figure itself while
    // its canvas is throttled, which is how a tab switched away during
    // resizes came back torn (axes moved, plot primitives stale).
    const measure = () => {
      if (el.clientWidth > 0 && el.clientHeight > 0) {
        setBox({ w: el.clientWidth, h: el.clientHeight });
      }
    };
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    measure();
    return () => ro.disconnect();
  }, [active, ready]);

  // A one-frame height nudge after load in fill mode: the served figure applies
  // its parent's width on mount but not always its height, and its own observer
  // only fires on a later resize — so cause one.
  const [nudge, setNudge] = useState(0);
  const sized = !!width && !!height;

  // A kernel re-fit re-lays the figure out, which lands as a second, visibly
  // later update after the instant CSS scale, so it must earn that jump: only
  // when the scaled figure leaves a good share of the panel unused (a shape
  // mismatch or real growth) is a re-fit requested, after the size settles. A
  // view already covering most of its panel is left exactly as it is, and the
  // one re-fit that fires converges to the server rule's optimum in a single
  // step (a full panel, or a gentle letterbox at the distortion bound, the
  // stage-size dedupe in refitView keeps that from re-firing forever).
  useEffect(() => {
    if (!onMismatch || !sized || !active || mode !== "scale") return;
    if (box.w < 100 || box.h < 100) return;
    const scale = Math.min(box.w / width!, box.h / height!, 1);
    const coverage = (width! * scale * (height! * scale)) / (box.w * box.h);
    if (coverage >= 0.75) return;
    const t = setTimeout(() => onMismatch(box.w, box.h), 1000);
    return () => clearTimeout(t);
  }, [onMismatch, sized, active, mode, box.w, box.h, width, height]);

  const handleLoad = () => {
    if (hold) timer.current = setTimeout(onLoaded, hold);
    else onLoaded();
    if (mode === "fill" && probe) {
      setNudge(1);
      setTimeout(() => setNudge(0), 80);
    }
  };

  // Never first-mount a frame hidden: the served figure measures its parent on
  // mount, and a hidden frame measures zero. Once shown it stays mounted, so tab
  // switches keep the frame's connection and camera.
  const wasActive = useRef(false);
  if (active) wasActive.current = true;
  if (!ready) return null; // keep the canvas's own loading spinner up while probing
  if (!active && !wasActive.current) return null;

  const g = sized ? stageGeometry(mode, width, height, box.w, box.h) : null;
  const scaled = g !== null && mode === "scale" && g.scale < 1;
  const frameStyle: React.CSSProperties =
    g && mode === "scale"
      ? { left: g.left, top: g.top, width: g.width, height: g.height }
      : { inset: 0, height: nudge ? `calc(100% - ${nudge}px)` : undefined };
  return (
    <div ref={stageRef} className={`canvas-stage${active ? " active" : ""}`}>
      <div className={`stage-frame${paper ? " paper" : ""}`} style={frameStyle}>
        <iframe
          title={title}
          loading="lazy"
          src={withToken(url, token)}
          onLoad={handleLoad}
          onError={onLoaded}
          style={
            scaled
              ? {
                  width: width ?? undefined,
                  height: height ?? undefined,
                  transform: `scale(${g.scale})`,
                  transformOrigin: "top left",
                }
              : undefined
          }
        />
      </div>
    </div>
  );
}

/** A live view whose figure is gone and that has no poster to fall back on. */
export function ExpiredPanel({ view, active }: PanelProps) {
  if (!active) return null;
  return (
    <div className="canvas-empty">
      <span>
        {view.title} is no longer live.
        {view.record ? " Use the regenerate button above to re-run its code." : ""}
      </span>
    </div>
  );
}

export function IframePanel({ view, active, reloadToken, onLoaded }: PanelProps) {
  const sessionId = useSel((s) => s.sessionId);
  const downgradeView = useSel((s) => s.downgradeView);
  const live = !!view.live && view.kind === "plot";

  // A live figure supports exactly one frame ever, so its frame lives in the
  // pool (rendered by LiveFrames), which survives a session switch; this panel
  // only registers it. A frame released to stay under the pool cap downgrades
  // its view to the poster (when it is this session's view to downgrade).
  useEffect(() => {
    if (!live || !sessionId) return;
    const released = framePool.register({
      sessionId,
      viewId: view.id,
      url: view.url,
      title: view.title,
      width: view.width,
      height: view.height,
      nonce: view.nonce,
    });
    for (const f of released) {
      if (f.sessionId === sessionId) downgradeView(f.viewId);
    }
  }, [live, sessionId, view.id, view.url, view.title, view.width, view.height, view.nonce, downgradeView]);

  if (live) return null;
  // A live WebGL figure reflows once right after `load` (WGLMakie sizes to its
  // parent only after mounting), so clearing the loader on `load` would flash a
  // mis-sized first frame; hold briefly for plots. Reports don't reflow.
  const sized = view.kind === "plot" && !!view.width && !!view.height;
  return (
    <StagedFrame
      url={view.url}
      title={view.title}
      width={view.width}
      height={view.height}
      mode={sized ? (view.mode ?? "scale") : "fill"}
      active={active}
      token={reloadToken}
      probe={view.kind === "plot"}
      hold={view.kind === "plot" ? 450 : 0}
      paper={view.kind !== "plot"}
      onLoaded={onLoaded}
    />
  );
}

export function panelFor(view: View): Panel {
  if (view.expired) return ExpiredPanel;
  if (isImageView(view)) return ImagePanel;
  return registry[view.kind] ?? IframePanel;
}
