// The canvas: the persistent right-side panel of pinned views. All views stay
// mounted (visibility toggled by the `.active` class) so switching tabs preserves
// each frame's state. Panels come from the canvas registry, so new view kinds plug
// in without touching this component.

import { useEffect, useState } from "react";

import { LiveFrames } from "../canvas/LiveFrames";
import { isImageView, panelFor } from "../canvas/registry";
import { CANVAS_W_KEY, clampFrac } from "../canvas/stageSize";
import { useController, useSel } from "../context";
import {
  BackIcon,
  CloseIcon,
  FillIcon,
  FitIcon,
  KindIcon,
  PopoutIcon,
  RegenerateIcon,
} from "../icons";

function restoreCanvasWidth(): void {
  const saved = Number(localStorage.getItem(CANVAS_W_KEY));
  if (Number.isFinite(saved) && saved > 0) {
    document.documentElement.style.setProperty(
      "--canvas-w",
      (clampFrac(saved) * 100).toFixed(1) + "%",
    );
  }
}

export function Canvas() {
  const sessionId = useSel((s) => s.sessionId);
  const views = useSel((s) => s.views);
  const viewOrder = useSel((s) => s.viewOrder);
  const activeView = useSel((s) => s.activeView);
  const canvasOpen = useSel((s) => s.canvasOpen);
  const openView = useSel((s) => s.openView);
  const removeView = useSel((s) => s.removeView);
  const closeCanvas = useSel((s) => s.closeCanvas);
  const setViewMode = useSel((s) => s.setViewMode);
  const controller = useController();

  // Per-(session, view, reload) "has loaded" set drives the spinner; a new
  // reload token is automatically "not loaded" until its panel fires onLoaded.
  // Session-scoped because this component (and the live-frame pool) outlive a
  // session switch, and two sessions can use the same slot name.
  const [loaded, setLoaded] = useState<ReadonlySet<string>>(() => new Set());
  const [backBump, setBackBump] = useState<Record<string, number>>({});

  const tokenOf = (id: string) => (views[id]?.nonce ?? 0) + (backBump[id] ?? 0);
  const loadKey = (id: string, token: number) => `${sessionId}:${id}@${token}`;
  // Recording a load is idempotent: a panel may report the same load more than once
  // (an image settles from both its mount check and its `load` event), and a fresh
  // set for a load already recorded would re-render the canvas for nothing.
  const markLoaded = (id: string, token: number) =>
    setLoaded((prev) => {
      const key = loadKey(id, token);
      return prev.has(key) ? prev : new Set(prev).add(key);
    });

  const active = activeView ? views[activeView] : null;
  const showLoading = !!active && !active.expired && !loaded.has(loadKey(active.id, tokenOf(active.id)));

  useEffect(restoreCanvasWidth, []);

  const onResizeStart = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    const grip = e.currentTarget;
    // Pointer capture routes every move to the grip no matter what is under the
    // cursor — without it the drag dies the moment the pointer crosses a live
    // plot's iframe, whose document swallows the events. The body class also
    // turns iframe pointer-events off for the drag, so an odd browser that drops
    // the capture still can't lose the pointer into a frame.
    grip.setPointerCapture(e.pointerId);
    grip.classList.add("dragging");
    document.body.classList.add("canvas-resizing");
    document.body.style.userSelect = "none";
    let frac = 0;
    const move = (ev: PointerEvent) => {
      // Store the width as a fraction of the viewport, so the split stays
      // proportional across window resizes and screen changes.
      frac = clampFrac((window.innerWidth - ev.clientX) / window.innerWidth);
      document.documentElement.style.setProperty("--canvas-w", (frac * 100).toFixed(1) + "%");
    };
    const up = () => {
      grip.classList.remove("dragging");
      document.body.classList.remove("canvas-resizing");
      document.body.style.userSelect = "";
      grip.removeEventListener("pointermove", move);
      grip.removeEventListener("pointerup", up);
      grip.removeEventListener("pointercancel", up);
      if (frac > 0) localStorage.setItem(CANVAS_W_KEY, String(frac));
      // The split changed the stage: refresh the server's panel-size hint so
      // the next plot (or regenerate) is fitted to the panel as it is now.
      controller.sendCanvasHint();
    };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", up);
    grip.addEventListener("pointercancel", up);
  };

  const reloadActive = () => {
    if (!active) return;
    setBackBump((b) => ({ ...b, [active.id]: (b[active.id] ?? 0) + 1 }));
  };

  return (
    <aside className="canvas" hidden={!canvasOpen}>
      <div className="canvas-grip" title="Drag to resize" onPointerDown={onResizeStart} />
      <div className="canvas-head">
        <div className="canvas-tabs">
          {viewOrder.map((id) => {
            const view = views[id];
            if (!view) return null;
            return (
              <button
                key={id}
                className={`tab${id === activeView ? " active" : ""}`}
                onClick={() => openView(id)}
              >
                <span className="tab-ico">
                  <KindIcon kind={view.kind} />
                </span>
                <span className="tab-label">{view.title}</span>
                <span
                  className="tab-close"
                  title="Remove view"
                  onClick={(e) => {
                    e.stopPropagation();
                    removeView(id);
                  }}
                >
                  <CloseIcon />
                </span>
              </button>
            );
          })}
        </div>
        <div className="canvas-actions">
          {active && active.kind === "plot" && !isImageView(active) && active.width && active.height ? (
            (active.mode ?? "scale") === "scale" ? (
              <button
                className="icon-btn"
                title="Fill the panel (reflow the figure)"
                onClick={() => setViewMode(active.id, "fill")}
              >
                <FillIcon />
              </button>
            ) : (
              <button
                className="icon-btn"
                title="Fit the figure at its own size"
                onClick={() => setViewMode(active.id, "scale")}
              >
                <FitIcon />
              </button>
            )
          ) : null}
          {active && active.kind === "plot" && active.record ? (
            <button
              className="icon-btn"
              title={
                active.live
                  ? "Regenerate (re-run the plot's code)"
                  : "Regenerate this plot (re-run its recorded code)"
              }
              onClick={() => controller.regenerate(active.id)}
            >
              <RegenerateIcon />
            </button>
          ) : null}
          {active && !isImageView(active) && !(active.kind === "plot" && active.live) ? (
            <button className="icon-btn" title="Back to this view" onClick={reloadActive}>
              <BackIcon />
            </button>
          ) : null}
          <button
            className="icon-btn"
            title="Open in its own window"
            onClick={() => active && controller.popout(active.id)}
          >
            <PopoutIcon />
          </button>
          <button className="icon-btn" title="Close panel" onClick={closeCanvas}>
            <CloseIcon />
          </button>
        </div>
      </div>
      <div className="canvas-body">
        {/* Every pooled live frame, across sessions; hidden unless its session
            and view are the active ones. Mounted here so the frames share the
            panel's coordinate space with the ordinary panels below. */}
        <LiveFrames onLoaded={markLoaded} />
        {viewOrder.map((id) => {
          const view = views[id];
          if (!view) return null;
          const token = tokenOf(id);
          const Panel = panelFor(view);
          return (
            <Panel
              key={id}
              view={view}
              active={id === activeView}
              reloadToken={token}
              onLoaded={() => markLoaded(id, token)}
            />
          );
        })}
        {showLoading ? (
          <div className="canvas-loading on">
            <span className="spinner" />
            <span>Loading view…</span>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
