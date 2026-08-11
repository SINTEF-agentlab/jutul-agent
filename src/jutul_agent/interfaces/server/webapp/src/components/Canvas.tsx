// The canvas: the persistent right-side panel of pinned views. All views stay
// mounted (visibility toggled by the `.active` class) so switching tabs preserves
// each frame's state. Panels come from the canvas registry, so new view kinds plug
// in without touching this component.

import { useEffect, useState } from "react";

import { isImageView, panelFor } from "../canvas/registry";
import { useSel } from "../context";
import { BackIcon, CloseIcon, FillIcon, FitIcon, KindIcon, PopoutIcon } from "../icons";

// The canvas width as a viewport fraction, persisted so the split survives a
// reload. Clamps mirror the CSS min/max so a restored value is always sane.
const CANVAS_W_KEY = "jutul.canvas-w";
const clampFrac = (frac: number) => Math.min(Math.max(frac, 0.3), 0.62);

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
  const views = useSel((s) => s.views);
  const viewOrder = useSel((s) => s.viewOrder);
  const activeView = useSel((s) => s.activeView);
  const canvasOpen = useSel((s) => s.canvasOpen);
  const openView = useSel((s) => s.openView);
  const removeView = useSel((s) => s.removeView);
  const closeCanvas = useSel((s) => s.closeCanvas);
  const setViewMode = useSel((s) => s.setViewMode);

  // Per-(view, reload) "has loaded" set drives the spinner; a new reload token is
  // automatically "not loaded" until its panel fires onLoaded.
  const [loaded, setLoaded] = useState<ReadonlySet<string>>(() => new Set());
  const [backBump, setBackBump] = useState<Record<string, number>>({});

  const tokenOf = (id: string) => (views[id]?.nonce ?? 0) + (backBump[id] ?? 0);
  const loadKey = (id: string) => `${id}@${tokenOf(id)}`;
  const markLoaded = (id: string, token: number) =>
    setLoaded((prev) => new Set(prev).add(`${id}@${token}`));

  const active = activeView ? views[activeView] : null;
  const showLoading = !!active && !loaded.has(loadKey(active.id));

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
          {active && !isImageView(active) ? (
            <button className="icon-btn" title="Back to this view" onClick={reloadActive}>
              <BackIcon />
            </button>
          ) : null}
          <button
            className="icon-btn"
            title="Open in a new tab"
            onClick={() => active && window.open(active.url, "_blank", "noopener")}
          >
            <PopoutIcon />
          </button>
          <button className="icon-btn" title="Close panel" onClick={closeCanvas}>
            <CloseIcon />
          </button>
        </div>
      </div>
      <div className="canvas-body">
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
