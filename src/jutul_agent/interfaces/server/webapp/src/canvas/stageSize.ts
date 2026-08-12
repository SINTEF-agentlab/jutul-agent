// Where the client answers "how big is the plot area?" — both for the live
// hint that shapes new figures server-side and for the target a regenerate
// re-fits to. One module so the persisted-split key and the layout constants
// live in a single place (Canvas.tsx reads them for the grip, the controller
// for the hint).

/** The canvas width as a viewport fraction, persisted so the split survives a
 *  reload. Clamps mirror the CSS min/max so a restored value is always sane. */
export const CANVAS_W_KEY = "jutul.canvas-w";
export const clampFrac = (frac: number) => Math.min(Math.max(frac, 0.3), 0.62);

// Mirrors layout.css: the default --canvas-w, the breakpoint below which the
// canvas overlays at a fixed width, and the header row above the stage.
const DEFAULT_FRAC = 0.46;
const OVERLAY_BREAKPOINT = 1024;
const OVERLAY_WIDTH = 640;
const HEAD_HEIGHT = 48;

/** The plot stage's size in CSS pixels: measured when one is visible, else the
 *  size the stage *will* have when the canvas opens (computed from the same
 *  split fraction and breakpoints the CSS uses). The prospective answer is what
 *  makes the first plot of a session land already shaped for the panel. */
export function stageSize(): { width: number; height: number } {
  const active = document.querySelector<HTMLElement>(".canvas-stage.active");
  if (active && active.clientWidth >= 100 && active.clientHeight >= 100) {
    return { width: active.clientWidth, height: active.clientHeight };
  }
  const winW = window.innerWidth;
  const winH = window.innerHeight;
  const saved = Number(localStorage.getItem(CANVAS_W_KEY));
  const frac = Number.isFinite(saved) && saved > 0 ? clampFrac(saved) : DEFAULT_FRAC;
  const width =
    winW <= OVERLAY_BREAKPOINT ? Math.min(OVERLAY_WIDTH, winW) : Math.round(winW * frac);
  return { width, height: Math.max(0, winH - HEAD_HEIGHT) };
}
