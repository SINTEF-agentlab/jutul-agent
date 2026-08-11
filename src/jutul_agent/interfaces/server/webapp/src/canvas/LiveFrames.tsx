// Renders every pooled live frame inside the canvas body. All frames stay
// mounted — across tab switches AND session switches — because a live figure
// supports exactly one frame for its lifetime; only the frame belonging to the
// active session's active view is shown. The pool is module state, so the
// per-session store reset (a sidebar switch) never unmounts a frame.

import { useSyncExternalStore } from "react";

import { useSel } from "../context";
import { StagedFrame } from "./registry";
import { framePool } from "./framePool";

export function LiveFrames({
  onLoaded,
}: {
  onLoaded: (viewId: string, nonce: number) => void;
}) {
  const frames = useSyncExternalStore(framePool.subscribe, framePool.snapshot);
  const sessionId = useSel((s) => s.sessionId);
  const activeView = useSel((s) => s.activeView);
  const canvasOpen = useSel((s) => s.canvasOpen);
  const views = useSel((s) => s.views);
  return (
    <>
      {frames.map((f) => {
        const view = f.sessionId === sessionId ? views[f.viewId] : null;
        const active =
          canvasOpen && f.sessionId === sessionId && f.viewId === activeView && !!view?.live;
        const sized = !!f.width && !!f.height;
        return (
          <StagedFrame
            key={f.key}
            url={f.url}
            title={f.title}
            width={f.width}
            height={f.height}
            mode={sized ? (view?.mode ?? "scale") : "fill"}
            active={active}
            token={f.nonce}
            probe
            hold={450}
            onLoaded={() => onLoaded(f.viewId, f.nonce)}
          />
        );
      })}
    </>
  );
}
