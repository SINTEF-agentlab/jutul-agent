// The live-frame pool: the registry of mounted live-plot frames, held OUTSIDE
// the session store so the frames survive a session switch (the store resets on
// switch; a live figure supports exactly one frame ever, so unmounting on switch
// would kill it server-side). The pool is the source of truth for which frames
// exist; `LiveFrames` renders them all inside the canvas body, hidden unless
// their session and view are the active ones, and the browser keeps a hidden
// frame's WebSocket and WebGL context alive.

export interface PoolFrame {
  /** Stable render key; a nonce bump makes a new key, so the frame remounts
   *  (which is the reload semantics a same-slot re-plot wants). */
  key: string;
  sessionId: string;
  viewId: string;
  url: string;
  title: string;
  width?: number | null;
  height?: number | null;
  nonce: number;
}

/** How many live frames stay mounted, across every session in the page. Each is
 *  its own WebGL context, and browsers give a page ~16 before evicting; beyond
 *  the cap the oldest frame is released and its view falls back to the poster
 *  (regenerate brings it back). Generous on purpose: this is the browser's
 *  resource budget, not a product limit. */
export const LIVE_VIEW_CAP = 12;

export function frameKey(sessionId: string, viewId: string, url: string, nonce: number): string {
  return `${sessionId}|${viewId}|${url}#${nonce}`;
}

type Listener = () => void;

class FramePool {
  private frames = new Map<string, PoolFrame>();
  private order: string[] = []; // oldest first
  private listeners = new Set<Listener>();
  private snapshotCache: PoolFrame[] = [];

  subscribe = (fn: Listener): (() => void) => {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  };

  /** Stable-reference snapshot for useSyncExternalStore. */
  snapshot = (): PoolFrame[] => this.snapshotCache;

  private emit(): void {
    this.snapshotCache = this.order
      .map((k) => this.frames.get(k))
      .filter((f): f is PoolFrame => !!f);
    for (const fn of this.listeners) fn();
  }

  /** Register a view's live frame. A *higher* nonce on the same view replaces
   *  its frame (a deliberate same-slot re-plot wants the reload); an equal or
   *  lower one adopts the existing frame instead — that is a replayed view
   *  finding its still-alive frame after a session switch, and remounting would
   *  kill the figure. Returns the frames released to stay under the cap, so the
   *  caller can downgrade their views. */
  register(frame: Omit<PoolFrame, "key">): PoolFrame[] {
    for (const [k, f] of this.frames) {
      if (f.sessionId !== frame.sessionId || f.viewId !== frame.viewId) continue;
      if (f.url === frame.url && frame.nonce <= f.nonce) {
        // Adopt: keep the mounted frame, refresh what may have changed.
        this.frames.set(k, { ...f, title: frame.title, width: frame.width, height: frame.height });
        this.order = [...this.order.filter((x) => x !== k), k];
        this.emit();
        return [];
      }
      this.frames.delete(k);
      this.order = this.order.filter((x) => x !== k);
    }
    const key = frameKey(frame.sessionId, frame.viewId, frame.url, frame.nonce);
    this.frames.set(key, { ...frame, key });
    this.order = [...this.order.filter((x) => x !== key), key];
    const released: PoolFrame[] = [];
    while (this.order.length > LIVE_VIEW_CAP) {
      const victim = this.order.shift()!;
      const f = this.frames.get(victim);
      this.frames.delete(victim);
      if (f) released.push(f);
    }
    this.emit();
    return released;
  }

  /** The mounted frame's nonce for a view+URL, or null. A replayed view aligns
   *  its own nonce to this, so its reload token matches the frame that exists. */
  nonceOf(sessionId: string, viewId: string, url: string): number | null {
    for (const f of this.frames.values()) {
      if (f.sessionId === sessionId && f.viewId === viewId && f.url === url) return f.nonce;
    }
    return null;
  }

  release(sessionId: string, viewId: string): void {
    let changed = false;
    for (const [k, f] of this.frames) {
      if (f.sessionId === sessionId && f.viewId === viewId) {
        this.frames.delete(k);
        this.order = this.order.filter((x) => x !== k);
        changed = true;
      }
    }
    if (changed) this.emit();
  }

  /** Whether a replayed live viz still has its original frame (same URL): only
   *  then may the view stay live — a fresh frame on a once-viewed figure comes
   *  up corrupt, so a pool miss means poster + regenerate. */
  has(sessionId: string, viewId: string, url: string): boolean {
    for (const f of this.frames.values()) {
      if (f.sessionId === sessionId && f.viewId === viewId && f.url === url) return true;
    }
    return false;
  }

  clear(): void {
    if (!this.frames.size) return;
    this.frames.clear();
    this.order = [];
    this.emit();
  }
}

export const framePool = new FramePool();
