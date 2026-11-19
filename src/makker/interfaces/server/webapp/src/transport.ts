// The per-session WebSocket transport. It owns the socket, parses frames into
// typed messages, and feeds them to the store. Text and reasoning deltas are
// coalesced to one store update per animation frame: re-parsing a long, growing
// reply on every token-chunk is O(N^2) and janks the page, so adjacent same-kind
// deltas are merged and applied once per frame. Everything else applies at once,
// after flushing any queued deltas, so ordering is preserved.

import type { StoreApi } from "zustand/vanilla";

import { parseServerMessage, type ClientMessage, type ServerMessage } from "./protocol";
import type { SessionStore } from "./store";
import { withBase } from "./basePath";

type Effects = (msg: ServerMessage) => void;
type OnDrop = () => void;
type OnOpen = () => void;
type Scheduler = (cb: () => void) => number;
type Canceller = (handle: number) => void;

interface QueuedDelta {
  type: "text" | "reasoning";
  text: string;
}

export class Transport {
  private ws: WebSocket | null = null;
  private queue: QueuedDelta[] = [];
  private frame = 0;
  // Client messages sent before the socket is open; flushed once it opens.
  private outbox: ClientMessage[] = [];

  constructor(
    private store: StoreApi<SessionStore>,
    private effects: Effects,
    // Called when the socket closes unexpectedly (not via `close`), so the
    // controller can reconnect to the session.
    private onDrop: OnDrop,
    // Called once the socket opens and its buffered messages have been flushed.
    private onOpen: OnOpen,
    // Injectable so tests can flush synchronously; defaults to rAF.
    private schedule: Scheduler = (cb) => requestAnimationFrame(cb),
    private cancel: Canceller = (h) => cancelAnimationFrame(h),
  ) {}

  open(sessionId: string): void {
    this.close();
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const path = withBase(`/sessions/${sessionId}/stream`);
    const ws = new WebSocket(`${proto}://${location.host}${path}`);
    ws.onopen = () => {
      this.store.setState({ reconnecting: false });
      // Flush anything queued while connecting (e.g. an example clicked the instant
      // the page loaded, or a prompt typed during a reconnect).
      const pending = this.outbox;
      this.outbox = [];
      for (const msg of pending) ws.send(JSON.stringify(msg));
      this.onOpen(); // the buffer is delivered; the controller can drop its durable copy
    };
    ws.onmessage = (e) => {
      const msg = parseServerMessage(typeof e.data === "string" ? e.data : "");
      if (msg) this.apply(msg);
    };
    ws.onclose = () => {
      this.flush();
      this.ws = null;
      this.store.setState({ busy: false, working: false });
      // `close` detaches this handler before closing, so only an unexpected drop
      // reaches here. Let the controller decide how to get back to the session.
      this.onDrop();
    };
    this.ws = ws;
    // The kernel warms in the background on a fresh/resumed session until the
    // first turn lands; show the hint until any message arrives.
    this.store.getState().setWarming(true);
  }

  close(): void {
    this.flush();
    this.outbox = [];
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onclose = null;
      this.ws.onmessage = null;
      try {
        this.ws.close();
      } catch {
        /* already closing */
      }
      this.ws = null;
    }
  }

  isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /** Deliver a message, or buffer it while the socket connects. Returns false when
   *  there is no socket (closed or not opened yet), so the caller can hold the
   *  message and trigger a reconnect. */
  send(msg: ClientMessage): boolean {
    const ws = this.ws;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
      return true;
    }
    if (ws?.readyState === WebSocket.CONNECTING) {
      this.outbox.push(msg); // delivered by onopen
      return true;
    }
    return false;
  }

  private apply(msg: ServerMessage): void {
    if (msg.type === "text" || msg.type === "reasoning") {
      this.enqueue(msg.type, msg.text);
      return;
    }
    this.flush(); // keep deltas ahead of the message that follows them
    // A host app embedding the UI can consume a `ui` action (wire it to its own
    // controls) by returning true from window.onJutulUi; then we don't surface it.
    if (msg.type === "ui" && window.onJutulUi?.(msg) === true) return;
    this.store.getState().handle(msg);
    this.effects(msg);
  }

  private enqueue(type: "text" | "reasoning", text: string): void {
    const last = this.queue.at(-1);
    if (last && last.type === type) last.text += text;
    else this.queue.push({ type, text });
    if (!this.frame) this.frame = this.schedule(() => this.flush());
  }

  private flush(): void {
    if (this.frame) {
      this.cancel(this.frame);
      this.frame = 0;
    }
    if (!this.queue.length) return;
    const queued = this.queue;
    this.queue = [];
    const handle = this.store.getState().handle;
    for (const item of queued) handle({ type: item.type, text: item.text });
  }
}
