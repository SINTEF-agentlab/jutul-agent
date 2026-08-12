// The session store: one place that turns wire messages into the render model.
// It is a plain Zustand store with no React or DOM dependency, so every state
// transition (streaming text, tool lifecycle, canvas views, approvals) is unit
// tested directly. The transport calls `handle`; React reads slices of the state.

import { createStore } from "zustand/vanilla";

import type { CredentialInfo, HistoryEntry, ModelInfo, SimDetails } from "./api";
import { framePool } from "./canvas/framePool";
import { formatTokens } from "./format";
import type { InterruptAction, ReplayMessage, ServerMessage } from "./protocol";
import { HISTORY_CHANGED, SIDE_OUTPUT_TYPES } from "./protocol";
import { toolPolicy } from "./toolPolicy";

export type ViewKind = "plot" | "report" | "image";

/** How a live figure sits in the panel. "scale": rendered at its own design
 *  size and scaled down to fit, so a wide dashboard's layout and text are never
 *  crushed (the default when the figure's size is known). "fill": the figure
 *  reflows to the panel's own shape and uses all of it. */
export type ViewMode = "scale" | "fill";

export interface View {
  id: string;
  url: string;
  title: string;
  kind: ViewKind;
  poster?: string | null;
  /** Bumped when a same-slot view is refreshed, to force its frame to reload. */
  nonce: number;
  /** Served by the session's live figure server. A live figure supports exactly
   *  one frame — a second view (or a reload after the first closed) corrupts and
   *  then kills it — so live frames stay mounted while live, and popout serves
   *  an independent replayed figure instead of this URL. */
  live?: boolean;
  /** A live view whose figure was released (closed, capped, or the page was
   *  reloaded). Its URL is dead; the panel shows the poster or an explanation,
   *  and `record` is how it comes back. */
  expired?: boolean;
  /** The plot's trace record when its source code was recorded: what regenerate
   *  and popout send back in a `replot` request. */
  record?: string | null;
  /** The figure's own pixel size (the scale stage's geometry). */
  width?: number | null;
  height?: number | null;
  mode?: ViewMode;
  /** Tab closed, record kept: the thread's chip reopens it (a closed live view
   *  comes back as its poster plus the regenerate button). */
  closed?: boolean;
}

export type ToolStatus = "running" | "done" | "error";

export type ThreadItem =
  | { kind: "user"; id: string; text: string }
  | { kind: "assistant"; id: string; text: string }
  | { kind: "reasoning"; id: string; text: string; live: boolean }
  | {
      kind: "tool";
      id: string;
      toolCallId: string;
      name: string | null;
      label?: string;
      args?: Record<string, unknown> | null;
      status: ToolStatus;
      output: string;
      note?: string;
    }
  | { kind: "viz-chip"; id: string; viewId: string; title: string; viewKind: ViewKind }
  | { kind: "artifact-image"; id: string; url: string; title: string }
  | { kind: "artifact-file"; id: string; url: string; caption: string }
  | { kind: "sys-note"; id: string; text: string; level?: "warn" }
  | { kind: "ui-note"; id: string; action: string; payload: Record<string, unknown> }
  | { kind: "error"; id: string; message: string; canRetry: boolean }
  | { kind: "help"; id: string }
  | { kind: "context"; id: string; markdown: string };

export interface PendingInterrupt {
  actions: InterruptAction[];
  allowed: string[];
  allowlist: string[];
}

/** A provider whose key the server is waiting on (a blocking key prompt). */
export interface CredentialPrompt {
  provider: string;
  label: string;
  env_var: string;
}

/** State of the API-keys modal: open in "manage" mode (required null) or because a
 *  specific provider's key is needed before the session/model switch can proceed. */
export interface ApiKeysModal {
  open: boolean;
  required: CredentialPrompt | null;
}

export interface SessionState {
  // identity / config
  sessionId: string | null;
  sim: string | null;
  simDetails: Record<string, SimDetails>;
  model: string | null;
  models: ModelInfo[];
  contextWindow: number | null;
  meta: string;
  // status
  busy: boolean;
  warming: boolean;
  working: boolean;
  // the socket dropped and we are re-establishing it (shows the reconnecting bar)
  reconnecting: boolean;
  // thread
  items: ThreadItem[];
  liveAssistantId: string | null;
  liveReasoningId: string | null;
  // bumped to force the conversation to scroll to the bottom (e.g. on a sent message)
  bottomPin: number;
  // canvas
  views: Record<string, View>;
  viewOrder: string[];
  activeView: string | null;
  canvasOpen: boolean;
  // approval
  pending: PendingInterrupt | null;
  // usage
  inputTokens: number;
  usageLabel: string;
  usageTitle: string;
  // history + retry
  history: HistoryEntry[];
  lastPrompt: string;
  // provider API keys (status + the key-prompt modal)
  credentials: CredentialInfo[];
  apiKeys: ApiKeysModal;
}

export interface SessionActions {
  handle: (msg: ServerMessage) => void;
  replay: (messages: ReplayMessage[]) => void;
  // lifecycle / config
  setConfig: (patch: Partial<SessionState>) => void;
  setSession: (id: string, meta: string) => void;
  setModel: (model: string) => void;
  setContextWindow: (window: number | null) => void;
  setHistory: (history: HistoryEntry[]) => void;
  setWarming: (on: boolean) => void;
  setCredentials: (credentials: CredentialInfo[]) => void;
  openApiKeys: (required: CredentialPrompt | null) => void;
  closeApiKeys: () => void;
  reset: () => void;
  // composer-driven
  addUser: (text: string) => void;
  startTurn: (text: string) => void;
  beginWorking: () => void;
  pinBottom: () => void;
  clearInterrupt: () => void;
  addSysNote: (text: string, level?: "warn") => void;
  addHelp: () => void;
  addContext: (markdown: string) => void;
  // canvas
  openView: (id: string) => void;
  openImage: (url: string, title: string) => void;
  closeCanvas: () => void;
  removeView: (id: string) => void;
  pinDoc: (url: string, title: string, slot: string) => void;
  setViewMode: (id: string, mode: ViewMode) => void;
  /** Record a live figure's new served size (a kernel-side re-fit landed). */
  setViewSize: (id: string, width: number, height: number) => void;
  downgradeView: (id: string) => void;
}

export type SessionStore = SessionState & SessionActions;

const initialState: SessionState = {
  sessionId: null,
  sim: null,
  simDetails: {},
  model: null,
  models: [],
  contextWindow: null,
  meta: "",
  busy: false,
  warming: false,
  working: false,
  reconnecting: false,
  items: [],
  liveAssistantId: null,
  liveReasoningId: null,
  views: {},
  viewOrder: [],
  activeView: null,
  canvasOpen: false,
  pending: null,
  inputTokens: 0,
  usageLabel: "",
  usageTitle: "",
  history: [],
  lastPrompt: "",
  bottomPin: 0,
  credentials: [],
  apiKeys: { open: false, required: null },
};

function viewIdOf(msg: { slot?: string | null; url: string }): string {
  return msg.slot ? `slot:${msg.slot}` : `url:${msg.url}`;
}

export function createSessionStore() {
  let seq = 0;
  const nextId = () => `i${++seq}`;

  return createStore<SessionStore>()((set, get) => {
    // --- internal item helpers (return new arrays; unchanged items keep refs) ---

    const finalizeReasoning = (items: ThreadItem[], liveId: string | null): ThreadItem[] =>
      liveId
        ? items.map((it) =>
            it.id === liveId && it.kind === "reasoning" ? { ...it, live: false } : it,
          )
        : items;

    const onText = (delta: string) => {
      if (!delta) return;
      set((s) => {
        if (s.liveAssistantId) {
          const id = s.liveAssistantId;
          return {
            working: false,
            items: s.items.map((it) =>
              it.id === id && it.kind === "assistant" ? { ...it, text: it.text + delta } : it,
            ),
          };
        }
        const items = finalizeReasoning(s.items, s.liveReasoningId);
        const id = nextId();
        return {
          working: false,
          liveReasoningId: null,
          liveAssistantId: id,
          items: [...items, { kind: "assistant", id, text: delta }],
        };
      });
    };

    const finalizeAssistant = () => set({ liveAssistantId: null });

    const onReasoning = (delta: string) => {
      if (!delta) return;
      set((s) => {
        if (s.liveReasoningId) {
          const id = s.liveReasoningId;
          return {
            working: false,
            items: s.items.map((it) =>
              it.id === id && it.kind === "reasoning" ? { ...it, text: it.text + delta } : it,
            ),
          };
        }
        const id = nextId();
        return {
          working: false,
          // A reasoning block ends the current assistant segment, so later text starts
          // a fresh bubble after it (not appended back into the pre-reasoning one).
          liveAssistantId: null,
          liveReasoningId: id,
          items: [...s.items, { kind: "reasoning", id, text: delta, live: true }],
        };
      });
    };

    const onTool = (msg: Extract<ServerMessage, { type: "tool" }>) => {
      const cid = msg.tool_call_id;
      if (!cid) return;
      const policy = toolPolicy(msg.name);
      set((s) => {
        let items = s.items;
        let exists = items.some((it) => it.kind === "tool" && it.toolCallId === cid);
        if (!exists) {
          // A new tool step closes any open assistant/reasoning segment first.
          items = finalizeReasoning(items, s.liveReasoningId);
          items = [
            ...items,
            {
              kind: "tool",
              id: nextId(),
              toolCallId: cid,
              name: msg.name,
              label: msg.label,
              args: msg.args ?? null,
              status: "running",
              output: "",
            },
          ];
          exists = true;
        }
        const update = (patch: Partial<Extract<ThreadItem, { kind: "tool" }>>) =>
          items.map((it) =>
            it.kind === "tool" && it.toolCallId === cid ? { ...it, ...patch } : it,
          );

        if (msg.event === "delta") {
          if (msg.content != null && policy.rawOutput !== false) {
            items = update({ output: msg.content });
          }
        } else if (msg.event === "finished") {
          // A tool that reports failure in its reply (an "ERROR: ..." string
          // the model reads) still "finishes" at the protocol level; the policy
          // predicate is what keeps the card honest, and the reply is surfaced
          // even for tools whose successful output is hidden.
          const failed = !!(msg.content && policy.failed?.(msg.content));
          const patch: Partial<Extract<ThreadItem, { kind: "tool" }>> = {
            status: failed ? "error" : "done",
          };
          if (!failed && policy.note && msg.content) patch.note = policy.note(msg.content);
          if (msg.content && (failed || policy.rawOutput !== false)) patch.output = msg.content;
          items = update(patch);
        } else if (msg.event === "error") {
          const patch: Partial<Extract<ThreadItem, { kind: "tool" }>> = { status: "error" };
          if (msg.content) patch.output = msg.content; // always surface errors
          items = update(patch);
        }
        return {
          working: false,
          liveAssistantId: null,
          liveReasoningId: null,
          items,
        };
      });
    };

    const upsertView = (view: View, replace: boolean) =>
      set((s) => {
        const existing = s.views[view.id];
        // A fresh viz for a closed view re-pins its tab (a re-plot or a
        // regenerate on a slot the user had closed).
        const next: View = existing
          ? { ...existing, ...view, closed: false, nonce: replace ? existing.nonce + 1 : existing.nonce }
          : view;
        return {
          views: { ...s.views, [view.id]: next },
          viewOrder: s.viewOrder.includes(view.id) ? s.viewOrder : [...s.viewOrder, view.id],
        };
      });

    const onViz = (msg: Extract<ServerMessage, { type: "viz" }>, replayed = false) => {
      finalizeAssistant();
      const id = viewIdOf(msg);
      const kind: ViewKind = msg.kind === "report" ? "report" : "plot";
      const title = msg.title || (kind === "report" ? "Report" : "Interactive plot");
      let live = !!msg.live;
      let url = msg.url;
      let expired = false;
      let nonce = 0;
      if (replayed) {
        // A replayed live view may only stay live when its original frame still
        // exists in the pool (an in-page session switch): a fresh frame on a
        // once-viewed figure comes up corrupt. Adopting aligns the nonce with
        // the mounted frame; a pool miss (a reload, a cold resume) falls back
        // to the poster and the regenerate button.
        const sid = get().sessionId ?? "";
        const pooled = live ? framePool.nonceOf(sid, id, msg.url) : null;
        if (live && pooled !== null) {
          nonce = pooled;
        } else if (live) {
          live = false;
          if (msg.poster) url = msg.poster;
          else expired = true;
        }
        // A non-live replay whose view still has a pooled frame: that frame
        // belongs to a figure the server no longer serves (the kernel
        // restarted); release it rather than adopt it later.
        if (!live) framePool.release(sid, id);
      }
      upsertView(
        {
          id,
          url,
          title,
          kind,
          poster: msg.poster ?? null,
          nonce,
          live,
          // A refresh replaces the view wholesale: a re-plot or a regenerate
          // revives an expired view with a fresh figure.
          expired,
          record: msg.record ?? null,
          width: msg.width ?? null,
          height: msg.height ?? null,
          // The upsert spreads the patch over the old view, so carry the mode the
          // user chose for this slot across the refresh explicitly.
          mode: get().views[id]?.mode,
        },
        true,
      );
      set((s) => ({
        items: [...s.items, { kind: "viz-chip", id: nextId(), viewId: id, title, viewKind: kind }],
      }));
      get().openView(id);
    };

    const onArtifact = (msg: Extract<ServerMessage, { type: "artifact" }>) => {
      finalizeAssistant();
      if (msg.mime && msg.mime.startsWith("image/")) {
        // No view yet: its card shows the image in full, so a canvas tab up front
        // is one per image nobody asked for. `openImage` registers one on demand.
        const title = msg.caption || "Image";
        set((s) => ({
          items: [...s.items, { kind: "artifact-image", id: nextId(), url: msg.url, title }],
        }));
      } else {
        set((s) => ({
          items: [
            ...s.items,
            { kind: "artifact-file", id: nextId(), url: msg.url, caption: msg.caption || "Artifact" },
          ],
        }));
      }
    };

    const onUi = (msg: Extract<ServerMessage, { type: "ui" }>) => {
      // history_changed is an internal refresh signal, handled by the controller.
      if (msg.action === HISTORY_CHANGED) return;
      finalizeAssistant();
      set((s) => ({
        items: [...s.items, { kind: "ui-note", id: nextId(), action: msg.action, payload: msg.payload }],
      }));
    };

    const onInterrupt = (msg: Extract<ServerMessage, { type: "interrupt" }>) => {
      finalizeAssistant();
      set({
        working: false,
        busy: false, // the turn is paused on the user; free the composer
        pending: {
          actions: msg.actions,
          allowed: msg.allowed_decisions,
          allowlist: msg.allowlist,
        },
      });
    };

    const onUsage = (msg: Extract<ServerMessage, { type: "usage" }>) => {
      set((s) => {
        const inputTokens = msg.input_tokens || s.inputTokens;
        return { inputTokens, ...usageLabels(inputTokens, s.contextWindow) };
      });
    };

    const onTurnEnd = () => {
      set((s) => ({
        busy: false,
        working: false,
        liveAssistantId: null,
        // collapse the live reasoning block once the turn is done
        items: finalizeReasoning(s.items, s.liveReasoningId),
        liveReasoningId: null,
      }));
    };

    const onError = (message: string) => {
      finalizeAssistant();
      set((s) => ({
        busy: false,
        working: false,
        items: [
          ...s.items,
          { kind: "error", id: nextId(), message, canRetry: !!s.lastPrompt && !s.pending },
        ],
      }));
    };

    const onNotice = (text: string) => {
      // A command's result (e.g. /compact, /add-dir): the command finished.
      set((s) => ({
        busy: false,
        working: false,
        items: [...s.items, { kind: "sys-note", id: nextId(), text }],
      }));
    };

    return {
      ...initialState,

      handle(msg) {
        // Anything but a side output means the agent produced content, so the
        // "thinking" indicator clears. Side outputs (a plot, a usage tick) arrive
        // mid-turn while the agent keeps working, so they must not clear it.
        if (!SIDE_OUTPUT_TYPES.has(msg.type)) set({ working: false });
        if (get().warming) set({ warming: false });
        switch (msg.type) {
          case "text":
            return onText(msg.text);
          case "reasoning":
            return onReasoning(msg.text);
          case "tool":
            return onTool(msg);
          case "viz":
            return onViz(msg);
          case "artifact":
            return onArtifact(msg);
          case "interrupt":
            return onInterrupt(msg);
          case "usage":
            return onUsage(msg);
          case "turn_end":
            return onTurnEnd();
          case "ui":
            return onUi(msg);
          case "popout_ready":
            return; // the controller navigates the waiting popup window
          case "notice":
            return onNotice(msg.text);
          case "error":
            return onError(msg.message);
        }
      },

      replay(messages) {
        for (const m of messages) {
          switch (m.type) {
            case "user":
              get().addUser(m.text);
              break;
            case "assistant":
              if (m.text)
                set((s) => ({
                  liveAssistantId: null,
                  items: [...s.items, { kind: "assistant", id: nextId(), text: m.text }],
                }));
              break;
            case "reasoning":
              if (m.text)
                set((s) => ({
                  items: [...s.items, { kind: "reasoning", id: nextId(), text: m.text, live: false }],
                }));
              break;
            case "tool":
              onTool(m);
              break;
            case "viz":
              onViz(m, true);
              break;
            case "artifact":
              onArtifact(m);
              break;
          }
        }
        set({ liveAssistantId: null, liveReasoningId: null });
      },

      setConfig: (patch) => set(patch),
      setSession: (id, meta) => set({ sessionId: id, meta }),
      setModel: (model) => set({ model }),
      setContextWindow: (window) =>
        set((s) => ({ contextWindow: window, ...usageLabels(s.inputTokens, window) })),
      setHistory: (history) => set({ history }),
      setWarming: (on) => set({ warming: on }),
      setCredentials: (credentials) => set({ credentials }),
      openApiKeys: (required) => set({ apiKeys: { open: true, required } }),
      closeApiKeys: () => set({ apiKeys: { open: false, required: null } }),

      reset: () =>
        set((s) => ({
          ...initialState,
          // keep the connection-independent config
          sim: s.sim,
          simDetails: s.simDetails,
          model: s.model,
          models: s.models,
          contextWindow: s.contextWindow,
          history: s.history,
          // key status is account-wide, not per-session; keep it across a reset
          credentials: s.credentials,
          // a reconnect resets the thread mid-recovery; keep the bar until the socket
          // reopens (the new socket's onopen clears it)
          reconnecting: s.reconnecting,
        })),

      addUser: (text) =>
        set((s) => ({
          liveAssistantId: null,
          items: [...s.items, { kind: "user", id: nextId(), text }],
        })),

      startTurn: (text) =>
        set((s) => ({
          busy: true,
          working: true,
          lastPrompt: text,
          liveAssistantId: null,
          items: [...s.items, { kind: "user", id: nextId(), text }],
        })),

      beginWorking: () => set({ busy: true, working: true }),

      pinBottom: () => set((s) => ({ bottomPin: s.bottomPin + 1 })),

      clearInterrupt: () => set({ pending: null }),

      addSysNote: (text, level) =>
        set((s) => ({ items: [...s.items, { kind: "sys-note", id: nextId(), text, level }] })),

      addHelp: () => set((s) => ({ items: [...s.items, { kind: "help", id: nextId() }] })),

      addContext: (markdown) =>
        set((s) => ({ items: [...s.items, { kind: "context", id: nextId(), markdown }] })),

      openView: (id) =>
        set((s) => {
          const v = s.views[id];
          if (!v) return {};
          // The thread's chip is the way back after a tab was closed: reopening
          // re-pins it (downgraded already if it was live; regenerate revives it).
          const views = v.closed ? { ...s.views, [id]: { ...v, closed: false } } : s.views;
          const viewOrder = s.viewOrder.includes(id) ? s.viewOrder : [...s.viewOrder, id];
          return { views, viewOrder, activeView: id, canvasOpen: true };
        }),

      openImage: (url, title) => {
        const id = viewIdOf({ url });
        upsertView({ id, url, title, kind: "image", poster: url, nonce: 0 }, false);
        get().openView(id);
      },

      closeCanvas: () => set({ canvasOpen: false }),

      removeView: (id) => {
        // Removing the tab releases its live frame too; the figure's recorded
        // code (the regenerate button on a re-pinned view) is how it comes back.
        const sid = get().sessionId;
        if (sid) framePool.release(sid, id);
        set((s) => {
          const v = s.views[id];
          if (!v) return {};
          // Keep the record, marked closed, so the thread's chip can reopen it.
          // A live view downgrades on close: its frame is gone, and a fresh one
          // on the same figure would come up corrupt, so the reopened tab shows
          // the poster (or an expired note) and regenerate brings it back live.
          const downgraded: View = !v.live
            ? v
            : v.poster
              ? { ...v, live: false, url: v.poster }
              : { ...v, live: false, expired: true };
          const views = { ...s.views, [id]: { ...downgraded, closed: true } };
          const viewOrder = s.viewOrder.filter((x) => x !== id);
          let { activeView, canvasOpen } = s;
          if (activeView === id) {
            activeView = viewOrder[viewOrder.length - 1] ?? null;
            // Fall back to another view if the canvas was open; never re-open one the
            // user had closed.
            canvasOpen = canvasOpen && activeView !== null;
          }
          return { views, viewOrder, activeView, canvasOpen };
        });
      },

      pinDoc: (url, title, slot) =>
        onViz({ type: "viz", url, title, kind: "report", slot, poster: null }),

      setViewMode: (id, mode) =>
        set((s) =>
          s.views[id] ? { views: { ...s.views, [id]: { ...s.views[id], mode } } } : {},
        ),

      setViewSize: (id, width, height) =>
        set((s) =>
          s.views[id] ? { views: { ...s.views, [id]: { ...s.views[id], width, height } } } : {},
        ),

      downgradeView: (id) => {
        // The view's frame is gone (released past the pool cap, or explicitly):
        // fall back to the poster, or mark it expired when there is none. The
        // regenerate button (its `record`) is the way back to a live figure.
        const sid = get().sessionId;
        if (sid) framePool.release(sid, id);
        set((s) => {
          const v = s.views[id];
          if (!v) return {};
          const next: View = v.poster
            ? { ...v, live: false, url: v.poster }
            : { ...v, live: false, expired: true };
          return { views: { ...s.views, [id]: next } };
        });
      },
    };
  });
}

function usageLabels(
  inputTokens: number,
  window: number | null,
): { usageLabel: string; usageTitle: string } {
  if (!inputTokens) return { usageLabel: "", usageTitle: "" };
  const pct = window ? Math.round((inputTokens / window) * 100) : 0;
  const usageLabel = window
    ? `${pct < 1 ? "<1" : pct}% ctx` // some tokens used always reads as at least <1%, never 0%
    : `${formatTokens(inputTokens)} ctx`;
  const usageTitle = `${formatTokens(inputTokens)}${
    window ? " / " + formatTokens(window) : ""
  } context tokens`;
  return { usageLabel, usageTitle };
}
