// The wire protocol, typed. This is the TypeScript mirror of the server's
// `interfaces/server/protocol.py` (+ the side-channel events app.py emits). Both
// ends now code against one explicit contract, so a renamed or reshaped field is
// a compile error here instead of a silent runtime bug.

/** A pending approval's individual action (one tool call awaiting a decision). */
export interface InterruptAction {
  name: string;
  label?: string;
  args?: Record<string, unknown>;
  description?: string | null;
  /** Markdown detail for the action: the command, a content preview, or the
   *  on-disk diff. The same card body the TUI shows. */
  body?: string | null;
}

/** Messages the server streams to the client over the WebSocket. */
export type ServerMessage =
  | { type: "text"; text: string }
  | { type: "reasoning"; text: string }
  | {
      type: "tool";
      event: "requested" | "started" | "delta" | "finished" | "error";
      name: string | null;
      label?: string;
      tool_call_id: string | null;
      args?: Record<string, unknown> | null;
      content?: string | null;
      /** A delta carries the full terminal-rendered output so far: replace, not append. */
      replace?: boolean;
    }
  | {
      type: "interrupt";
      interrupt_id: string;
      actions: InterruptAction[];
      allowed_decisions: string[];
      allowlist: string[];
    }
  | {
      type: "usage";
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
      model_calls: number;
    }
  | { type: "turn_end"; text: string; cancelled?: boolean }
  | {
      type: "artifact";
      url: string;
      mime?: string | null;
      caption?: string | null;
      slot?: string | null;
      format?: string | null;
    }
  | {
      type: "viz";
      url: string;
      title?: string | null;
      kind: string;
      poster?: string | null;
      slot?: string | null;
      /** Served by the session's live figure server (widgets work; the view is
       *  stateful and supports exactly one frame). */
      live?: boolean;
      /** The plot's trace record when its source code was recorded — what a
       *  `replot` request sends back (regenerate a dead view, or a popout). */
      record?: string | null;
      /** The figure's own pixel size, so the canvas can present it at its
       *  designed shape (the scale stage) instead of crushing it. */
      width?: number | null;
      height?: number | null;
    }
  | { type: "popout_ready"; record: string; url: string | null; error?: string | null }
  | { type: "notice"; text: string }
  | { type: "ui"; action: string; payload: Record<string, unknown> }
  | { type: "credential_required"; provider: string; label: string; env_var: string }
  | { type: "error"; message: string };

export type ServerMessageType = ServerMessage["type"];

/** Messages the client sends to the server. */
export type ClientMessage =
  | { type: "prompt"; text: string }
  | { type: "decision"; decision: string; message?: string }
  | { type: "cancel" }
  | { type: "command"; command: "set_model" | "set_approval" | "add_dir" | "compact"; arg: string }
  /** Replay a recorded plot: revive its view in place, or serve an independent
   *  figure for a popout window (answered by `popout_ready`). */
  | { type: "replot"; record: string; target?: "revive" | "popout" }
  /** A popout window this client opened has closed; the server releases its route. */
  | { type: "popout_closed"; url: string }
  | { type: "ui_event"; payload: unknown };

/**
 * One recorded item from `GET /sessions/{id}/messages`, replayed to reconstruct a
 * resumed conversation. A superset of the live stream: it also carries the `user`
 * and `assistant` text the live path delivers as bubbles and `text` deltas.
 */
export type ReplayMessage =
  | { type: "user"; text: string }
  | { type: "assistant"; text: string }
  | { type: "reasoning"; text: string }
  | Extract<ServerMessage, { type: "tool" | "viz" | "artifact" }>;

/** Parse a raw WebSocket text frame, or return null if it is not a typed message. */
export function parseServerMessage(data: string): ServerMessage | null {
  try {
    const msg = JSON.parse(data);
    return msg && typeof msg.type === "string" ? (msg as ServerMessage) : null;
  } catch {
    return null;
  }
}

/** Side outputs (a plot/report, a usage tick, a host-app ui action) can arrive
 *  mid-turn while the agent keeps working, so they must NOT clear the "thinking"
 *  indicator — only the agent's own content (or the turn ending) does. A `notice`
 *  is deliberately excluded: it signals a command finished (e.g. /compact), so its
 *  handler legitimately ends the working/busy state rather than passing through. */
export const SIDE_OUTPUT_TYPES: ReadonlySet<ServerMessageType> = new Set([
  "usage",
  "viz",
  "artifact",
  "ui",
]);

/** A `ui` action the client consumes internally (a history-refresh signal) instead
 *  of rendering it as a note in the thread. Named once here so the store (which
 *  suppresses the thread item) and the controller (which runs the refresh) reference
 *  the same action and can't drift apart on a rename. */
export const HISTORY_CHANGED = "history_changed";
