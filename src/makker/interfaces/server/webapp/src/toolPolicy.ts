// Per-tool card policy and the small helpers that summarize a tool call. Unlisted
// tools default to: open, with the args/code body and the raw text output shown.
// Listed tools get a compact, web-native rendering. Pure functions, unit-tested.

export interface ToolPolicy {
  /** Start collapsed (a quiet, read-only step). */
  collapsed?: boolean;
  /** "none" = summary only; "path" = just the file path; default = a full body. */
  body?: "none" | "path";
  /** false: don't dump the text result (it's noise, or it lives in the canvas). */
  rawOutput?: boolean;
  /** A short result summary appended to the summary line ("42 lines"). */
  note?: (content: string) => string;
  /** The tool reports failure in its reply text (an "ERROR: ..." string the
   *  model reads, rather than a raised error). The card then shows failed, with
   *  the reply surfaced — without this a failed plot call reads as done while
   *  no figure ever appears. */
  failed?: (content: string) => boolean;
}

/** The convention the Julia-facing tools use: a failure is a returned reply
 *  starting with ERROR, so the model can read and react to it. */
const errorReply = (content: string): boolean => content.trimStart().startsWith("ERROR");

export const TOOL_POLICY: Record<string, ToolPolicy> = {
  write_todos: { rawOutput: false }, // the checklist is the body
  read_file: { collapsed: true, body: "path", rawOutput: false, note: (c) => unitNote(c, "line") },
  grep: { collapsed: true, body: "none", note: (c) => unitNote(c, "match", "matches") },
  glob: { collapsed: true, body: "none", note: (c) => unitNote(c, "file") },
  ls: { collapsed: true, body: "none", rawOutput: false, note: listingNote },
  // The figure is pinned in the canvas; a failure must still surface here.
  plot_julia: { collapsed: true, rawOutput: false, failed: errorReply },
  recapture_plot: { collapsed: true, failed: errorReply },
  close_plots: { collapsed: true, failed: errorReply },
  write_report: { collapsed: true, body: "none", rawOutput: false }, // the report is in the canvas
  record_attempt: { rawOutput: false }, // a structured body (rationale + metrics) below
};

export function toolPolicy(name: string | null | undefined): ToolPolicy {
  return (name && TOOL_POLICY[name]) || {};
}

/** Args worth previewing on the collapsed summary line, in priority order. */
const PREVIEW_KEYS = [
  "title", "caption", "code", "command", "file_path", "path", "pattern", "query", "slot",
];

interface TodoItem {
  content?: string;
  activeForm?: string;
  status?: string;
}

/** A one-line preview of what a tool call is doing, for the collapsed summary. */
export function argPreview(args: Record<string, unknown> | null | undefined, name?: string): string {
  if (!args) return "";
  if (name === "write_todos" && Array.isArray(args.todos)) {
    const todos = args.todos as TodoItem[];
    const active = todos.find((t) => t.status === "in_progress");
    if (active) return active.content ?? "";
    return `${todos.length} item${todos.length === 1 ? "" : "s"}`;
  }
  for (const key of PREVIEW_KEYS) {
    const v = args[key];
    if (v) return String(v).split("\n").find((l) => l.trim()) ?? "";
  }
  const first = Object.values(args)[0];
  if (first == null || typeof first === "object") return "";
  return String(first).split("\n")[0];
}

/** ls returns a single-line list repr (['a/', 'b/', …]); count the entries by their
 *  commas rather than quotes, so a filename containing an apostrophe (Python reprs it
 *  with double quotes) doesn't skew the count. */
export function listingNote(content: string): string {
  const inner = String(content || "")
    .trim()
    .replace(/^\[/, "")
    .replace(/\]$/, "")
    .trim();
  const n = inner === "" ? 0 : inner.split(",").length;
  return `${n} ${n === 1 ? "entry" : "entries"}`;
}

/** Count non-blank lines and label them ("3 matches", "1 file"). */
export function unitNote(content: string, singular: string, plural?: string): string {
  const n = String(content || "").split("\n").filter((l) => l.trim()).length;
  return `${n} ${n === 1 ? singular : plural || singular + "s"}`;
}

export function summarizeArgs(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([k, v]) => `${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join("\n");
}

/** Parse a value that may be a dict or a JSON string into an object, else null. */
export function asObject(v: unknown): Record<string, unknown> | null {
  if (v && typeof v === "object" && !Array.isArray(v)) return v as Record<string, unknown>;
  if (typeof v === "string") {
    try {
      const o = JSON.parse(v);
      return o && typeof o === "object" && !Array.isArray(o) ? o : null;
    } catch {
      return null;
    }
  }
  return null;
}
