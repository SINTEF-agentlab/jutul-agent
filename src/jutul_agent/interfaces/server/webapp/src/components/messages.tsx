// Renderers for every kind of thread item. Presentational components driven by the
// store's render model; the streaming ones (assistant, tool) are memoized so only
// the item that changed re-renders during a turn.

import { Fragment, memo, useEffect, useLayoutEffect, useRef, useState } from "react";

import { terminalSegments } from "../ansi";
import { useController, useSel } from "../context";
import { fmtNum } from "../format";
import { KindIcon, KIND_LABEL, ChevronRight } from "../icons";
import { tokenizeJulia } from "../julia";
import { Markdown } from "../markdown";
import { SLASH_COMMANDS } from "../controller";
import {
  argPreview,
  asObject,
  summarizeArgs,
  toolPolicy,
  type ToolPolicy,
} from "../toolPolicy";
import type { ThreadItem } from "../store";

// --- shared building blocks -------------------------------------------------

function CopyablePre({ className, children }: { className: string; children: React.ReactNode }) {
  const ref = useRef<HTMLPreElement>(null);
  const btn = useRef<HTMLButtonElement>(null);
  return (
    <pre ref={ref} className={className} style={{ position: "relative" }}>
      {children}
      <button
        ref={btn}
        type="button"
        className="copy-btn"
        onClick={(e) => {
          e.stopPropagation();
          const code = ref.current?.querySelector("code") ?? ref.current;
          navigator.clipboard.writeText(code?.textContent ?? "").then(() => {
            if (btn.current) {
              btn.current.textContent = "Copied";
              setTimeout(() => btn.current && (btn.current.textContent = "Copy"), 1200);
            }
          });
        }}
      >
        Copy
      </button>
    </pre>
  );
}

function CodeBlock({ text, julia = false }: { text: string; julia?: boolean }) {
  return (
    <CopyablePre className="tool-code">
      <code>
        {julia
          ? tokenizeJulia(text).map((t, i) => <span key={i} className={t.cls}>{t.text}</span>)
          : text}
      </code>
    </CopyablePre>
  );
}

function TerminalOutput({ raw }: { raw: string }) {
  const ref = useRef<HTMLPreElement>(null);
  // Tail to the newest line like a terminal. `stick` stays true until the user
  // scrolls up to read, so a chunk larger than the box still follows the bottom
  // (inferring it from the current position alone breaks: the box opens scrolled to
  // the top, so the first overflow already reads as "scrolled up" and never tails).
  const stick = useRef(true);
  const onScroll = () => {
    const box = ref.current;
    if (box) stick.current = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
  };
  useLayoutEffect(() => {
    const box = ref.current;
    if (box && stick.current) box.scrollTop = box.scrollHeight;
  }, [raw]);
  return (
    <pre ref={ref} className="tool-output" onScroll={onScroll}>
      {terminalSegments(raw).map((seg, i) => (
        <span key={i} style={{ color: seg.color, fontWeight: seg.bold ? 600 : undefined }}>
          {seg.text}
        </span>
      ))}
    </pre>
  );
}

// --- simple items -----------------------------------------------------------

export function UserBubble({ text }: { text: string }) {
  return (
    <div className="msg user">
      <div className="bubble">{text}</div>
    </div>
  );
}

export const AssistantMessage = memo(function AssistantMessage({ text }: { text: string }) {
  return (
    <div className="msg assistant">
      <Markdown text={text} />
    </div>
  );
});

function reasoningSnippet(text: string): string {
  const first = text.split("\n").map((l) => l.trim()).find((l) => l) || "";
  const clean = first.replace(/[*#`>_]/g, "").trim();
  return clean.length > 90 ? clean.slice(0, 90).replace(/\s+\S*$/, "") + "…" : clean;
}

export function ReasoningBlock({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(live);
  const wasLive = useRef(live);
  useEffect(() => {
    if (wasLive.current && !live) setOpen(false); // collapse when the segment ends
    wasLive.current = live;
  }, [live]);
  return (
    <details className="block reasoning" open={open}>
      <summary onClick={(e) => { e.preventDefault(); setOpen((o) => !o); }}>
        <span className="tool-name">Reasoning</span>
        {!open && <span className="tool-preview">{reasoningSnippet(text)}</span>}
      </summary>
      <div className="body">{text}</div>
    </details>
  );
}

// --- tool card --------------------------------------------------------------

const TODO_MARK: Record<string, string> = { completed: "✓", in_progress: "▸", pending: "○" };

interface Todo {
  content?: string;
  activeForm?: string;
  status?: string;
}

function Todos({ todos }: { todos: Todo[] }) {
  return (
    <ul className="todos">
      {todos.map((t, i) => {
        const status = t.status || "pending";
        return (
          <li key={i} className={`todo ${status}`}>
            <span className="todo-mark">{TODO_MARK[status] || "○"}</span>
            <span className="todo-text">{t.content || t.activeForm || ""}</span>
          </li>
        );
      })}
    </ul>
  );
}

function Diff({ oldStr, newStr }: { oldStr: string; newStr: string }) {
  return (
    <pre className="tool-diff">
      {oldStr && oldStr.split("\n").map((l, i) => <div key={`d${i}`} className="del">{"- " + l}</div>)}
      {newStr && newStr.split("\n").map((l, i) => <div key={`a${i}`} className="add">{"+ " + l}</div>)}
    </pre>
  );
}

function KV({ obj, label }: { obj: Record<string, unknown>; label?: string }) {
  return (
    <div className="tool-kv">
      {label && <div className="kv-label">{label}</div>}
      <div className="kv-grid">
        {Object.entries(obj).map(([k, v]) => (
          <Fragment key={k}>
            <span className="kv-key">{k}</span>
            <span className="kv-val">{Array.isArray(v) ? v.map(fmtNum).join(", ") : fmtNum(v)}</span>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function ToolBody({ name, args, policy }: { name: string | null; args: Record<string, unknown> | null; policy: ToolPolicy }) {
  if (policy.body === "none" || !args) return null;
  if (policy.body === "path") {
    const p = args.file_path || args.path;
    return p ? <div className="tool-path">{String(p)}</div> : null;
  }
  if (name === "write_todos" && Array.isArray(args.todos)) return <Todos todos={args.todos as Todo[]} />;
  if (name === "edit_file" && (args.old_string != null || args.new_string != null)) {
    return (
      <>
        {args.file_path ? <div className="tool-path">{String(args.file_path)}</div> : null}
        <Diff oldStr={String(args.old_string || "")} newStr={String(args.new_string || "")} />
      </>
    );
  }
  if (name === "write_file" && args.content != null) {
    return (
      <>
        {args.file_path ? <div className="tool-path">{String(args.file_path)}</div> : null}
        <CodeBlock text={String(args.content)} julia={/\.jl$/.test(String(args.file_path || ""))} />
      </>
    );
  }
  if (name === "record_attempt") {
    const metrics = asObject(args.metrics);
    const params = asObject(args.parameters_changed);
    return (
      <>
        {args.rationale ? <div className="attempt-rationale">{String(args.rationale)}</div> : null}
        {params ? <KV obj={params} label="changed" /> : null}
        {metrics ? <KV obj={metrics} label="metrics" /> : null}
        {args.notes ? <div className="attempt-notes">{String(args.notes)}</div> : null}
      </>
    );
  }
  if (args.code != null) return <CodeBlock text={String(args.code)} julia />;
  if (args.command != null) return <CodeBlock text={String(args.command)} />;
  if (Object.keys(args).length) return <div className="tool-args">{summarizeArgs(args)}</div>;
  return null;
}

type ToolItem = Extract<ThreadItem, { kind: "tool" }>;

export const ToolCard = memo(function ToolCard({ item }: { item: ToolItem }) {
  const policy = toolPolicy(item.name);
  const [open, setOpen] = useState(!policy.collapsed);
  const preview = argPreview(item.args, item.name ?? undefined);
  const summary = item.note ? (preview ? `${preview} · ${item.note}` : item.note) : preview;
  const showOutput = item.output && policy.rawOutput !== false;
  return (
    <details className="block tool" open={open}>
      <summary onClick={(e) => { e.preventDefault(); setOpen((o) => !o); }}>
        <span className="tool-name">{item.label || item.name}</span>
        {summary && <span className="tool-preview">{summary}</span>}
        <span className={`chip-status ${item.status === "running" ? "running" : item.status === "error" ? "error" : ""}`}>
          {item.status === "running" ? <span className="spinner" /> : item.status === "error" ? "error" : "done"}
        </span>
      </summary>
      <div className="body">
        <ToolBody name={item.name} args={item.args ?? null} policy={policy} />
        {showOutput ? <TerminalOutput raw={item.output} /> : null}
      </div>
    </details>
  );
});

// --- canvas references ------------------------------------------------------

export function VizChip({ viewId, title, viewKind }: { viewId: string; title: string; viewKind: "plot" | "report" | "image" }) {
  const openView = useSel((s) => s.openView);
  const active = useSel((s) => s.canvasOpen && s.activeView === viewId);
  return (
    <button className={`viz-chip${active ? " active" : ""}`} onClick={() => openView(viewId)}>
      <span className={`ico ${viewKind}`}>
        <KindIcon kind={viewKind} />
      </span>
      <div className="info">
        <div className="t">{title}</div>
        <div className="s">{KIND_LABEL[viewKind] || "View"}</div>
      </div>
      <span className="go">
        Open <ChevronRight />
      </span>
    </button>
  );
}

export function ArtifactImageCard({ url, title }: { url: string; title: string }) {
  const openImage = useSel((s) => s.openImage);
  return (
    <div className="art-card">
      <div className="head">
        <span className="grow">{title}</span>
        <button className="ghost" onClick={() => openImage(url, title)}>Open</button>
      </div>
      <img src={url} alt={title} onClick={() => openImage(url, title)} />
    </div>
  );
}

export function ArtifactFileCard({ url, caption }: { url: string; caption: string }) {
  return (
    <div className="art-card">
      <div className="head">{caption}</div>
      <a className="file" href={url} target="_blank" rel="noopener noreferrer">
        {url}
      </a>
    </div>
  );
}

export function SysNote({ text, level }: { text: string; level?: "warn" }) {
  return <div className={`sys-note${level ? " " + level : ""}`}>{text}</div>;
}

export function UiNote({ action, payload }: { action: string; payload: Record<string, unknown> }) {
  const json = payload && Object.keys(payload).length ? JSON.stringify(payload) : "";
  return (
    <div className="ui-note">
      <span className="ui-gear">⚙</span>
      <span className="ui-action">{action || "ui"}</span>
      {json && <span className="ui-payload">{json}</span>}
    </div>
  );
}

export function ErrorCard({ message, canRetry }: { message: string; canRetry: boolean }) {
  const controller = useController();
  const lastPrompt = useSel((s) => s.lastPrompt);
  return (
    <div className="error-card">
      <div className="err-msg">{message}</div>
      {canRetry && lastPrompt ? (
        <button className="btn" onClick={() => controller.retry(lastPrompt)}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

export function HelpCard() {
  return (
    <div className="help-card">
      <div className="help-title">Commands</div>
      {SLASH_COMMANDS.map((c) => (
        <div key={c.name} className="help-row">
          <span className="help-name">{c.name + (c.hint ? " " + c.hint : "")}</span>
          <span className="help-desc">{c.desc}</span>
        </div>
      ))}
    </div>
  );
}

export function ContextCard({ markdown }: { markdown: string }) {
  return <Markdown text={markdown} className="context-card markdown" />;
}
