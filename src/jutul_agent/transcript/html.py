"""Render a trace log as a self-contained HTML transcript."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from jutul_agent.tool_labels import tool_label
from jutul_agent.trace import Event, schema
from jutul_agent.transcript.events import ArtifactPayload
from jutul_agent.transcript.highlight import (
    PLAIN_OUTPUT_TOOLS,
    SIMULATOR_TOOLS,
    highlight_code,
    highlight_code_blocks_in_html,
    infer_tool_code_language,
    is_julia_source,
    pygments_css,
)
from jutul_agent.transcript.markdown_html import (
    looks_like_markdown,
    render_markdown_html,
    strip_line_number_prefixes,
    strip_yaml_frontmatter,
)

_STYLES = """
:root {
  color-scheme: light dark;
  --bg: #f8f9fb;
  --surface: #ffffff;
  --border: #d8dee9;
  --text: #1f2937;
  --muted: #6b7280;
  --user: #2563eb;
  --assistant: #059669;
  --reasoning: #7c3aed;
  --tool: #d97706;
  --approval: #dc2626;
  --artifact: #0891b2;
  --session: #475569;
  --code-bg: #f6f8fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1419;
    --surface: #1a2332;
    --border: #334155;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --code-bg: #0d1117;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}
.wrap { max-width: 960px; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
header {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.9rem 1.25rem;
  margin-bottom: 0.75rem;
}
header h1 { margin: 0 0 0.3rem; font-size: 1.3rem; }
.title-sub { margin: 0 0 0.35rem; font-size: 1rem; font-weight: 500; }
.meta { color: var(--muted); font-size: 0.85rem; margin: 0.15rem 0; }
.count { color: var(--muted); font-variant-numeric: tabular-nums; }
.badges { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.75rem; }
.badge {
  font-size: 0.75rem;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  background: var(--bg);
}
.controls {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.5rem 1rem;
  margin-bottom: 0.75rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem 1.25rem;
  align-items: center;
}
.controls fieldset {
  border: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  align-items: center;
}
.controls legend { font-size: 0.85rem; color: var(--muted); margin-right: 0.25rem; }
.controls label { font-size: 0.85rem; cursor: pointer; }
.controls button {
  font-size: 0.85rem;
  padding: 0.35rem 0.75rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text);
  cursor: pointer;
}
.timeline { display: flex; flex-direction: column; gap: 0.75rem; }
.event {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.85rem 1rem;
}
.event h3 {
  margin: 0.15rem 0 0.5rem;
  font-size: 1rem;
  font-weight: 600;
}
.event h4 {
  margin: 0.75rem 0 0.35rem;
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.event-header {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: baseline;
  margin-bottom: 0.5rem;
  font-size: 0.85rem;
}
.event-kind { font-weight: 600; }
.event-kind.user { color: var(--user); }
.event-kind.assistant { color: var(--assistant); }
.event-kind.reasoning { color: var(--reasoning); }
.event-kind.tool { color: var(--tool); }
.event-kind.approval { color: var(--approval); }
.event-kind.artifact { color: var(--artifact); }
.event-kind.session { color: var(--session); }
.event-kind.unknown { color: var(--muted); }
.event time { color: var(--muted); font-variant-numeric: tabular-nums; }
.md-content { line-height: 1.6; word-break: break-word; }
.md-content > :first-child { margin-top: 0; }
.md-content > :last-child { margin-bottom: 0; }
.md-content h1, .md-content h2, .md-content h3, .md-content h4 {
  margin: 1rem 0 0.4rem;
  line-height: 1.25;
}
.md-content h1 { font-size: 1.35rem; }
.md-content h2 { font-size: 1.15rem; margin-top: 1.1rem; }
.md-content h3 { font-size: 1.02rem; margin-top: 0.95rem; }
.md-content p { margin: 0.55rem 0; }
.md-content ul, .md-content ol { margin: 0.55rem 0 0.75rem; padding-left: 1.5rem; }
.md-content li { margin: 0.25rem 0; }
.md-content li > p { margin: 0.15rem 0; }
.md-content blockquote {
  margin: 0.5rem 0;
  padding: 0.25rem 0.75rem;
  border-left: 3px solid var(--border);
  color: var(--muted);
}
.md-content table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.5rem 0;
  font-size: 0.9rem;
}
.md-content th, .md-content td {
  border: 1px solid var(--border);
  padding: 0.35rem 0.5rem;
  text-align: left;
}
.md-content th { background: var(--bg); }
.md-content img { max-width: 100%; height: auto; }
.md-content a { color: var(--user); }
.prose-plain { white-space: pre-wrap; word-break: break-word; }
.tool-output {
  margin: 0;
}
.inline-value {
  display: inline-block;
  padding: 0.1rem 0.35rem;
  border-radius: 4px;
  background: var(--bg);
  border: 1px solid var(--border);
  font-family: ui-monospace, "Cascadia Code", "Source Code Pro", monospace;
  font-size: 0.85rem;
}
.todo-list { list-style: none; margin: 0; padding: 0; }
.todo-item {
  display: flex;
  gap: 0.5rem;
  align-items: baseline;
  padding: 0.25rem 0;
  border-bottom: 1px solid var(--border);
}
.todo-item:last-child { border-bottom: none; }
.todo-status {
  flex: 0 0 auto;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  padding: 0.1rem 0.4rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--muted);
}
.todo-status.completed { color: var(--assistant); }
.todo-status.in_progress { color: var(--tool); }
.todo-text { flex: 1; }
.share-note {
  margin-top: 1rem;
  padding: 0.75rem 1rem;
  border-radius: 8px;
  border: 1px dashed var(--border);
  color: var(--muted);
  font-size: 0.85rem;
}
pre, code {
  font-family: ui-monospace, "Cascadia Code", "Source Code Pro", monospace;
  font-size: 0.85rem;
}
pre {
  margin: 0;
  padding: 0.65rem 0.75rem;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
.md-content pre {
  margin: 0.5rem 0;
}
.md-content pre code, .md-content code {
  font-family: ui-monospace, "Cascadia Code", "Source Code Pro", monospace;
  font-size: 0.85rem;
}
.md-content :not(pre) > code {
  padding: 0.1rem 0.3rem;
  border-radius: 4px;
  background: var(--bg);
  border: 1px solid var(--border);
}
dl { margin: 0; }
dt { font-weight: 600; font-size: 0.85rem; margin-top: 0.35rem; }
dd { margin: 0.15rem 0 0 0; }
details { margin-top: 0.35rem; }
details > summary { cursor: pointer; color: var(--muted); font-size: 0.9rem; }
/* Tool calls are secondary to the conversation: a compact, collapsed row with a
   one-line peek at the input; expand for the full code and result. */
.tool-event { padding: 0.3rem 0.75rem; }
.tool-event > details { margin-top: 0; }
.tool-summary {
  display: flex;
  gap: 0.6rem;
  align-items: baseline;
  list-style: none;
  font-size: 0.85rem;
}
.tool-summary::-webkit-details-marker { display: none; }
.tool-summary::before { content: "\\25B8"; color: var(--muted); flex: 0 0 auto; }
details[open] > .tool-summary::before { content: "\\25BE"; }
.tool-peek {
  flex: 1;
  min-width: 0;
  color: var(--muted);
  font-family: ui-monospace, "Cascadia Code", "Source Code Pro", monospace;
  font-size: 0.8rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.tool-summary time { flex: 0 0 auto; }
.tool-body { margin-top: 0.5rem; }
.event.approval { border-left: 3px solid var(--approval); }
.approval-actions { margin: 0; padding-left: 1.2rem; }
.approval-actions li { margin: 0.1rem 0; }
.approval-desc { color: var(--muted); font-size: 0.9rem; margin-top: 0.15rem; }
.approval-outcome {
  margin-top: 0.6rem;
  padding-top: 0.6rem;
  border-top: 1px solid var(--border);
}
.approval-outcome p { margin: 0; }
figure { margin: 0.5rem 0 0; }
figure img {
  max-width: 100%;
  height: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg);
}
figcaption { font-size: 0.85rem; color: var(--muted); margin-top: 0.35rem; }
.artifact-meta { margin: 0.25rem 0 0.5rem; }
.artifact-meta .badge { margin-right: 0.35rem; }
.inline-artifact { margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px dashed var(--border); }
.inline-artifact figure img { max-height: 320px; object-fit: contain; }
.artifact-source { margin-top: 0.5rem; }
.session-marker {
  text-align: center;
  color: var(--muted);
  font-size: 0.9rem;
  padding: 0.5rem;
}
"""

_SCRIPT = """
(function () {
  const events = document.querySelectorAll(".event[data-filter]");
  const checkboxes = document.querySelectorAll(".kind-filter");
  function applyFilters() {
    const hidden = new Set(
      [...checkboxes].filter((cb) => !cb.checked).map((cb) => cb.dataset.filter)
    );
    events.forEach((el) => {
      el.style.display = hidden.has(el.dataset.filter) ? "none" : "";
    });
  }
  checkboxes.forEach((cb) => cb.addEventListener("change", applyFilters));
  document.getElementById("collapse-all")?.addEventListener("click", () => {
    document.querySelectorAll("details").forEach((d) => { d.open = false; });
  });
  document.getElementById("expand-all")?.addEventListener("click", () => {
    document.querySelectorAll("details").forEach((d) => { d.open = true; });
  });
})();
"""


def _script_hash() -> str:
    """CSP source hash for the one trusted inline script (its exact bytes)."""
    digest = hashlib.sha256(_SCRIPT.encode("utf-8")).digest()
    return "sha256-" + base64.b64encode(digest).decode("ascii")


# The transcript is opened from disk and renders untrusted content (LLM messages,
# tool output, files the agent read). This CSP allows only the page's own inline
# script (matched by hash, so an injected ``<script>`` is refused) and inline
# styles; images may be inlined ``data:`` or local files (artifacts shipped beside
# the transcript) but never remote, and there is no network egress, so a tracking
# pixel or fetch smuggled into the transcript cannot fire.
_CSP = (
    "default-src 'none'; "
    f"script-src '{_script_hash()}'; "
    "img-src 'self' data: file:; "
    "style-src 'unsafe-inline'; font-src data:; base-uri 'none'; form-action 'none'"
)

_FILTER_GROUPS: dict[str, str] = {
    schema.MESSAGE_USER: "user",
    schema.MESSAGE_ASSISTANT: "assistant",
    schema.MESSAGE_REASONING: "reasoning",
    schema.TOOL_CALL: "tools",
    "tool_result": "tools",
    schema.HITL_REQUEST: "approval",
    schema.HITL_RESPONSE: "approval",
    schema.ARTIFACT: "artifact",
}

_FILTER_LABELS: dict[str, str] = {
    "user": "User",
    "assistant": "Assistant",
    "reasoning": "Reasoning",
    "tools": "Tools",
    "approval": "Approval",
    schema.ARTIFACT: "Artifact",
}

# Lifecycle markers: rendered as dividers/header metadata, not filterable cards.
_SESSION_MARKER_KINDS = frozenset(
    {
        schema.SESSION_START,
        schema.SESSION_END,
        schema.SESSION_RESUME,
        schema.SESSION_TITLE,
        schema.CONTEXT_COMPACTION,
    }
)

# Internal telemetry the trace keeps for cost/efficiency analysis and eval grading,
# but which is not part of the conversation: dropped entirely from the human
# transcript (no card, no filter chip).
_INTERNAL_KINDS = frozenset({schema.MODEL_USAGE, schema.EVAL_TARGET})

_KIND_LABELS: dict[str, str] = {
    **{kind: "Session" for kind in _SESSION_MARKER_KINDS},
    **{kind: _FILTER_LABELS[group] for kind, group in _FILTER_GROUPS.items()},
}


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=False)


def _esc_attr(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _nl2br(text: str) -> str:
    return _esc(text).replace("\n", "<br>\n")


def _render_markdown(text: str) -> str:
    rendered = highlight_code_blocks_in_html(render_markdown_html(text.strip()))
    return f'<div class="md-content">{rendered}</div>'


def _render_prose(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if looks_like_markdown(stripped):
        return _render_markdown(stripped)
    return f'<div class="prose-plain">{_nl2br(stripped)}</div>'


def _render_tool_content(content: str, *, tool_name: str | None = None) -> str:
    stripped = content.strip()
    if not stripped:
        return ""

    body = stripped

    if tool_name == "read_file":
        body = strip_line_number_prefixes(body)
        body = strip_yaml_frontmatter(body)
        if looks_like_markdown(body):
            return _render_markdown(body)
        language = infer_tool_code_language(body, tool_name)
        return highlight_code(body, language)

    if tool_name in PLAIN_OUTPUT_TOOLS:
        return highlight_code(body, "text")

    if tool_name in SIMULATOR_TOOLS:
        return highlight_code(body, "julia")

    if looks_like_markdown(body) and not is_julia_source(body):
        return _render_markdown(body)

    language = infer_tool_code_language(body, tool_name)
    return highlight_code(body, language)


def _fmt_todos(value: Any) -> str:
    if not isinstance(value, list):
        return f"<pre>{_esc(repr(value))}</pre>"
    items: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            items.append(f"<li><pre>{_esc(repr(item))}</pre></li>")
            continue
        status = str(item.get("status") or "pending").replace(" ", "_")
        content = str(item.get("content") or "")
        items.append(
            f'<li class="todo-item">'
            f'<span class="todo-status {_esc_attr(status)}">{_esc(status.replace("_", " "))}</span>'
            f'<span class="todo-text">{_esc(content)}</span>'
            f"</li>"
        )
    return f'<ul class="todo-list">{"".join(items)}</ul>'


def _fmt_arg_value(key: str, value: Any) -> str:
    if key == "todos":
        return _fmt_todos(value)
    if key == "code" and isinstance(value, str):
        return highlight_code(value, "julia")
    if isinstance(value, str):
        if "\n" in value or len(value) > 120:
            language = infer_tool_code_language(value)
            return highlight_code(value, language)
        return f'<span class="inline-value">{_esc(value)}</span>'
    if isinstance(value, (dict, list)):
        try:
            pretty = json.dumps(value, indent=2, default=str)
        except TypeError:
            pretty = repr(value)
        return highlight_code(pretty, "json")
    return f'<span class="inline-value">{_esc(repr(value))}</span>'


def _artifact_meta_html(artifact: ArtifactPayload) -> str:
    badges: list[str] = []
    if artifact.slot:
        badges.append(f'<span class="badge">slot: {_esc(artifact.slot)}</span>')
    if artifact.format:
        badges.append(f'<span class="badge">format: {_esc(artifact.format)}</span>')
    if artifact.size_px is not None:
        badges.append(
            f'<span class="badge">{_esc(artifact.size_px[0])}x{_esc(artifact.size_px[1])}</span>'
        )
    if not badges:
        return ""
    return f'<p class="artifact-meta">{"".join(badges)}</p>'


def _artifact_source_html(artifact: ArtifactPayload) -> str:
    if not artifact.source_code:
        return ""
    return (
        '<details class="artifact-source">'
        "<summary>Source code</summary>"
        f"{highlight_code(artifact.source_code.rstrip(), 'julia')}"
        "</details>"
    )


def _fmt_args(args: Any) -> str:
    if not args:
        return "<p><em>(no args)</em></p>"
    if isinstance(args, dict):
        items = "".join(
            f"<dt>{_esc(k)}</dt><dd>{_fmt_arg_value(k, v)}</dd>" for k, v in args.items()
        )
        return f"<dl>{items}</dl>"
    return f"<pre>{_esc(repr(args))}</pre>"


def _fmt_hitl_request(payload: dict[str, Any]) -> str:
    """Human-readable approval request: each action's tool and a readable view
    of its arguments (the command for a shell call, the path for a file edit),
    plus any description. The internal interrupt id is left out of the prose."""
    value = payload.get("value")
    actions = value.get("action_requests") if isinstance(value, dict) else None
    if not isinstance(actions, list) or not actions:
        return _render_prose("An action is awaiting approval.")
    items: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        line = f"<code>{_esc(str(action.get('name', '?')))}</code>"
        summary = _hitl_action_summary(action.get("args"))
        if summary:
            line += f" {summary}"
        desc = str(action.get("description") or "").strip()
        if desc:
            line += f'<div class="approval-desc">{_esc(desc)}</div>'
        items.append(f"<li>{line}</li>")
    return f'<ul class="approval-actions">{"".join(items)}</ul>'


def _hitl_action_summary(args: Any) -> str:
    """Inline summary of an action's args: the command or path if present, else
    a compact key=value list."""
    if not isinstance(args, dict) or not args:
        return ""
    for key in ("command", "path", "file_path"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return f"<code>{_esc(value)}</code>"
    return _esc(", ".join(f"{k}={v}" for k, v in args.items()))


_HITL_DECISION_LABELS = {"approve": "Approved", "reject": "Rejected", "respond": "Responded"}


def _fmt_hitl_response(payload: dict[str, Any]) -> str:
    """Human-readable approval outcome: approved / rejected / responded, with
    any message the reviewer gave."""
    inner = payload.get("payload")
    decisions = inner.get("decisions") if isinstance(inner, dict) else None
    if not isinstance(decisions, list) or not decisions:
        return _render_prose("The request was resolved.")
    lines: list[str] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        dtype = str(decision.get("type") or "?")
        label = _HITL_DECISION_LABELS.get(dtype, dtype.capitalize())
        out = f"<strong>{_esc(label)}</strong>"
        message = str(decision.get("message") or "").strip()
        if message:
            out += f": {_esc(message)}"
        lines.append(f"<p>{out}</p>")
    return "".join(lines)


def _first_line(text: str, limit: int = 90) -> str:
    """The first non-empty line of ``text``, truncated for a one-line summary."""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return f"{line[: limit - 1]}…" if len(line) > limit else line


def _short_time(ts: str) -> str:
    """Just the clock time from an ISO timestamp; the date sits in the header."""
    if "T" not in ts:
        return ts
    return ts.split("T", 1)[1].split("+", 1)[0].split(".", 1)[0]


def _tool_summary(args: Any) -> str:
    """A one-line peek at what a tool call did, for the collapsed summary.

    Shows the argument that carries the intent (the command, the code, the path
    it touched), so a reader can scan the run without expanding every call.
    """
    if not isinstance(args, dict) or not args:
        return ""
    for key in ("command", "code", "file_path", "path", "pattern", "query"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _first_line(value, 80)
    todos = args.get("todos")
    if isinstance(todos, list):
        return f"{len(todos)} item(s)"
    for key, value in args.items():
        if isinstance(value, str | int | float | bool):
            return _first_line(f"{key}={value}", 80)
    return ""


def _tool_event(*, label: str, peek: str, detail: str, ts: str, kind: str) -> str:
    """A collapsed tool row: the tool, a one-line peek, and the timestamp, with
    the full args/result tucked inside the disclosure."""
    peek_html = f'<span class="tool-peek">{_esc(peek)}</span>' if peek else ""
    return (
        f'<article class="event tool-event" {_event_attrs(kind)}>'
        f'<details><summary class="tool-summary">'
        f'<span class="event-kind tool">{_esc(label)}</span>'
        f"{peek_html}"
        f'<time datetime="{_esc_attr(ts)}">{_esc(_short_time(ts))}</time>'
        f"</summary>"
        f'<div class="tool-body">{detail}</div>'
        f"</details></article>"
    )


def _event_header(kind: str, label: str, ts: str, css_kind: str) -> str:
    return (
        f'<div class="event-header">'
        f'<span class="event-kind {css_kind}">{_esc(label)}</span>'
        f'<time datetime="{_esc_attr(ts)}">{_esc(_short_time(ts))}</time>'
        f"</div>"
    )


def _tool_result_block(content: str, *, tool_name: str | None = None) -> str:
    lines = content.splitlines()
    rendered = _render_tool_content(content, tool_name=tool_name)
    if len(lines) > 20:
        return f"<details><summary>Result ({len(lines)} lines)</summary>{rendered}</details>"
    return rendered


def _event_attrs(kind: str) -> str:
    group = _FILTER_GROUPS.get(kind, kind)
    return f'data-kind="{_esc_attr(kind)}" data-filter="{_esc_attr(group)}"'


def _index_tool_results(events: list[Event]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev.kind != schema.TOOL_RESULT:
            continue
        tool_call_id = ev.payload.get("tool_call_id")
        if tool_call_id is not None:
            indexed[str(tool_call_id)] = ev.payload
    return indexed


def _index_artifacts_by_tool_call(events: list[Event]) -> dict[str, ArtifactPayload]:
    indexed: dict[str, ArtifactPayload] = {}
    for ev in events:
        if ev.kind != schema.ARTIFACT:
            continue
        artifact = ArtifactPayload.from_payload(ev.payload)
        if artifact.tool_call_id is not None:
            indexed[artifact.tool_call_id] = artifact
    return indexed


def _index_hitl_responses(events: list[Event]) -> dict[str, dict[str, Any]]:
    """interrupt_id -> response payload, so a request can show its outcome inline."""
    indexed: dict[str, dict[str, Any]] = {}
    for ev in events:
        if ev.kind == schema.HITL_RESPONSE:
            interrupt_id = ev.payload.get("interrupt_id")
            if interrupt_id is not None:
                indexed[str(interrupt_id)] = ev.payload
    return indexed


# Largest artifact to base64-inline into a transcript; above this, keep the path.
_MAX_INLINE_BYTES = 10 * 1024 * 1024


def _resolve_artifact_file(path: str, artifact_dirs: Sequence[Path]) -> Path | None:
    """Find the on-disk file for a recorded artifact path, contained to the run's dirs.

    The transcript inlines file *bytes*, so a recorded path must resolve to a file
    inside one of the run dirs; an absolute or ``..`` path that escapes every dir is
    rejected (mirrors the server's ``_resolve_artifact`` guard) so a stray/crafted
    artifact path can't pull arbitrary files into a shared transcript.
    """
    raw = Path(path)
    for root in artifact_dirs:
        base = root.resolve()
        candidate = (base / raw).resolve()
        if candidate.is_file() and candidate.is_relative_to(base):
            return candidate
        if path.startswith("artifacts/"):  # the transcript may sit inside artifacts/
            tail = (base / raw.name).resolve()
            if tail.is_file() and tail.is_relative_to(base):
                return tail
    return None


def _image_src(path: str, artifact_dirs: Sequence[Path]) -> str:
    """A self-contained ``<img>`` src for an artifact path: a base64 data URI when
    the file resolves under the run's artifact dirs, else the path unchanged.

    Embedding makes the transcript show its plots regardless of where it's opened
    (a sidecar inside ``artifacts/``, a standalone download, or a bundle); the
    plain-path fallback still works when the file sits beside the transcript.
    """
    if not path or path.startswith(("data:", "http:", "https:")):
        return path
    resolved = _resolve_artifact_file(path, artifact_dirs)
    if resolved is None:
        return path
    try:
        # Don't inline a very large file: it would bloat the transcript (and the
        # server response) by ~1.33x its size, held in memory. Leave the path instead.
        if resolved.stat().st_size > _MAX_INLINE_BYTES:
            return path
        data = base64.standard_b64encode(resolved.read_bytes()).decode("ascii")
    except OSError:
        return path
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{data}"


def _inline_artifact_preview(
    artifact: ArtifactPayload | None, artifact_dirs: Sequence[Path] = ()
) -> str:
    if artifact is None or not artifact.path or not artifact.is_image:
        return ""
    meta = _artifact_meta_html(artifact)
    src = _image_src(artifact.path, artifact_dirs)
    return (
        f'<div class="inline-artifact">'
        f"{meta}"
        f"<figure>"
        f'<a href="{_esc_attr(src)}">'
        f'<img src="{_esc_attr(src)}" alt="{_esc_attr(artifact.caption)}">'
        f"</a>"
        f"<figcaption>{_esc(artifact.caption)}</figcaption>"
        f"</figure>"
        f"</div>"
    )


def _render_body(
    events: Iterable[Event], artifact_dirs: Sequence[Path] = ()
) -> tuple[str, dict[str, int]]:
    event_list = list(events)
    tool_results = _index_tool_results(event_list)
    artifacts_by_call = _index_artifacts_by_tool_call(event_list)
    hitl_responses = _index_hitl_responses(event_list)
    consumed_results: set[str] = set()
    consumed_responses: set[str] = set()
    parts: list[str] = []
    kind_counts: dict[str, int] = {}

    session_id: str | None = None
    session_title: str | None = None
    simulator: str | None = None
    started: str | None = None

    for ev in event_list:
        payload = ev.payload
        ts = ev.timestamp
        kind = ev.kind
        if kind in _INTERNAL_KINDS:
            continue
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

        if kind == schema.SESSION_START:
            session_id = payload.get("session_id") or session_id
            simulator = payload.get("simulator") or "(none)"
            started = ts
            continue

        if kind == schema.SESSION_END:
            parts.append(
                f'<div class="session-marker">Session ended · <time datetime="{_esc_attr(ts)}">'
                f"{_esc(ts)}</time></div>"
            )
            continue

        if kind == schema.SESSION_RESUME:
            parts.append(
                f'<div class="session-marker">Session resumed · '
                f'<time datetime="{_esc_attr(ts)}">{_esc(ts)}</time></div>'
            )
            continue

        if kind == schema.SESSION_TITLE:
            session_title = str(payload.get("title") or "").strip() or session_title
            continue

        if kind == schema.CONTEXT_COMPACTION:
            trigger = "manual" if payload.get("manual") else "automatic"
            summarized = payload.get("summarized")
            kept = payload.get("kept")
            parts.append(
                f'<div class="session-marker">Context compacted ({trigger}): '
                f"{_esc(summarized if summarized is not None else '?')} messages summarized, "
                f"{_esc(kept if kept is not None else '?')} kept · "
                f'<time datetime="{_esc_attr(ts)}">{_esc(ts)}</time></div>'
            )
            continue

        if kind == schema.MESSAGE_USER:
            parts.append(
                f'<article class="event" {_event_attrs(kind)}>'
                f"{_event_header(kind, 'User', ts, 'user')}"
                f"{_render_prose(str(payload.get('content', '')))}"
                f"</article>"
            )
            continue

        if kind == schema.MESSAGE_REASONING:
            content = str(payload.get("content", ""))
            if not content.strip():
                continue
            parts.append(
                f'<article class="event" {_event_attrs(kind)}>'
                f"{_event_header(kind, 'Reasoning', ts, 'reasoning')}"
                f"<details>"
                f"<summary>{_esc(_first_line(content))}</summary>"
                f"{_render_prose(content)}"
                f"</details>"
                f"</article>"
            )
            continue

        if kind == schema.MESSAGE_ASSISTANT:
            content = str(payload.get("content", ""))
            if not content.strip():
                continue
            parts.append(
                f'<article class="event" {_event_attrs(kind)}>'
                f"{_event_header(kind, 'Assistant', ts, 'assistant')}"
                f"{_render_prose(content)}"
                f"</article>"
            )
            continue

        if kind == schema.TOOL_CALL:
            name = payload.get("name", "?")
            tool_call_id = payload.get("id")
            result_payload = None
            if tool_call_id is not None:
                result_payload = tool_results.get(str(tool_call_id))
                if result_payload is not None:
                    consumed_results.add(str(tool_call_id))

            detail = _fmt_args(payload.get("args"))
            if result_payload is not None:
                content = str(result_payload.get("content", ""))
                detail += f"<h4>Result</h4>{_tool_result_block(content, tool_name=name)}"
            if tool_call_id is not None and name == "plot_julia":
                preview = _inline_artifact_preview(
                    artifacts_by_call.get(str(tool_call_id)), artifact_dirs
                )
                if preview:
                    detail += preview
            parts.append(
                _tool_event(
                    label=tool_label(str(name)),
                    peek=_tool_summary(payload.get("args")),
                    detail=detail,
                    ts=ts,
                    kind=kind,
                )
            )
            continue

        if kind == schema.TOOL_RESULT:
            tool_call_id = payload.get("tool_call_id")
            if tool_call_id is not None and str(tool_call_id) in consumed_results:
                continue
            name = payload.get("name", "?")
            content = str(payload.get("content", ""))
            parts.append(
                _tool_event(
                    label=f"{tool_label(str(name))} result",
                    peek=_first_line(content, 80),
                    detail=_tool_result_block(content, tool_name=name),
                    ts=ts,
                    kind=kind,
                )
            )
            continue

        if kind == schema.HITL_REQUEST:
            interrupt_id = str(payload.get("interrupt_id") or "")
            outcome = ""
            response = hitl_responses.get(interrupt_id)
            if response is not None:
                consumed_responses.add(interrupt_id)
                outcome = f'<div class="approval-outcome">{_fmt_hitl_response(response)}</div>'
            parts.append(
                f'<article class="event approval" {_event_attrs(kind)}>'
                f"{_event_header(kind, 'Approval', ts, 'approval')}"
                f"{_fmt_hitl_request(payload)}{outcome}"
                f"</article>"
            )
            continue

        if kind == schema.HITL_RESPONSE:
            # Already shown inline under its request; only a response with no
            # matching request (shouldn't happen) falls through to its own card.
            if str(payload.get("interrupt_id") or "") in consumed_responses:
                continue
            parts.append(
                f'<article class="event approval" {_event_attrs(kind)}>'
                f"{_event_header(kind, 'Approval response', ts, 'approval')}"
                f"{_fmt_hitl_response(payload)}"
                f"</article>"
            )
            continue

        if kind == schema.ARTIFACT:
            artifact = ArtifactPayload.from_payload(payload)
            meta_html = _artifact_meta_html(artifact)
            source_html = _artifact_source_html(artifact)
            if artifact.is_image:
                parts.append(
                    f'<article class="event" {_event_attrs(kind)}>'
                    f"{_event_header(kind, 'Artifact', ts, 'artifact')}"
                    f"{meta_html}"
                    f"<figure>"
                    f'<img src="{_esc_attr(_image_src(artifact.path, artifact_dirs))}"'
                    f' alt="{_esc_attr(artifact.caption)}">'
                    f"<figcaption>{_esc(artifact.caption)}</figcaption>"
                    f"</figure>"
                    f"{source_html}"
                    f"</article>"
                )
            else:
                parts.append(
                    f'<article class="event" {_event_attrs(kind)}>'
                    f"{_event_header(kind, 'Artifact', ts, 'artifact')}"
                    f'<p><a href="{_esc_attr(artifact.path)}">'
                    f"{_esc(artifact.caption or artifact.path)}</a></p>"
                    f"</article>"
                )
            continue

        parts.append(
            f'<article class="event" {_event_attrs(kind)}>'
            f"{_event_header(kind, f'Event · {kind}', ts, 'unknown')}"
            f"<details>"
            f"<summary>Raw payload</summary>"
            f"<pre>{_esc(repr(payload))}</pre>"
            f"</details>"
            f"</article>"
        )

    # One compact meta line; the date rides on `started` and the session-end
    # marker at the foot carries the end time, so neither needs its own row.
    header_meta = ""
    if session_title:
        header_meta += f'<p class="title-sub">{_esc(session_title)}</p>'
    meta_bits: list[str] = []
    if session_id:
        meta_bits.append(f"Session <code>{_esc(session_id)}</code>")
    if simulator:
        meta_bits.append(_esc(simulator))
    if started:
        meta_bits.append(f'<time datetime="{_esc_attr(started)}">{_esc(started)}</time>')
    if meta_bits:
        header_meta += f'<p class="meta">{" · ".join(meta_bits)}</p>'

    # Filter chips carry their own count, so there's no separate badges row
    # repeating the same kinds. The count is per rendered card: a tool result
    # folded into its call, or an approval response folded into its request,
    # shows no card of its own, so it must not inflate the tally.
    rendered_counts = dict(kind_counts)
    rendered_counts["tool_result"] = rendered_counts.get("tool_result", 0) - len(consumed_results)
    rendered_counts[schema.HITL_RESPONSE] = rendered_counts.get(schema.HITL_RESPONSE, 0) - len(
        consumed_responses
    )
    group_counts: dict[str, int] = {}
    for kind, count in rendered_counts.items():
        if kind in _SESSION_MARKER_KINDS:
            continue
        label = _FILTER_LABELS.get(_FILTER_GROUPS.get(kind, kind), _FILTER_GROUPS.get(kind, kind))
        group_counts[label] = group_counts.get(label, 0) + count

    filter_groups = sorted(
        {_FILTER_GROUPS.get(k, k) for k in kind_counts if k not in _SESSION_MARKER_KINDS}
    )
    filter_controls = "".join(
        f'<label><input type="checkbox" class="kind-filter" '
        f'data-filter="{_esc_attr(group)}" checked> '
        f'{_esc(label)} <span class="count">{group_counts.get(label, 0)}</span></label>'
        for group in filter_groups
        for label in (_FILTER_LABELS.get(group, group),)
    )

    header = (
        f"<header>"
        f"<h1>jutul-agent transcript</h1>"
        f"{header_meta}"
        f"</header>"
        f'<div class="controls">'
        f"<fieldset><legend>Filter</legend>{filter_controls}</fieldset>"
        f'<button type="button" id="collapse-all">Collapse all</button>'
        f'<button type="button" id="expand-all">Expand all</button>'
        f"</div>"
        f'<div class="timeline">{"".join(parts)}</div>'
        f'<p class="share-note">To share this transcript, distribute the session folder '
        f"(<code>transcript.html</code> plus any <code>artifacts/</code>) or run "
        f"<code>jutul-agent transcript --bundle</code> to create a zip.</p>"
    )
    return header, kind_counts


def render_html(events: Iterable[Event], *, artifact_dirs: Sequence[Path] = ()) -> str:
    """Render trace events as a complete, self-contained HTML document.

    When ``artifact_dirs`` are given, image artifacts are inlined as base64 data
    URIs so the transcript shows its plots wherever it is opened; otherwise they
    keep their recorded relative path (which resolves beside the file, e.g. in a
    ``--bundle`` zip).
    """
    body, _ = _render_body(events, artifact_dirs)
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f'<meta http-equiv="Content-Security-Policy" content="{_CSP}">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>jutul-agent transcript</title>\n"
        f"<style>{_STYLES}\n{pygments_css()}</style>\n"
        "</head>\n"
        "<body>\n"
        f'<div class="wrap">{body}</div>\n'
        f"<script>{_SCRIPT}</script>\n"
        "</body>\n"
        "</html>\n"
    )
