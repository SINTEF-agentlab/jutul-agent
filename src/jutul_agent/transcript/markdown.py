"""Render a trace log as a markdown transcript."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jutul_agent.trace import Event, schema
from jutul_agent.transcript.events import ArtifactPayload


def _artifact_meta_line(artifact: ArtifactPayload) -> str | None:
    parts: list[str] = []
    if artifact.slot:
        parts.append(f"slot `{artifact.slot}`")
    if artifact.format:
        parts.append(artifact.format)
    if artifact.size_px is not None:
        parts.append(f"{artifact.size_px[0]}x{artifact.size_px[1]}")
    if not parts:
        return None
    return "*(" + ", ".join(parts) + ")*"


def _fmt_artifact_block(artifact: ArtifactPayload) -> list[str]:
    lines: list[str] = []
    meta = _artifact_meta_line(artifact)
    if meta:
        lines.append(meta)
    lines.append(f"![{artifact.caption}]({artifact.path or '?'})")
    if artifact.source_code:
        lines.extend(["", "```julia", artifact.source_code.rstrip(), "```"])
    return lines


def _fmt_args(args: Any) -> str:
    if not args:
        return "(no args)"
    if isinstance(args, dict):
        return "\n".join(f"- `{k}` = {v!r}" for k, v in args.items())
    return repr(args)


def _count(payload: dict[str, Any], key: str) -> Any:
    """A numeric payload field for display; unknown counts render as ``?``."""
    value = payload.get(key)
    return value if value is not None else "?"


def render_markdown(events: Iterable[Event]) -> str:
    lines: list[str] = []
    session_id: str | None = None

    for ev in events:
        payload = ev.payload
        ts = ev.timestamp
        kind = ev.kind

        if kind == schema.SESSION_START:
            session_id = payload.get("session_id") or session_id
            sim = payload.get("simulator") or "(none)"
            lines += [
                f"# Session `{session_id}`",
                "",
                f"- Started: `{ts}`",
                f"- Simulator: `{sim}`",
                "",
            ]
        elif kind == schema.SESSION_END:
            lines += ["---", f"_Session ended: `{ts}`_", ""]
        elif kind == schema.SESSION_RESUME:
            lines += ["---", f"_Session resumed: `{ts}`_", ""]
        elif kind == schema.SESSION_TITLE:
            title = str(payload.get("title") or "").strip()
            if title:
                lines += [f"**{title}**", ""]
        elif kind == schema.CONTEXT_COMPACTION:
            trigger = "manual" if payload.get("manual") else "automatic"
            summarized = _count(payload, "summarized")
            kept = _count(payload, "kept")
            lines += [
                f"_Context compacted ({trigger}): {summarized} messages summarized, "
                f"{kept} kept · `{ts}`_",
                "",
            ]
        elif kind == schema.MESSAGE_USER:
            lines += [f"## User · `{ts}`", "", str(payload.get("content", "")), ""]
        elif kind == schema.MESSAGE_REASONING:
            content = str(payload.get("content", ""))
            if not content.strip():
                continue
            lines += [f"## Reasoning · `{ts}`", "", content, ""]
        elif kind == schema.MESSAGE_ASSISTANT:
            content = str(payload.get("content", ""))
            if not content.strip():
                continue
            lines += [f"## Assistant · `{ts}`", "", content, ""]
        elif kind == schema.TOOL_CALL:
            name = payload.get("name", "?")
            lines += [
                f"### Tool call · `{name}` · `{ts}`",
                "",
                _fmt_args(payload.get("args")),
                "",
            ]
        elif kind == schema.TOOL_RESULT:
            name = payload.get("name", "?")
            content = str(payload.get("content", ""))
            lines += [
                f"### Tool result · `{name}` · `{ts}`",
                "",
                "```",
                content,
                "```",
                "",
            ]
        elif kind == schema.HITL_REQUEST:
            lines += [
                f"### Approval request · `{ts}`",
                "",
                _fmt_hitl_request(payload),
                "",
            ]
        elif kind == schema.HITL_RESPONSE:
            lines += [
                f"### Approval response · `{ts}`",
                "",
                _fmt_hitl_response(payload),
                "",
            ]
        elif kind == schema.ARTIFACT:
            artifact = ArtifactPayload.from_payload(payload)
            lines += [
                f"### Artifact · `{ts}`",
                "",
                *_fmt_artifact_block(artifact),
                "",
            ]
        else:
            lines += [f"### Event `{kind}` · `{ts}`", "", repr(payload), ""]

    return "\n".join(lines).rstrip() + "\n"


def _fmt_hitl_request(payload: dict[str, Any]) -> str:
    interrupt_id = payload.get("interrupt_id") or "?"
    value = payload.get("value")
    if not isinstance(value, dict):
        return f"- `interrupt_id` = {interrupt_id!r}\n- `value` = {value!r}"

    action_requests = value.get("action_requests") or []
    lines = [f"- `interrupt_id` = {interrupt_id!r}"]
    if isinstance(action_requests, list) and action_requests:
        lines.append("- `actions`:")
        for action in action_requests:
            if not isinstance(action, dict):
                lines.append(f"  - {action!r}")
                continue
            name = action.get("name", "?")
            args = action.get("args")
            lines.append(f"  - `{name}` {args!r}")
        return "\n".join(lines)

    lines.append(f"- `value` = {value!r}")
    return "\n".join(lines)


def _fmt_hitl_response(payload: dict[str, Any]) -> str:
    interrupt_id = payload.get("interrupt_id") or "?"
    response = payload.get("payload")
    return "\n".join(
        [
            f"- `interrupt_id` = {interrupt_id!r}",
            f"- `payload` = {response!r}",
        ]
    )
