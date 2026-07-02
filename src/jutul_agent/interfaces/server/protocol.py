"""Wire protocol: serialize a turn's events to JSON for network front ends.

The server attaches an ``on_message`` callback to a ``TurnRunner`` exactly as
the TUI does, but instead of rendering it serializes each event with ``to_wire``
and sends the dict down a WebSocket. This module is the single definition of
that schema, so every front end codes against one contract and the live stream
never drifts from what the runner emits.

The streaming events (``to_wire``) come from the runner's callback; the
end-of-turn events (interrupts, usage, final text) are built from the
``TurnRunResult`` after the turn drains. ``artifact``/``viz``/``ui`` are emitted
out of band by the server and capability tools.

Import-light on purpose (no FastAPI): the schema can be used and tested without
loading the web server stack.
"""

from __future__ import annotations

from typing import Any

from jutul_agent.agent.approval import (
    SupportsInterrupt,
    allowed_decisions_for_interrupt,
    always_allow_categories,
)
from jutul_agent.agent.turns import (
    TurnReasoningDelta,
    TurnTextDelta,
    TurnTextEnd,
    TurnToolEvent,
    final_assistant_text,
    usage_from_messages,
)
from jutul_agent.tool_labels import tool_label

__all__ = [
    "artifact_to_wire",
    "credential_required_to_wire",
    "interrupt_to_wire",
    "notice_to_wire",
    "to_wire",
    "turn_end_to_wire",
    "ui_command",
    "usage_to_wire",
    "viz_to_wire",
]


def credential_required_to_wire(*, provider: str, label: str, env_var: str) -> dict[str, Any]:
    """A model switch that needs a provider key the server doesn't have yet.

    The front end shows a key prompt, POSTs the key to ``/credentials``, then
    retries the switch. The same shape is used as the ``detail`` of the HTTP 400
    that ``POST /sessions`` raises when the new session's model has no key.
    """

    return {
        "type": "credential_required",
        "error": "credential_required",
        "provider": provider,
        "label": label,
        "env_var": env_var,
    }


def to_wire(event: Any) -> dict[str, Any] | None:
    """Serialize one streamed ``TurnRunner`` event, or ``None`` if it carries nothing.

    Handles the three event types the runner emits to ``on_message``: assistant
    text chunks, reasoning deltas, and tool-call lifecycle events. Anything else
    (and an empty text/reasoning delta) returns ``None`` so the caller can skip it.
    """

    if isinstance(event, TurnTextDelta):
        return {"type": "text", "text": event.text} if event.text else None

    if isinstance(event, TurnTextEnd):
        # The end-of-turn signal on the wire is ``turn_end``, built from the
        # final messages; the flush marker carries nothing for a network client.
        return None

    if isinstance(event, TurnReasoningDelta):
        return {"type": "reasoning", "text": event.text} if event.text else None

    if isinstance(event, TurnToolEvent):
        return {
            "type": "tool",
            "event": event.event,
            "name": event.tool_name,
            "label": tool_label(event.tool_name),
            "tool_call_id": event.tool_call_id,
            "args": event.args,
            "content": event.content,
        }

    return None


def interrupt_to_wire(interrupt: SupportsInterrupt) -> dict[str, Any]:
    """Serialize a pending approval interrupt: its id, actions, and allowed decisions."""

    value = interrupt.value if isinstance(interrupt.value, dict) else {}
    raw_actions = value.get("action_requests")
    actions: list[dict[str, Any]] = []
    if isinstance(raw_actions, list):
        for action in raw_actions:
            if not isinstance(action, dict):
                continue
            name = str(action.get("name") or "tool")
            args = action.get("args")
            actions.append(
                {
                    "name": name,
                    "label": tool_label(name),
                    "args": args if isinstance(args, dict) else {},
                    "description": action.get("description"),
                }
            )
    return {
        "type": "interrupt",
        "interrupt_id": interrupt.interrupt_id,
        "actions": actions,
        "allowed_decisions": sorted(allowed_decisions_for_interrupt(interrupt.value)),
        # Categories a front end may offer to "always allow" this session (empty for
        # one-off-only interrupts like shell). The client turns these into a button.
        "allowlist": sorted(always_allow_categories(interrupt.value)),
    }


def usage_to_wire(messages: list[Any]) -> dict[str, Any] | None:
    """Token usage for the turn, from the newest model message that reported it."""

    usages = usage_from_messages(messages)
    if not usages:
        return None
    last = usages[-1]
    return {
        "type": "usage",
        "input_tokens": int(last.get("input_tokens") or 0),
        "output_tokens": int(last.get("output_tokens") or 0),
        "total_tokens": int(last.get("total_tokens") or 0),
        "model_calls": len(usages),
    }


def turn_end_to_wire(messages: list[Any]) -> dict[str, Any]:
    """Signal the turn finished, carrying the final assistant text.

    ``final_assistant_text`` picks the last *assistant* message, not just the
    last message, so a trailing ToolMessage never surfaces as prose.
    """

    return {"type": "turn_end", "text": final_assistant_text(messages)}


def artifact_to_wire(payload: dict[str, Any], *, url: str) -> dict[str, Any]:
    """Serialize a produced artifact (plot PNG, report) as a fetchable URL.

    ``payload`` is the trace ``artifact`` event payload; ``url`` is where the
    server exposes that file for this session.
    """

    return {
        "type": "artifact",
        "url": url,
        "mime": payload.get("mime"),
        "caption": payload.get("caption"),
        "slot": payload.get("slot"),
        "format": payload.get("format"),
    }


def viz_to_wire(
    url: str,
    *,
    title: str | None = None,
    kind: str = "plot",
    poster: str | None = None,
    slot: str | None = None,
) -> dict[str, Any]:
    """Serialize an interactive view to pin in the front end's canvas.

    ``kind`` is ``"plot"`` (an interactive figure) or ``"report"`` (a document);
    a front end uses it only for the label/icon. ``poster`` is an optional image
    URL for a lightweight inline thumbnail, and ``slot`` is the view's stable key
    so a refreshed view replaces the previous one in place rather than stacking.
    """

    return {"type": "viz", "url": url, "title": title, "kind": kind, "poster": poster, "slot": slot}


def notice_to_wire(text: str) -> dict[str, Any]:
    """A server-originated system note (a command's result, e.g. /compact, /add-dir)."""

    return {"type": "notice", "text": text}


def ui_command(action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """A UI-control command for the front end: an opaque action plus its payload.

    The envelope is fixed; the ``action`` vocabulary belongs to the capability
    bundle that owns that part of the UI, not to this module.
    """

    return {"type": "ui", "action": action, "payload": payload or {}}


