"""Tests for the server wire protocol and the shared HITL decision helpers."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from jutul_agent.agent.approval import (
    SUPPORTED_APPROVAL_DECISIONS,
    allowed_decisions_for_interrupt,
    build_resume_payload,
    pending_allowed_decisions,
)
from jutul_agent.agent.turns import (
    TurnInterrupt,
    TurnReasoningDelta,
    TurnTextDelta,
    TurnTextEnd,
    TurnToolEvent,
)
from jutul_agent.interfaces.server import protocol
from jutul_agent.tool_labels import tool_label

# --- to_wire: streamed events ---------------------------------------------


def test_text_chunk_wire() -> None:
    assert protocol.to_wire(TurnTextDelta(text="hello")) == {
        "type": "text",
        "text": "hello",
    }


def test_empty_text_chunk_is_skipped() -> None:
    assert protocol.to_wire(TurnTextDelta(text="")) is None
    assert protocol.to_wire(TurnTextEnd()) is None


def test_reasoning_delta_wire() -> None:
    assert protocol.to_wire(TurnReasoningDelta(text="thinking")) == {
        "type": "reasoning",
        "text": "thinking",
    }


def test_empty_reasoning_is_skipped() -> None:
    assert protocol.to_wire(TurnReasoningDelta(text="")) is None


def test_tool_event_wire() -> None:
    event = TurnToolEvent(
        event="finished",
        tool_name="run_julia",
        tool_call_id="abc",
        args={"code": "1+1"},
        content="2",
    )
    assert protocol.to_wire(event) == {
        "type": "tool",
        "event": "finished",
        "name": "run_julia",
        "label": "Julia",  # tool_label("run_julia")
        "tool_call_id": "abc",
        "args": {"code": "1+1"},
        "content": "2",
    }


def test_unknown_event_is_none() -> None:
    assert protocol.to_wire(object()) is None


# --- end-of-turn / out-of-band serializers --------------------------------


def test_interrupt_wire() -> None:
    value = {
        "action_requests": [
            {"name": "execute", "args": {"command": "ls"}, "description": "run ls"},
        ],
        "review_configs": [
            {"action_name": "execute", "allowed_decisions": ["approve", "reject"]},
        ],
    }
    wire = protocol.interrupt_to_wire(TurnInterrupt(interrupt_id="i1", value=value))
    body = wire["actions"][0].pop("body")
    assert "```" in body and "ls" in body  # the TUI card body rides along per action
    assert wire == {
        "type": "interrupt",
        "interrupt_id": "i1",
        "actions": [
            {
                "name": "execute",
                "label": tool_label("execute"),
                "args": {"command": "ls"},
                "description": "run ls",
            }
        ],
        "allowed_decisions": ["approve", "reject"],
        "allowlist": [],  # shell approvals are one-off — no "always allow"
    }


def test_interrupt_wire_offers_always_allow_for_file_edits() -> None:
    # A file-edit interrupt carries its allowlist category, so a front end can offer
    # "always allow" (shell does not — see test_interrupt_wire).
    value = {
        "action_requests": [{"name": "write_file", "args": {"file_path": "a.jl"}}],
        "review_configs": [
            {"action_name": "write_file", "allowed_decisions": ["approve", "reject"]}
        ],
    }
    wire = protocol.interrupt_to_wire(TurnInterrupt(interrupt_id="i2", value=value))
    assert wire["allowlist"] == ["file_edits"]


def test_usage_wire_takes_last() -> None:
    m1 = AIMessage(
        content="a",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    m2 = AIMessage(
        content="b",
        usage_metadata={"input_tokens": 20, "output_tokens": 4, "total_tokens": 24},
    )
    assert protocol.usage_to_wire([m1, m2]) == {
        "type": "usage",
        "input_tokens": 20,
        "output_tokens": 4,
        "total_tokens": 24,
        "model_calls": 2,
    }


def test_usage_wire_none_when_absent() -> None:
    assert protocol.usage_to_wire([AIMessage(content="x")]) is None


def test_turn_end_wire() -> None:
    assert protocol.turn_end_to_wire([AIMessage(content="done")]) == {
        "type": "turn_end",
        "text": "done",
    }


def test_artifact_wire() -> None:
    payload = {
        "mime": "image/png",
        "caption": "fig",
        "slot": "s",
        "format": "png",
        "path": "artifacts/x.png",
    }
    assert protocol.artifact_to_wire(payload, url="/sessions/1/artifacts/x.png") == {
        "type": "artifact",
        "url": "/sessions/1/artifacts/x.png",
        "mime": "image/png",
        "caption": "fig",
        "slot": "s",
        "format": "png",
    }


def test_viz_and_ui_wire() -> None:
    assert protocol.viz_to_wire("http://x/viz", title="T") == {
        "type": "viz",
        "url": "http://x/viz",
        "title": "T",
        "kind": "plot",
        "poster": None,
        "slot": None,
    }
    assert protocol.viz_to_wire(
        "http://x/r.html", title="R", kind="report", poster="http://x/p.png", slot="report"
    ) == {
        "type": "viz",
        "url": "http://x/r.html",
        "title": "R",
        "kind": "report",
        "poster": "http://x/p.png",
        "slot": "report",
    }
    assert protocol.ui_command("set_param", {"p": 2}) == {
        "type": "ui",
        "action": "set_param",
        "payload": {"p": 2},
    }
    assert protocol.ui_command("noop") == {"type": "ui", "action": "noop", "payload": {}}


# --- shared HITL helpers (now in agent.approval) ---------------------------


def _interrupt(value: dict) -> TurnInterrupt:
    return TurnInterrupt(interrupt_id="i", value=value)


def test_allowed_decisions_intersect_across_actions() -> None:
    value = {
        "action_requests": [{"name": "execute"}, {"name": "write_file"}],
        "review_configs": [
            {"action_name": "execute", "allowed_decisions": ["approve", "reject"]},
            {
                "action_name": "write_file",
                "allowed_decisions": ["approve", "reject", "respond"],
            },
        ],
    }
    assert allowed_decisions_for_interrupt(value) == frozenset({"approve", "reject"})


def test_allowed_decisions_default_for_malformed() -> None:
    assert allowed_decisions_for_interrupt(None) == SUPPORTED_APPROVAL_DECISIONS


def test_pending_allowed_decisions_empty() -> None:
    assert pending_allowed_decisions([]) == frozenset()


def test_pending_allowed_decisions_intersect_across_interrupts() -> None:
    a = _interrupt(
        {
            "action_requests": [{"name": "execute"}],
            "review_configs": [
                {"action_name": "execute", "allowed_decisions": ["approve", "reject"]}
            ],
        }
    )
    b = _interrupt(
        {
            "action_requests": [{"name": "write_file"}],
            "review_configs": [{"action_name": "write_file", "allowed_decisions": ["approve"]}],
        }
    )
    assert pending_allowed_decisions([a, b]) == frozenset({"approve"})


def test_build_resume_payload_one_decision_per_action() -> None:
    interrupt = TurnInterrupt(
        interrupt_id="i9",
        value={"action_requests": [{"name": "execute"}, {"name": "execute"}]},
    )
    assert build_resume_payload([interrupt], {"type": "approve"}) == {
        "i9": {"decisions": [{"type": "approve"}, {"type": "approve"}]}
    }


def test_build_resume_payload_copies_are_independent() -> None:
    interrupt = TurnInterrupt(
        interrupt_id="i9",
        value={"action_requests": [{"name": "execute"}, {"name": "execute"}]},
    )
    decisions = build_resume_payload([interrupt], {"type": "reject"})["i9"]["decisions"]
    decisions[0]["message"] = "x"
    assert "message" not in decisions[1]


def test_interrupt_wire_carries_the_edit_diff(tmp_path, monkeypatch) -> None:
    """A file edit's approval body shows the on-disk diff, browser and terminal alike."""
    from jutul_agent import paths

    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    (tmp_path / "a.jl").write_text("x = 1\n", encoding="utf-8")
    value = {
        "action_requests": [
            {
                "name": "edit_file",
                "args": {"file_path": "a.jl", "old_string": "x = 1", "new_string": "x = 2"},
            }
        ],
        "review_configs": [{"action_name": "edit_file", "allowed_decisions": ["approve"]}],
    }
    wire = protocol.interrupt_to_wire(TurnInterrupt(interrupt_id="i3", value=value))
    body = wire["actions"][0]["body"]
    assert "-x = 1" in body and "+x = 2" in body  # a real unified diff, not an arg dump
