"""Tests for the markdown transcript renderer."""

from __future__ import annotations

from jutul_agent.lab.fakes import make_event
from jutul_agent.transcript import render_markdown


def test_render_full_turn(snapshot) -> None:
    events = [
        make_event(1, "session_start", {"session_id": "abc", "simulator": None}),
        make_event(2, "message_user", {"content": "what is 2+2?"}),
        make_event(3, "tool_call", {"name": "run_julia", "args": {"code": "2+2"}}),
        make_event(4, "tool_result", {"name": "run_julia", "content": "4"}),
        make_event(5, "message_assistant", {"content": "The answer is 4."}),
        make_event(6, "session_end", {"session_id": "abc"}),
    ]
    assert render_markdown(events) == snapshot


def test_render_skips_empty_assistant(snapshot) -> None:
    events = [
        make_event(1, "message_assistant", {"content": ""}),
        make_event(2, "message_assistant", {"content": "hello"}),
    ]
    md = render_markdown(events)
    assert md.count("## Assistant") == 1
    assert md == snapshot


def test_render_hitl_events(snapshot) -> None:
    events = [
        make_event(
            1,
            "hitl_request",
            {
                "interrupt_id": "abc",
                "value": {"action_requests": [{"name": "execute", "args": {"command": "ls -la"}}]},
            },
        ),
        make_event(
            2,
            "hitl_response",
            {
                "interrupt_id": "abc",
                "payload": {"decisions": [{"type": "approve"}]},
            },
        ),
    ]
    assert render_markdown(events) == snapshot


def test_render_artifact(snapshot) -> None:
    events = [
        make_event(
            1,
            "artifact",
            {
                "path": "artifacts/plot-abc.png",
                "mime": "image/png",
                "caption": "pressure",
            },
        ),
    ]
    assert render_markdown(events) == snapshot


def test_render_artifact_with_provenance(snapshot) -> None:
    events = [
        make_event(
            1,
            "artifact",
            {
                "path": "artifacts/comparison.png",
                "mime": "image/png",
                "caption": "Well rates",
                "slot": "comparison",
                "format": "png",
                "size_px": [900, 500],
                "source_code": "well_rates_figure(wd)",
            },
        ),
    ]
    assert render_markdown(events) == snapshot
