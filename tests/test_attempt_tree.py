"""Tests for the attempt tree assembled from trace events."""

from __future__ import annotations

from jutul_agent.lab.fakes import make_event
from jutul_agent.transcript.attempts import Attempt, build_attempt_tree


def _attempt_payload(
    attempt_id: str,
    *,
    parent_id: str | None = None,
    rationale: str = "",
    metrics: dict | None = None,
) -> dict:
    return {
        "id": attempt_id,
        "parent_id": parent_id,
        "rationale": rationale,
        "parameters_changed": {},
        "metrics": metrics or {},
    }


def test_build_attempt_tree_linear() -> None:
    events = [
        make_event(1, "attempt", _attempt_payload("a1", rationale="first")),
        make_event(2, "attempt", _attempt_payload("a2", parent_id="a1", rationale="second")),
        make_event(3, "attempt", _attempt_payload("a3", parent_id="a2", rationale="third")),
    ]
    roots = build_attempt_tree(events)
    assert len(roots) == 1
    assert roots[0].id == "a1"
    assert roots[0].children[0].id == "a2"
    assert roots[0].children[0].children[0].id == "a3"


def test_build_attempt_tree_branching() -> None:
    events = [
        make_event(1, "attempt", _attempt_payload("a1", rationale="root")),
        make_event(2, "attempt", _attempt_payload("a2", parent_id="a1", rationale="branch-a")),
        make_event(3, "attempt", _attempt_payload("a3", parent_id="a1", rationale="branch-b")),
    ]
    roots = build_attempt_tree(events)
    assert len(roots) == 1
    assert {c.id for c in roots[0].children} == {"a2", "a3"}


def test_build_attempt_tree_orphan_parent_becomes_root() -> None:
    events = [
        make_event(1, "attempt", _attempt_payload("a1", parent_id="missing", rationale="orphan")),
    ]
    roots = build_attempt_tree(events)
    assert len(roots) == 1
    assert isinstance(roots[0], Attempt)
    assert roots[0].id == "a1"
