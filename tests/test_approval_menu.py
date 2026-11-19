"""Tests for the approval selection menu."""

from __future__ import annotations

from makker.interfaces.tui.approval_menu import build_approval_options


def test_build_approval_options_for_shell() -> None:
    options = build_approval_options(
        allowed_decisions=frozenset({"approve", "reject"}),
        tool_names=["execute"],
        interrupt_values=[
            {"action_requests": [{"name": "execute", "args": {"command": "ls"}}]},
        ],
    )
    labels = [option.label for option in options]
    assert labels == ["Yes", "No, and tell the agent what to do differently"]


def test_build_approval_options_for_file_edits() -> None:
    options = build_approval_options(
        allowed_decisions=frozenset({"approve", "reject"}),
        tool_names=["edit_file"],
        interrupt_values=[
            {"action_requests": [{"name": "edit_file", "args": {"path": "/candidate.jl"}}]},
        ],
    )
    assert "file edits" in options[1].label
