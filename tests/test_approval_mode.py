"""Tests for approval mode configuration."""

from __future__ import annotations

from jutul_agent.agent.approval import (
    ALLOWLIST_FILE_EDITS,
    ApprovalMode,
    ToolAllowlist,
    interrupt_is_workspace_edits_only,
    interrupt_matches_allowlist,
    interrupt_on_for_mode,
    parse_approval_mode,
    should_auto_approve_interrupt,
)


def test_parse_approval_mode_defaults_to_ask() -> None:
    assert parse_approval_mode(None) is ApprovalMode.ASK


def test_interrupt_on_for_auto_is_empty() -> None:
    assert interrupt_on_for_mode(ApprovalMode.AUTO) == {}


def test_interrupt_on_for_workspace_drops_only_the_edit_tools() -> None:
    config = interrupt_on_for_mode(ApprovalMode.WORKSPACE)
    assert not {"write_file", "edit_file"} & set(config)
    assert {"execute", "delete"} <= set(config)


def test_interrupt_on_for_ask_includes_file_tools() -> None:
    config = interrupt_on_for_mode(ApprovalMode.ASK)
    assert {"execute", "write_file", "edit_file", "delete"} <= set(config)


def test_workspace_edits_only_interrupt() -> None:
    payload = {
        "action_requests": [
            {"name": "write_file", "args": {"path": "/candidate.jl"}},
        ]
    }
    assert interrupt_is_workspace_edits_only(payload)
    assert should_auto_approve_interrupt(payload, ApprovalMode.WORKSPACE)


def test_execute_interrupt_not_auto_in_workspace_mode() -> None:
    payload = {"action_requests": [{"name": "execute", "args": {"command": "ls"}}]}
    assert not should_auto_approve_interrupt(payload, ApprovalMode.WORKSPACE)


def test_allowlist_auto_approves_matching_interrupt() -> None:
    payload = {
        "action_requests": [{"name": "write_file", "args": {"path": "/candidate.jl"}}],
    }
    allowlist = ToolAllowlist()
    allowlist.add(ALLOWLIST_FILE_EDITS)
    assert interrupt_matches_allowlist(payload, allowlist)
    assert should_auto_approve_interrupt(payload, ApprovalMode.ASK, allowlist=allowlist)


def test_delete_is_never_waved_through_with_file_edits() -> None:
    # A recursive delete is gated on its own: neither "accept edits" mode nor an
    # "always allow file edits" choice covers it, so it costs one prompt each time.
    payload = {"action_requests": [{"name": "delete", "args": {"file_path": "/scratch"}}]}
    allowlist = ToolAllowlist()
    allowlist.add(ALLOWLIST_FILE_EDITS)
    assert not interrupt_is_workspace_edits_only(payload)
    assert not interrupt_matches_allowlist(payload, allowlist)
    assert not should_auto_approve_interrupt(payload, ApprovalMode.WORKSPACE, allowlist=allowlist)


def test_approval_mode_cycles() -> None:
    assert ApprovalMode.ASK.cycle_next() is ApprovalMode.WORKSPACE
    assert ApprovalMode.WORKSPACE.cycle_next() is ApprovalMode.AUTO
    assert ApprovalMode.AUTO.cycle_next() is ApprovalMode.ASK
