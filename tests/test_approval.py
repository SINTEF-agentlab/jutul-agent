from __future__ import annotations

from pathlib import Path

from jutul_agent.approval_preview import (
    _apply_edit_preview,
    allowed_decisions_for_interrupt,
    render_interrupt_cards,
)


def test_render_execute_approval_card(tmp_path: Path) -> None:
    cards = render_interrupt_cards(
        "interrupt-1",
        {
            "action_requests": [
                {
                    "name": "execute",
                    "args": {"command": "pwd"},
                    "description": "Run a shell command in the jutul-agent workspace.",
                }
            ],
            "review_configs": [
                {"action_name": "execute", "allowed_decisions": ["approve", "reject"]}
            ],
        },
        workspace_root=tmp_path,
    )

    assert len(cards) == 1
    assert cards[0].title == "Approval · Shell"
    assert cards[0].allowed_decisions == frozenset({"approve", "reject"})
    assert "Run a shell command in the jutul-agent workspace." in cards[0].body
    assert "```sh" in cards[0].body
    assert "pwd" in cards[0].body


def test_render_write_file_card_for_new_file(tmp_path: Path) -> None:
    cards = render_interrupt_cards(
        "interrupt-1",
        {
            "action_requests": [
                {
                    "name": "write_file",
                    "args": {"file_path": "/notes.txt", "content": "hello\n"},
                    "description": "Write a file in the jutul-agent workspace.",
                }
            ],
            "review_configs": [
                {"action_name": "write_file", "allowed_decisions": ["approve", "reject"]}
            ],
        },
        workspace_root=tmp_path,
    )

    assert len(cards) == 1
    assert "`/notes.txt`" in cards[0].body  # the target path (title already names the tool)
    assert "#### Content Preview" in cards[0].body
    assert "hello" in cards[0].body


def test_render_write_file_card_for_existing_file_shows_diff(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("old\n", encoding="utf-8")

    cards = render_interrupt_cards(
        "interrupt-1",
        {
            "action_requests": [
                {
                    "name": "write_file",
                    "args": {"file_path": "/notes.txt", "content": "new\n"},
                    "description": "Write a file in the jutul-agent workspace.",
                }
            ],
            "review_configs": [
                {"action_name": "write_file", "allowed_decisions": ["approve", "reject"]}
            ],
        },
        workspace_root=tmp_path,
    )

    assert len(cards) == 1
    assert "#### Diff" in cards[0].body
    assert "-old" in cards[0].body
    assert "+new" in cards[0].body


def test_render_edit_file_card_shows_diff(tmp_path: Path) -> None:
    target = tmp_path / "notes.txt"
    target.write_text("hello\nworld\n", encoding="utf-8")

    cards = render_interrupt_cards(
        "interrupt-1",
        {
            "action_requests": [
                {
                    "name": "edit_file",
                    "args": {
                        "file_path": "/notes.txt",
                        "old_string": "world",
                        "new_string": "battmo",
                    },
                    "description": "Edit a file in the jutul-agent workspace.",
                }
            ],
            "review_configs": [
                {"action_name": "edit_file", "allowed_decisions": ["approve", "reject"]}
            ],
        },
        workspace_root=tmp_path,
    )

    assert len(cards) == 1
    assert "#### Diff" in cards[0].body
    assert "-world" in cards[0].body
    assert "+battmo" in cards[0].body


def test_render_edit_file_card_reports_unavailable_preview(tmp_path: Path) -> None:
    cards = render_interrupt_cards(
        "interrupt-1",
        {
            "action_requests": [
                {
                    "name": "edit_file",
                    "args": {
                        "file_path": "/missing.txt",
                        "old_string": "hello",
                        "new_string": "goodbye",
                    },
                    "description": "Edit a file in the jutul-agent workspace.",
                }
            ],
            "review_configs": [
                {"action_name": "edit_file", "allowed_decisions": ["approve", "reject"]}
            ],
        },
        workspace_root=tmp_path,
    )

    assert len(cards) == 1
    assert "Preview unavailable" in cards[0].body


def test_allowed_decisions_intersection() -> None:
    value = {
        "action_requests": [{"name": "execute"}, {"name": "write_file"}],
        "review_configs": [
            {"action_name": "execute", "allowed_decisions": ["approve", "reject", "respond"]},
            {"action_name": "write_file", "allowed_decisions": ["approve", "reject"]},
        ],
    }
    assert allowed_decisions_for_interrupt(value) == frozenset({"approve", "reject"})


def test_allowed_decisions_defaults_when_malformed() -> None:
    assert allowed_decisions_for_interrupt(None) == frozenset({"approve", "reject", "respond"})


def test_apply_edit_preview_reports_ambiguous_match() -> None:
    updated, occurrences, error = _apply_edit_preview(
        "hello hello",
        "hello",
        "goodbye",
        replace_all=False,
    )
    assert updated is None
    assert occurrences == 2
    assert error is not None
    assert "ambiguous" in error
