"""Render human-in-the-loop approval requests as markdown cards.

Shared by every surface that shows an approval: the TUI mounts the cards as
widgets, the web server carries each card body over the wire so the browser
shows the same diff/command/preview the terminal does. Approval is the trust
surface; no front end should show less than another.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jutul_agent.agent.approval import (
    SUPPORTED_APPROVAL_DECISIONS,
    allowed_decisions_for_interrupt,
    review_config_map,
)
from jutul_agent.paths import resolve_in_workspace
from jutul_agent.preview import fenced_block, truncate_preview
from jutul_agent.tool_labels import tool_label

# The decision helpers live in agent.approval; re-exported for card call sites.
__all__ = [
    "SUPPORTED_APPROVAL_DECISIONS",
    "ApprovalCard",
    "allowed_decisions_for_interrupt",
    "compute_unified_diff",
    "fenced_block",
    "render_interrupt_cards",
    "truncate_preview",
]


@dataclass(frozen=True)
class ApprovalCard:
    """Structured approval content ready for chat rendering."""

    title: str
    body: str
    tool_name: str
    allowed_decisions: frozenset[str]


def render_interrupt_cards(
    interrupt_id: str,
    value: dict[str, Any],
    *,
    workspace_root: Path,
) -> list[ApprovalCard]:
    """Render one approval card per action in a LangGraph interrupt payload.

    Assumes deepagents' interrupt contract: ``value`` is a dict with an
    ``action_requests`` list (each item carrying ``name`` and ``args``) and
    a parallel ``review_configs`` list of allowed-decision policies.
    """

    config_map = review_config_map(value.get("review_configs"))
    cards: list[ApprovalCard] = []
    for action in value["action_requests"]:
        tool_name = str(action.get("name") or "tool")
        tool_args = action.get("args") if isinstance(action.get("args"), dict) else {}
        description = action.get("description")
        cards.append(
            ApprovalCard(
                title=f"Approval · {tool_label(tool_name)}",
                body=_render_card_body(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    description=str(description) if description else None,
                    workspace_root=workspace_root,
                ),
                tool_name=tool_name,
                allowed_decisions=config_map.get(tool_name, SUPPORTED_APPROVAL_DECISIONS),
            )
        )
    return cards


def _render_card_body(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    description: str | None,
    workspace_root: Path,
) -> str:
    lines: list[str] = []

    if description:
        lines.append(description)
        lines.append("")

    # The card title already names the tool; show only the target it acts on
    # (the file for an edit) and the action itself (the command, the diff).
    target = _tool_target(tool_name, tool_args)
    if target is not None:
        lines.append(_inline_code(target))
        lines.append("")

    lines.extend(_tool_sections(tool_name, tool_args, workspace_root=workspace_root))
    return "\n".join(lines).strip()


def _tool_sections(tool_name: str, tool_args: dict[str, Any], *, workspace_root: Path) -> list[str]:
    if tool_name == "execute":
        command = str(tool_args.get("command") or "").strip()
        if not command:
            return []
        return [fenced_block(truncate_preview(command), language="sh")]

    if tool_name == "write_file":
        return _write_file_sections(tool_args, workspace_root=workspace_root)

    if tool_name == "edit_file":
        return _edit_file_sections(tool_args, workspace_root=workspace_root)

    if not tool_args:
        return []

    rendered_args = "\n".join(
        f"- {_inline_code(str(key))}: {_inline_code(repr(value))}"
        for key, value in tool_args.items()
    )
    return ["#### Arguments", "", rendered_args]


def _write_file_sections(tool_args: dict[str, Any], *, workspace_root: Path) -> list[str]:
    sections: list[str] = []
    path_str = str(tool_args.get("file_path") or tool_args.get("path") or "")
    content = str(tool_args.get("content") or "")
    physical_path = resolve_in_workspace(path_str, workspace=workspace_root)
    existing_text = _safe_read(physical_path) if physical_path and physical_path.exists() else None

    if existing_text is None:
        sections.append("#### Content Preview")
        sections.append("")
        sections.append(fenced_block(truncate_preview(content), language=_guess_language(path_str)))
        return sections

    diff = compute_unified_diff(existing_text, content, path_str or physical_path.name)
    sections.append("#### Diff")
    sections.append("")
    if diff is None:
        sections.append("No content changes detected.")
    else:
        sections.append(fenced_block(truncate_preview(diff), language="diff"))
    return sections


def _edit_file_sections(tool_args: dict[str, Any], *, workspace_root: Path) -> list[str]:
    path_str = str(tool_args.get("file_path") or tool_args.get("path") or "")
    physical_path = resolve_in_workspace(path_str, workspace=workspace_root)
    if physical_path is None:
        return ["> Preview unavailable: unable to resolve the target path inside the workspace."]
    if not physical_path.exists():
        return ["> Preview unavailable: the target file does not exist in the workspace."]

    existing_text = _safe_read(physical_path)
    if existing_text is None:
        return ["> Preview unavailable: unable to read the current file contents."]

    old_string = str(tool_args.get("old_string") or "")
    new_string = str(tool_args.get("new_string") or "")
    replace_all = bool(tool_args.get("replace_all"))
    updated_text, occurrences, error = _apply_edit_preview(
        existing_text,
        old_string,
        new_string,
        replace_all=replace_all,
    )
    if error is not None or updated_text is None:
        return [f"> Preview unavailable: {error or 'unknown edit preview failure.'}"]

    details = [
        "#### Edit Summary",
        "",
        f"- Matches: {occurrences}",
        f"- Replace all: {'yes' if replace_all else 'no'}",
        "",
        "#### Diff",
        "",
    ]

    diff = compute_unified_diff(existing_text, updated_text, path_str or physical_path.name)
    if diff is None:
        details.append("No content changes detected.")
    else:
        details.append(fenced_block(truncate_preview(diff), language="diff"))
    return details


def _apply_edit_preview(
    existing_text: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool,
) -> tuple[str | None, int, str | None]:
    if not old_string:
        return None, 0, "the edit request did not include old_string."

    occurrences = existing_text.count(old_string)
    if occurrences == 0:
        return None, 0, "old_string was not found in the current file contents."
    if not replace_all and occurrences > 1:
        return (
            None,
            occurrences,
            "old_string matches multiple regions; preview is ambiguous until the edit is narrowed.",
        )

    updated = (
        existing_text.replace(old_string, new_string)
        if replace_all
        else existing_text.replace(old_string, new_string, 1)
    )
    return updated, occurrences, None


def compute_unified_diff(before: str, after: str, display_path: str) -> str | None:
    """Compute a unified diff for approval previews."""

    diff_lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"{display_path} (before)",
            tofile=f"{display_path} (after)",
            lineterm="",
            n=3,
        )
    )
    if not diff_lines:
        return None
    return "\n".join(diff_lines)


def _tool_target(tool_name: str, tool_args: dict[str, Any]) -> str | None:
    if tool_name in {"write_file", "edit_file", "read_file"}:
        value = tool_args.get("file_path") or tool_args.get("path")
        return str(value) if value else None
    return None


def _guess_language(path_str: str) -> str:
    suffix = Path(path_str).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".jl":
        return "julia"
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".json", ".toml", ".yaml", ".yml", ".txt"}:
        return suffix.lstrip(".") or "text"
    return "text"


def _inline_code(text: str) -> str:
    return f"`{text.replace('`', '\\`')}`"


def _safe_read(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
