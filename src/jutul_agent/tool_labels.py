"""Friendly, user-facing names for the agent's tools.

One source of truth shared by every surface that shows a tool to a person (the
live TUI tool cards, the approval prompt, and the HTML transcript), so a tool
reads the same everywhere. The agent-facing tool *id* (``run_julia``,
``execute``, ...) is never shown raw to the user.
"""

from __future__ import annotations

_TOOL_LABELS: dict[str, str] = {
    "run_julia": "Julia",
    "plot_julia": "Julia plot",
    "execute": "Shell",
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "delete": "Delete",
    "glob": "Find files",
    "grep": "Search",
    "ls": "List",
    "write_todos": "Plan",
    "task": "Delegate",
    "record_attempt": "Record attempt",
    "write_report": "Report",
    "reset_julia": "Reset Julia",
    "remember": "Remember",
}


def tool_label(name: str) -> str:
    """The user-facing name for a tool id; unknown tools show their raw id."""
    return _TOOL_LABELS.get(name, name)
