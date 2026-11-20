"""Human-in-the-loop approval modes for workspace file and shell tools."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from enum import StrEnum
from typing import Any, Protocol

from langchain.agents.middleware import InterruptOnConfig

# Tools that require human approval. Action name → human-readable description
# shown in the approval card. All entries use approve/reject only.
_APPROVAL_TOOLS: dict[str, str] = {
    "execute": "Run a shell command in the workspace.",
    "write_file": "Write a file in the workspace.",
    "edit_file": "Edit a file in the workspace.",
    "delete": "Delete a file or directory (directories go recursively).",
}

# Deliberately excludes `delete`: these are the tools "accept edits" mode and the
# session allowlist wave through in bulk, and a recursive delete is not something
# a blanket "always allow file edits" should cover. It stays one prompt per call.
_WORKSPACE_EDIT_TOOLS = frozenset({"write_file", "edit_file"})
_SHELL_TOOLS = frozenset({"execute"})

# Session allowlist categories.
ALLOWLIST_FILE_EDITS = "file_edits"
ALLOWLIST_SHELL = "shell"


def interrupt_on_config() -> dict[str, InterruptOnConfig]:
    """``interrupt_on`` map keyed by tool name (used by `create_deep_agent`)."""

    return {
        name: {"allowed_decisions": ["approve", "reject"], "description": description}
        for name, description in _APPROVAL_TOOLS.items()
    }


class ApprovalMode(StrEnum):
    """How side-effecting Deep Agents tools are gated before execution."""

    ASK = "ask"
    """Prompt before every gated tool (default)."""

    WORKSPACE = "workspace"
    """Auto-allow workspace write_file and edit_file; still prompt for the rest."""

    AUTO = "auto"
    """Auto-allow all configured approval tools (non-interactive trust mode)."""

    def display_label(self) -> str:
        return {
            ApprovalMode.ASK: "default",
            ApprovalMode.WORKSPACE: "accept edits",
            ApprovalMode.AUTO: "auto",
        }[self]

    def cycle_next(self) -> ApprovalMode:
        order = (ApprovalMode.ASK, ApprovalMode.WORKSPACE, ApprovalMode.AUTO)
        index = order.index(self)
        return order[(index + 1) % len(order)]


class ToolAllowlist:
    """Per-session categories the user chose to always allow."""

    def __init__(self) -> None:
        self._categories: set[str] = set()

    def add(self, category: str) -> None:
        self._categories.add(category)

    def contains(self, category: str) -> bool:
        return category in self._categories

    def __bool__(self) -> bool:
        return bool(self._categories)

    def categories(self) -> frozenset[str]:
        return frozenset(self._categories)


def tool_allowlist_category(tool_name: str) -> str | None:
    if tool_name in _WORKSPACE_EDIT_TOOLS:
        return ALLOWLIST_FILE_EDITS
    if tool_name in _SHELL_TOOLS:
        return ALLOWLIST_SHELL
    return None


def categories_for_interrupt(value: Any) -> frozenset[str]:
    if not isinstance(value, dict):
        return frozenset()
    action_requests = value.get("action_requests")
    if not isinstance(action_requests, list):
        return frozenset()
    categories: set[str] = set()
    for action in action_requests:
        if not isinstance(action, dict):
            continue
        category = tool_allowlist_category(str(action.get("name") or ""))
        if category:
            categories.add(category)
    return frozenset(categories)


def interrupt_matches_allowlist(value: Any, allowlist: ToolAllowlist) -> bool:
    categories = categories_for_interrupt(value)
    if not categories:
        return False
    return all(allowlist.contains(category) for category in categories)


def always_allow_categories(value: Any) -> frozenset[str]:
    """Categories a front end may offer to 'always allow' for this interrupt.

    Mirrors the TUI's policy so every interface behaves the same: shell approvals
    are deliberately one-off (a blanket 'always allow shell' is too broad), so a
    shell-only interrupt offers nothing; file edits and other categories do.
    """
    categories = categories_for_interrupt(value)
    if not categories or categories == {ALLOWLIST_SHELL}:
        return frozenset()
    return categories


def parse_approval_mode(value: str | None) -> ApprovalMode:
    if not value:
        return ApprovalMode.ASK
    normalized = value.strip().lower()
    for mode in ApprovalMode:
        if mode == normalized:
            return mode
    raise ValueError(
        f"unknown approval mode {value!r}; expected one of: "
        + ", ".join(m.value for m in ApprovalMode)
    )


def interrupt_on_for_mode(mode: ApprovalMode) -> dict[str, InterruptOnConfig]:
    """Map an approval mode to Deep Agents ``interrupt_on`` configuration."""

    if mode == ApprovalMode.AUTO:
        return {}
    if mode == ApprovalMode.WORKSPACE:
        # "Accept edits" auto-allows exactly the workspace edit tools; anything
        # else that is gated at all stays gated, so a newly gated tool is not
        # silently waved through by this mode.
        full = interrupt_on_config()
        return {name: cfg for name, cfg in full.items() if name not in _WORKSPACE_EDIT_TOOLS}
    return interrupt_on_config()


def interrupt_is_workspace_edits_only(value: Any) -> bool:
    """True when every action in an interrupt is write_file or edit_file."""

    if not isinstance(value, dict):
        return False
    action_requests = value.get("action_requests")
    if not isinstance(action_requests, list) or not action_requests:
        return False
    for action in action_requests:
        if not isinstance(action, dict):
            return False
        if action.get("name") not in _WORKSPACE_EDIT_TOOLS:
            return False
    return True


def should_auto_approve_interrupt(
    value: Any,
    mode: ApprovalMode,
    *,
    allowlist: ToolAllowlist | None = None,
) -> bool:
    if allowlist is not None and interrupt_matches_allowlist(value, allowlist):
        return True
    if mode == ApprovalMode.AUTO:
        return True
    if mode == ApprovalMode.WORKSPACE:
        return interrupt_is_workspace_edits_only(value)
    return False


# ---------------------------------------------------------------------------
# Interrupt decisions and resume payloads.
#
# Shared across interfaces (TUI, server) so every front end resolves an
# approval interrupt the same way. An interrupt payload follows deepagents'
# contract: a dict with an ``action_requests`` list and a parallel
# ``review_configs`` list of allowed-decision policies.

# The decisions a front end can resolve an interrupt with.
SUPPORTED_APPROVAL_DECISIONS = frozenset({"approve", "reject", "respond"})


class SupportsInterrupt(Protocol):
    """A pending interrupt: an id and its raw payload value."""

    interrupt_id: str
    value: Any


def review_config_map(review_configs: Any) -> dict[str, frozenset[str]]:
    """Map each action name to its allowed decisions from a ``review_configs`` list."""

    config_map: dict[str, frozenset[str]] = {}
    if not isinstance(review_configs, list):
        return config_map
    for review in review_configs:
        if not isinstance(review, dict):
            continue
        action_name = review.get("action_name")
        allowed = review.get("allowed_decisions")
        if isinstance(action_name, str) and isinstance(allowed, list):
            config_map[action_name] = frozenset(str(item) for item in allowed)
    return config_map


def allowed_decisions_for_interrupt(value: Any) -> frozenset[str]:
    """Return the intersection of decisions allowed across an interrupt's actions.

    An interrupt may bundle multiple ``action_requests``; each can declare its
    own ``allowed_decisions`` in a sibling ``review_configs`` block. A front end
    can only resume an interrupt with a decision every action accepts, so we
    intersect. Empty / malformed payloads fall back to all supported decisions,
    matching deepagents' default.
    """

    if not isinstance(value, dict):
        return SUPPORTED_APPROVAL_DECISIONS

    config_map = review_config_map(value.get("review_configs"))
    action_requests = value.get("action_requests")
    if not isinstance(action_requests, list):
        return SUPPORTED_APPROVAL_DECISIONS

    shared: frozenset[str] | None = None
    for action in action_requests:
        if not isinstance(action, dict):
            continue
        action_name = str(action.get("name") or "")
        allowed = config_map.get(action_name, SUPPORTED_APPROVAL_DECISIONS)
        shared = allowed if shared is None else shared & allowed
    return shared if shared is not None else SUPPORTED_APPROVAL_DECISIONS


def pending_allowed_decisions(interrupts: Iterable[SupportsInterrupt]) -> frozenset[str]:
    """Decisions every pending interrupt accepts, scoped to the supported set."""

    shared: frozenset[str] | None = None
    for interrupt in interrupts:
        allowed = allowed_decisions_for_interrupt(interrupt.value)
        shared = allowed if shared is None else shared & allowed
    if shared is None:
        return frozenset()
    return shared & SUPPORTED_APPROVAL_DECISIONS


def build_resume_payload(
    interrupts: Iterable[SupportsInterrupt],
    decision: dict[str, str],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    """Apply one decision to every action of every pending interrupt, for ``resume``.

    The result is the ``Command(resume=...)`` payload deepagents expects: keyed
    by interrupt id, each carrying one decision per ``action_request``.
    """

    payload: dict[str, dict[str, list[dict[str, str]]]] = {}
    for interrupt in interrupts:
        value = interrupt.value if isinstance(interrupt.value, dict) else {}
        action_requests = value.get("action_requests")
        count = len(action_requests) if isinstance(action_requests, list) else 1
        payload[interrupt.interrupt_id] = {"decisions": [deepcopy(decision) for _ in range(count)]}
    return payload
