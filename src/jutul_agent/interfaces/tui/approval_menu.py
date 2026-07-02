"""Selectable approval menu for pending tool interrupts."""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Static

from jutul_agent.agent.approval import (
    ALLOWLIST_FILE_EDITS,
    ALLOWLIST_SHELL,
    SUPPORTED_APPROVAL_DECISIONS,
    categories_for_interrupt,
)


@dataclass(frozen=True)
class ApprovalOption:
    """One row in the approval selection menu."""

    label: str
    decision: dict[str, str]
    allowlist_categories: frozenset[str] = frozenset()


def build_approval_options(
    *,
    allowed_decisions: frozenset[str],
    tool_names: list[str],
    interrupt_values: list[dict],
) -> list[ApprovalOption]:
    """Build Approve / always-allow / Reject options for the current interrupt."""

    supported = allowed_decisions & SUPPORTED_APPROVAL_DECISIONS
    options: list[ApprovalOption] = []

    if "approve" in supported:
        options.append(
            ApprovalOption(label="Yes", decision={"type": "approve"}),
        )
        always_label = _always_allow_label(tool_names, interrupt_values)
        if always_label:
            categories: set[str] = set()
            for value in interrupt_values:
                categories.update(categories_for_interrupt(value))
            options.append(
                ApprovalOption(
                    label=always_label,
                    decision={"type": "approve"},
                    allowlist_categories=frozenset(categories),
                ),
            )

    if "reject" in supported:
        options.append(
            ApprovalOption(
                label="No, and tell the agent what to do differently",
                decision={"type": "reject"},
            )
        )

    return options


def _always_allow_label(tool_names: list[str], interrupt_values: list[dict]) -> str | None:
    categories: set[str] = set()
    for value in interrupt_values:
        categories.update(categories_for_interrupt(value))
    if not categories:
        return None
    # Shell approvals are always one-off; no broad "always allow shell" policy.
    if categories == {ALLOWLIST_SHELL}:
        return None
    if categories == {ALLOWLIST_FILE_EDITS}:
        return "Yes, and always allow file edits in this workspace"
    if len(categories) == 1:
        name = next(iter(categories)).replace("_", " ")
        return f"Yes, and always allow {name} in this workspace"
    return "Yes, and always allow these actions in this workspace"


class ApprovalMenu(Vertical):
    """Arrow-key menu shown while a tool approval is pending."""

    DEFAULT_CSS = """
    ApprovalMenu {
        height: auto;
        padding: 0 1 1 1;
        display: none;
        border-top: solid $surface-lighten-1;
    }

    ApprovalMenu.visible {
        display: block;
    }

    ApprovalMenu .approval-options {
        height: auto;
        padding: 0;
    }

    ApprovalMenu .approval-hint {
        height: auto;
        color: $text-muted;
        padding: 0;
    }
    """

    # Navigation is driven by the prompt's key handler (`_handle_approval_nav`)
    # while the prompt is focused and empty, so the menu needs no bindings of its
    # own, and not having them keeps Enter free to submit a typed reply.

    class Selected(Message):
        """Posted when the user confirms a menu option."""

        def __init__(self, option: ApprovalOption) -> None:
            self.option = option
            super().__init__()

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._options: list[ApprovalOption] = []
        self._index = 0
        self._reply_preview = ""

    def compose(self):
        yield Static("", classes="approval-options", id="approval-options")
        yield Static("", classes="approval-hint", id="approval-menu-hint")

    def on_mount(self) -> None:
        self._refresh()

    def set_options(self, options: list[ApprovalOption]) -> None:
        self._options = options
        self._index = 0
        self._reply_preview = ""
        self._refresh()

    def show_menu(self) -> None:
        self.add_class("visible")
        self._index = 0
        self._refresh()

    def hide_menu(self) -> None:
        self.remove_class("visible")
        self._options = []
        self._index = 0
        self._reply_preview = ""

    def set_reply_preview(self, text: str) -> None:
        """Mirror the user's typed reply in the "No" option, so it's clear that
        typing fills in an optional reason ("No, <text>") that Enter will send.

        Moves the selection onto the reject option while composing, since Enter
        sends the reply (a reject carrying this text). No-op when the request
        has no reject option to carry a reply.
        """
        self._reply_preview = text.strip()
        if self._reply_preview:
            for index, option in enumerate(self._options):
                if option.decision.get("type") == "reject":
                    self._index = index
                    break
        self._refresh()

    @property
    def visible(self) -> bool:
        return self.has_class("visible")

    def _refresh(self) -> None:
        if not self.is_mounted:
            return
        options_widget = self.query_one("#approval-options", Static)
        hint_widget = self.query_one("#approval-menu-hint", Static)

        if not self._options:
            options_widget.update("")
            hint_widget.update("")
            return

        text = Text()
        for index, option in enumerate(self._options):
            selected = index == self._index
            prefix = "> " if selected else "  "
            style = "bold cyan" if selected else "dim"
            label = option.label
            # As the user types, the reject option mirrors it as "No, <text>" so
            # the typed reply is visibly the (optional) reason for declining.
            if option.decision.get("type") == "reject" and self._reply_preview:
                preview = self._reply_preview
                label = f"No, {preview[:59]}…" if len(preview) > 60 else f"No, {preview}"
            text.append(prefix, style=style)
            text.append(label, style=style)
            text.append("\n")
        options_widget.update(text)
        # This hint is the single home for the approval key map; the footer no
        # longer repeats it. Mention the type-to-reply path only when a reject
        # option exists to carry the typed reason.
        has_reject = any(option.decision.get("type") == "reject" for option in self._options)
        hint = "↑/↓ move · enter confirm"
        if has_reject:
            hint += " · or type a reply"
        hint_widget.update(hint)

    def action_move_up(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._refresh()

    def action_move_down(self) -> None:
        if self._index < len(self._options) - 1:
            self._index += 1
            self._refresh()

    def action_confirm(self) -> None:
        if not self._options:
            return
        self.post_message(self.Selected(self._options[self._index]))

    def action_select_reject(self) -> None:
        for option in self._options:
            if option.decision.get("type") == "reject":
                self.post_message(self.Selected(option))
                return
