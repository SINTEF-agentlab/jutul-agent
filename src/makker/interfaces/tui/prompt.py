"""Multi-line prompt input for the makker TUI."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import ClassVar

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class PromptTextArea(TextArea):
    """Chat-style prompt: Enter submits, Shift+Enter inserts a newline."""

    DEFAULT_CSS = """
    PromptTextArea {
        height: auto;
        min-height: 1;
        max-height: 8;
        width: 1fr;
        border: none;
        background: transparent;
        padding: 0;
        overflow-y: auto;
    }

    PromptTextArea:focus {
        border: none;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding(
            "shift+enter,ctrl+enter,alt+enter",
            "insert_newline",
            "New Line",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+up,ctrl+arrow_up,ctrl+p",
            "history_previous",
            "Previous Input",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+down,ctrl+arrow_down,ctrl+n",
            "history_next",
            "Next Input",
            show=False,
            priority=True,
        ),
    ]

    _NEWLINE_KEYS: ClassVar[frozenset[str]] = frozenset(
        key
        for binding in BINDINGS
        if binding.action == "insert_newline"
        for key in binding.key.split(",")
    )

    _HISTORY_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "ctrl+up",
            "ctrl+down",
            "ctrl+arrow_up",
            "ctrl+arrow_down",
            "ctrl+p",
            "ctrl+n",
        }
    )

    class Submitted(Message):
        """Posted when the user submits the prompt."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    class HistoryPrevious(Message):
        """Request the previous history entry."""

        def __init__(self, current_text: str) -> None:
            self.current_text = current_text
            super().__init__()

    class HistoryNext(Message):
        """Request the next history entry."""

    def __init__(self, *, placeholder: str = "", id: str | None = None) -> None:
        super().__init__(id=id, soft_wrap=True, show_line_numbers=False)
        self.placeholder = placeholder
        self._in_history = False
        self._skip_changed_events = 0
        self._approval_nav_handler: Callable[[str], bool | Awaitable[bool]] | None = None

    def set_approval_nav_handler(
        self,
        handler: Callable[[str], bool | Awaitable[bool]] | None,
    ) -> None:
        self._approval_nav_handler = handler

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, text: str) -> None:
        self.load_text(text)
        lines = text.split("\n")
        last_row = max(0, len(lines) - 1)
        self.move_cursor((last_row, len(lines[last_row])))

    @property
    def disabled(self) -> bool:
        return self.read_only

    @disabled.setter
    def disabled(self, value: bool) -> None:
        self.read_only = value

    @property
    def cursor_position(self) -> int:
        row, col = self.cursor_location
        lines = self.text.split("\n")
        return sum(len(line) + 1 for line in lines[:row]) + col

    @cursor_position.setter
    def cursor_position(self, index: int) -> None:
        index = max(0, min(index, len(self.text)))
        consumed = 0
        for row, line in enumerate(self.text.split("\n")):
            line_len = len(line)
            if consumed + line_len >= index:
                self.move_cursor((row, index - consumed))
                return
            consumed += line_len + 1
        lines = self.text.split("\n")
        last_row = max(0, len(lines) - 1)
        self.move_cursor((last_row, len(lines[last_row])))

    def suppress_next_changed(self) -> None:
        """Skip the next Changed event after a programmatic text replacement."""
        self._skip_changed_events += 1

    def consume_changed_suppression(self) -> bool:
        if self._skip_changed_events <= 0:
            return False
        self._skip_changed_events -= 1
        return True

    def action_insert_newline(self) -> None:
        self.insert("\n")

    def clear(self) -> None:
        """Reset prompt text and scroll position after submit."""
        self.load_text("")
        self.scroll_home(animate=False)

    def action_history_previous(self) -> None:
        self._in_history = True
        self.post_message(self.HistoryPrevious(self.text))

    def action_history_next(self) -> None:
        self._in_history = True
        self.post_message(self.HistoryNext())

    async def _on_key(self, event: events.Key) -> None:
        if self._approval_nav_handler is not None and not self.text:
            nav_keys = {"up", "down", "enter", "escape"}
            if event.key in nav_keys:
                result = self._approval_nav_handler(event.key)
                if inspect.isawaitable(result):
                    handled = await result
                else:
                    handled = bool(result)
                if handled:
                    event.prevent_default()
                    event.stop()
                    return

        if event.key in self._HISTORY_KEYS:
            event.prevent_default()
            event.stop()
            if event.key in {"ctrl+up", "ctrl+arrow_up", "ctrl+p"}:
                self.action_history_previous()
            else:
                self.action_history_next()
            return

        if event.key in self._NEWLINE_KEYS:
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return

        if event.key == "enter":
            event.prevent_default()
            event.stop()
            value = self.text.strip()
            if value:
                self._in_history = False
                self.post_message(self.Submitted(value))
            return

        await super()._on_key(event)
