"""Local Textual widgets for the jutul-agent TUI."""

from __future__ import annotations

import ast
import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from markdown_it import MarkdownIt
from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Markdown, Static

from jutul_agent.agent.tool_output import is_interrupt_payload, normalize_tool_output
from jutul_agent.approval_preview import ApprovalCard
from jutul_agent.interfaces.tui.tool_display import (
    display_path,
    display_tool_body,
    is_expandable,
    uses_compact_display,
)
from jutul_agent.juliakernel.text import render_terminal_output
from jutul_agent.paths import is_dated_session_id
from jutul_agent.preview import shorten, shorten_single_line
from jutul_agent.tool_labels import tool_label

if TYPE_CHECKING:
    from textual.timer import Timer
    from textual.widgets._markdown import MarkdownStream

_TODO_PREVIEW_ITEMS = 3
_TODO_PREVIEW_TEXT = 72
_TODO_FULL_TEXT = 140
# Cap on the live-streamed buffer re-rendered per delta, so a long solve's output
# can't make each refresh cost grow without bound.
_STREAM_RENDER_CAP = 256 * 1024
# Streamed tool output is re-rendered at most this often. Rendering runs the
# whole buffer through the terminal emulator and re-parses the markdown body,
# so doing it per delta makes a chatty tool freeze the UI.
_STREAM_REFRESH_SECONDS = 0.1


def _link_friendly_parser() -> MarkdownIt:
    """A gfm-like Markdown parser that also renders ``file://`` links.

    Textual's default parser blocks the ``file:`` scheme, so a local file link
    shows as raw ``[text](file://…)`` instead of a clickable link. Allowing it
    lets a reported path open in the OS viewer (the click handler routes
    ``file://`` to the local opener); other unsafe schemes stay blocked.
    """
    md = MarkdownIt("gfm-like")
    default_validate = md.validateLink
    md.validateLink = lambda url: url.startswith("file:") or default_validate(url)
    return md


def display_session_id(session_id: str) -> str:
    """A session id as shown in the UI: dated ids in full, UUIDs shortened."""
    return session_id if is_dated_session_id(session_id) else session_id[:8]


class MessageBlock(Vertical):
    """Generic conversation card for user, assistant, system, and error messages."""

    DEFAULT_CSS = """
    MessageBlock {
        border: solid $surface-lighten-1;
        padding: 0 1;
        margin: 1 0 0 0;
        height: auto;
        background: $surface;
    }

    MessageBlock.user {
        background: $surface-darken-1;
    }

    MessageBlock.assistant {
        background: $panel;
    }

    MessageBlock.reasoning {
        background: $surface-darken-1;
        border: solid $surface;
    }

    MessageBlock.reasoning .message-body {
        color: $text-muted;
    }

    MessageBlock.system {
        background: $surface;
    }

    MessageBlock.error {
        border: solid $error;
    }

    MessageBlock.welcome {
        background: $surface;
    }

    MessageBlock Markdown,
    MessageBlock .message-body {
        padding: 0;
        margin: 0;
        height: auto;
    }
    """

    def __init__(
        self,
        role_label: str,
        role_class: str,
        content: str = "",
        *,
        markdown: bool = False,
    ) -> None:
        super().__init__()
        self.add_class(role_class)
        self.border_title = role_label
        self._content = content
        self._markdown = markdown
        self._body_widget: Static | Markdown | None = None
        self._stream: MarkdownStream | None = None

    def compose(self):
        if self._markdown:
            yield Markdown(self._content, id="message-body", parser_factory=_link_friendly_parser)
            return
        yield Static(self._content, classes="message-body", id="message-body", markup=False)

    @property
    def content_text(self) -> str:
        """The message's raw text (markdown source for assistant replies)."""
        return self._content

    def on_mount(self) -> None:
        self._body_widget = self.query_one("#message-body")
        self.refresh_for_width()

    async def _ensure_body(self) -> None:
        if self._body_widget is not None:
            return
        if not self.is_mounted:
            await self._mounted_event.wait()
        self._body_widget = self.query_one("#message-body")

    def _get_markdown(self) -> Markdown:
        if self._body_widget is None:
            raise RuntimeError("MessageBlock body is not mounted yet")
        if not isinstance(self._body_widget, Markdown):
            raise TypeError("MessageBlock markdown stream requested for non-markdown body")
        return self._body_widget

    def _ensure_stream(self) -> MarkdownStream:
        if self._stream is None:
            self._stream = Markdown.get_stream(self._get_markdown())
        return self._stream

    def refresh_for_width(self) -> None:
        if self._body_widget is not None:
            self._body_widget.refresh(layout=True)
        self.refresh(layout=True)

    async def append_content(self, content: str) -> None:
        if not content:
            return
        await self._ensure_body()
        if not self._markdown:
            await self.set_content(self._content + content)
            return
        self._content += content
        stream = self._ensure_stream()
        await stream.write(content)

    async def stop_stream(self) -> None:
        if self._stream is not None:
            await self._stream.stop()
            self._stream = None
            await self._get_markdown().update(self._content)
            self.refresh_for_width()

    async def set_content(self, content: str) -> None:
        self._content = content
        await self._ensure_body()
        if isinstance(self._body_widget, Markdown):
            if self._stream is not None:
                await self._stream.stop()
                self._stream = None
            await self._body_widget.update(content)
        else:
            self._body_widget.update(content)
        self.refresh_for_width()


class ReasoningBlock(MessageBlock):
    """Streamed model reasoning: a rolling tail while thinking, a one-line
    summary once the answer starts.

    The full text accumulates in ``content_text`` (and lands in the trace via
    the recorder middleware); only the rendering is reduced. Showing the tail
    keeps the card a stable height while streaming, and re-rendering a few
    lines stays cheap no matter how long the model thinks. ``set_expanded``
    follows the same verbose toggle as tool cards.
    """

    _TAIL_LINES = 3
    _PREVIEW_CHARS = 100

    def __init__(self) -> None:
        super().__init__("Reasoning", "reasoning", "")
        self._started = time.monotonic()
        self._finished = False
        self._expanded = False

    async def append_content(self, content: str) -> None:
        if not content:
            return
        self._content += content
        await self._ensure_body()
        self._render_state()

    async def finish(self, *, expanded: bool = False) -> None:
        """Stop streaming: record the thinking time and collapse (or expand)."""
        if not self._finished:
            self._finished = True
            elapsed = time.monotonic() - self._started
            self.border_subtitle = f"thought for {elapsed:.0f}s"
        self._expanded = expanded
        await self._ensure_body()
        self._render_state()

    async def set_expanded(self, expanded: bool) -> None:
        if self._expanded == expanded:
            return
        self._expanded = expanded
        if self._finished:
            await self._ensure_body()
            self._render_state()

    def _render_state(self) -> None:
        if self._body_widget is None:
            return
        if not self._finished or self._expanded:
            self._body_widget.update(self._display_text())
        else:
            self._body_widget.update(self._preview_line())
        self.refresh_for_width()

    def _display_text(self) -> str:
        if self._finished:
            return self._content.strip()
        lines = self._content.strip().splitlines()
        return "\n".join(lines[-self._TAIL_LINES :])

    def _preview_line(self) -> str:
        lines = (line.strip() for line in self._content.strip().splitlines())
        first = next((line for line in lines if line), "")
        return shorten_single_line(first, self._PREVIEW_CHARS)


class WelcomeBlock(MessageBlock):
    """Landing card shown when the TUI starts or the visible log is cleared."""

    def __init__(self, *, simulator_label: str) -> None:
        super().__init__(
            "Welcome",
            "welcome",
            _render_welcome_message(simulator_label=simulator_label),
            markdown=True,
        )


class ApprovalBlock(Vertical):
    """Card for a pending approval: the tool and what it will do.

    Controls (approve / reject / reply) live in the approval menu at the prompt,
    so the request is shown once here and resolved in one place below.
    """

    DEFAULT_CSS = """
    ApprovalBlock {
        border: solid $surface-lighten-1;
        padding: 0 1;
        margin: 1 0 0 0;
        height: auto;
        background: $surface;
    }

    ApprovalBlock Markdown {
        padding: 0;
        margin: 0;
    }
    """

    def __init__(self, card: ApprovalCard) -> None:
        super().__init__()
        self._card = card
        self._body_widget: Markdown | None = None
        self.border_title = card.title

    @property
    def tool_name(self) -> str:
        return self._card.tool_name

    def compose(self):
        yield Markdown(self._card.body, id="approval-body", parser_factory=_link_friendly_parser)

    def on_mount(self) -> None:
        self._body_widget = self.query_one("#approval-body", Markdown)
        self.border_subtitle = "awaiting review"
        self.refresh_for_width()

    def refresh_for_width(self) -> None:
        if self._body_widget is not None:
            self._body_widget.refresh(layout=True)
        self.refresh(layout=True)


def _render_welcome_message(*, simulator_label: str) -> str:
    # Keep it short: the session id, model, and approval mode are all in the status
    # bar, the prompt placeholder already invites a question, and the footer lists
    # the keys. The card only adds the two non-obvious controls and points at /help.
    return (
        f"**jutul-agent** is ready for **{simulator_label}**.\n"
        "**Ctrl+C** interrupts a turn (twice to exit) · `/help` for commands."
    )


class PromptGuide(Static):
    """Dynamic helper line under the prompt for commands, history, and activity."""

    DEFAULT_CSS = """
    PromptGuide {
        height: auto;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id, markup=False)
        self._message = ""
        self._activity = "ready"

    @property
    def message(self) -> str:
        return self._message

    def set_message(self, message: str) -> None:
        self._message = message
        self.refresh()

    def set_activity(self, activity: str) -> None:
        self._activity = activity
        self.refresh()

    def render(self) -> Text:
        text = Text()
        text.append(self._activity, style=_status_style(self._activity))
        if self._message:
            text.append(" · ", style="dim")
            text.append(self._message, style="dim")
        return text


class StatusBar(Static):
    """Compact bottom status bar with session context and live state."""

    DEFAULT_CSS = """
    StatusBar {
        dock: top;
        height: 1;
        padding: 0 1;
        background: $surface-darken-1;
        color: $text-muted;
    }
    """

    def __init__(
        self,
        *,
        simulator_label: str,
        session_id: str,
        model_label: str | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, markup=False)
        self._simulator_label = simulator_label
        self._session_id = session_id
        self._model_label = model_label
        self._pending_count = 0
        self._tool_toggle_available = False
        self._tools_expanded = False
        self._approval_mode_label = "default"
        self._context_label: str | None = None
        self._context_alert = "ok"
        self._warming = False

    def set_state(
        self,
        *,
        pending_count: int = 0,
        tool_toggle_available: bool = False,
        tools_expanded: bool = False,
        approval_mode_label: str = "default",
    ) -> None:
        self._pending_count = pending_count
        self._tool_toggle_available = tool_toggle_available
        self._tools_expanded = tools_expanded
        self._approval_mode_label = approval_mode_label
        self.refresh()

    def set_model(self, model_label: str | None) -> None:
        self._model_label = model_label
        self.refresh()

    def set_context(self, label: str | None, *, alert: str = "ok") -> None:
        self._context_label = label
        self._context_alert = alert
        self.refresh()

    def set_warming(self, warming: bool) -> None:
        """Show or clear the Julia warm-up chip.

        It lives here, not in the bottom activity line, so running a turn or
        a pending approval never replaces it while the kernel still compiles.
        """
        self._warming = warming
        self.refresh()

    def render(self) -> Text:
        text = Text()
        text.append("jutul-agent", style="bold")
        text.append(" · ", style="dim")
        text.append(self._simulator_label, style="bold cyan")

        if self._model_label:
            text.append(" · ", style="dim")
            text.append(shorten(self._model_label, 28), style="dim")

        text.append(" · ", style="dim")
        text.append(display_session_id(self._session_id), style="dim")

        if self._approval_mode_label:
            text.append(" · ", style="dim")
            text.append(self._approval_mode_label, style="cyan")

        if self._context_label:
            style = {"warn": "yellow", "high": "bold red"}.get(self._context_alert, "dim")
            text.append(" · ", style="dim")
            text.append(self._context_label, style=style)

        if self._warming:
            text.append(" · ", style="dim")
            text.append("warming Julia", style="yellow")

        if self._pending_count:
            label = "approval" if self._pending_count == 1 else "approvals"
            text.append(" · ", style="dim")
            text.append(f"{self._pending_count} {label}", style="yellow")

        # Live state (turn status, warm-up, Ctrl+G cancel) lives in the bottom
        # bar next to the input; see ``_activity_label`` / the prompt guide.
        if self._tools_expanded:
            text.append(" · ", style="dim")
            text.append("verbose", style="cyan")
        elif self._tool_toggle_available:
            text.append(" · ", style="dim")
            text.append("details available", style="dim")

        return text


class ToolBlock(Vertical):
    """Richer single-card rendering for a tool call and its eventual result."""

    DEFAULT_CSS = """
    ToolBlock {
        border: solid $surface-lighten-1;
        padding: 0 1;
        margin: 1 0 0 0;
        height: auto;
        background: $surface;
    }

    ToolBlock.approval {
        border: solid $warning;
    }

    ToolBlock.rejected,
    ToolBlock.cancelled,
    ToolBlock.error {
        border: solid $error;
    }

    ToolBlock.responded {
        border: solid $accent;
    }

    ToolBlock.compact {
        border: none;
        /* A thin solid rail (│) rather than a block glyph (▊): the block
           renders with gaps in some terminals/fonts over SSH, and the thin
           line matches the full card's border. */
        border-left: solid $surface-lighten-1;
        margin: 0;
        padding: 0 1;
        background: transparent;
    }

    ToolBlock.compact Markdown {
        color: $text-muted;
    }

    ToolBlock Markdown {
        padding: 0;
        margin: 0;
        /* File-path links sit inside muted summaries; keep them the same
           muted tone (an underline still marks them clickable) so a card
           reads as one calm line instead of a bright link in grey text. */
        link-color: $text-muted;
        link-style: underline;
        link-color-hover: $accent;
        link-style-hover: underline;
    }
    """

    def __init__(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        tool_call_id: str | None = None,
        expanded: bool = False,
    ) -> None:
        super().__init__()
        self._tool_name = tool_name
        self._args = args or {}
        self._tool_call_id = tool_call_id
        self._status = "running"
        self._output = ""
        self._streamed = ""
        self._expanded = expanded
        self._is_error = False
        self._reject_reason: str | None = None
        self._body_widget: Markdown | None = None
        self._elapsed_task: asyncio.Task[None] | None = None
        self._stream_refresh_timer: Timer | None = None
        self.border_title = tool_label(tool_name)
        self.border_subtitle = _status_text(self._status, self._reject_reason)

    def compose(self):
        yield Markdown("", id="tool-body", parser_factory=_link_friendly_parser)

    async def on_mount(self) -> None:
        self._body_widget = self.query_one("#tool-body", Markdown)
        await self._refresh()

    async def _ensure_body(self) -> None:
        if self._body_widget is not None:
            return
        if not self.is_mounted:
            await self._mounted_event.wait()
        self._body_widget = self.query_one("#tool-body", Markdown)

    def start_elapsed_timer(self) -> None:
        """Update the subtitle with elapsed seconds while the tool is running."""

        self.stop_elapsed_timer()

        async def _tick() -> None:
            start = time.monotonic()
            while self._status == "running":
                elapsed = time.monotonic() - start
                self.border_subtitle = f"running · {elapsed:.0f}s"
                self.refresh()
                await asyncio.sleep(1.0)

        self._elapsed_task = asyncio.create_task(_tick())

    def stop_elapsed_timer(self) -> None:
        if self._elapsed_task is not None:
            self._elapsed_task.cancel()
            self._elapsed_task = None

    def refresh_for_width(self) -> None:
        if self._body_widget is not None:
            self._body_widget.refresh(layout=True)
        self.refresh(layout=True)

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def tool_call_id(self) -> str | None:
        return self._tool_call_id

    @property
    def expandable(self) -> bool:
        if not self._output:
            return False
        return is_expandable(self._output, tool_name=self._tool_name)

    @property
    def has_output(self) -> bool:
        return bool(self._output or self._streamed)

    @property
    def status(self) -> str:
        return self._status

    async def set_running(self) -> None:
        self._status = "running"
        self._reject_reason = None
        await self._refresh()

    async def set_pending_approval(self) -> None:
        self._status = "approval"
        await self._refresh()

    async def set_rejected(self, reason: str | None = None) -> None:
        self.stop_elapsed_timer()
        self._stop_stream_refresh()
        self._status = "rejected"
        self._reject_reason = reason or None
        await self._refresh()

    async def set_responded(self, message: str | None = None) -> None:
        """The user answered instead of running the tool."""
        self.stop_elapsed_timer()
        self._stop_stream_refresh()
        self._status = "responded"
        self._reject_reason = message or None
        await self._refresh()

    async def set_cancelled(self, reason: str | None = None) -> None:
        self.stop_elapsed_timer()
        self._stop_stream_refresh()
        self._status = "cancelled"
        self._reject_reason = reason or None
        await self._refresh()

    async def append_output(self, delta: str) -> None:
        if not delta:
            return
        # Keep only the tail so each re-render stays bounded. Rendering tolerates a
        # truncated escape at the cut; it sits far up in scrollback anyway.
        self._streamed = (self._streamed + delta)[-_STREAM_RENDER_CAP:]
        self._status = "running"
        await self._ensure_body()
        if not self.is_running:
            # No message pump (plain widget under unit test): render in line.
            await self._render_streamed()
            return
        if self._stream_refresh_timer is None:
            self._stream_refresh_timer = self.set_timer(
                _STREAM_REFRESH_SECONDS, self._flush_streamed
            )

    async def _render_streamed(self) -> None:
        # Render the raw buffer to its on-screen state so carriage returns and cursor
        # moves (ProgressMeter/Jutul bars) collapse to a single updating line,
        # matching a real terminal and the final EvalResult.output.
        self._output = render_terminal_output(self._streamed)
        await self._refresh()

    async def _flush_streamed(self) -> None:
        self._stream_refresh_timer = None
        if self._status != "running" or not self._streamed:
            return  # a final result landed first and already rendered
        await self._render_streamed()

    def _stop_stream_refresh(self) -> None:
        if self._stream_refresh_timer is not None:
            self._stream_refresh_timer.stop()
            self._stream_refresh_timer = None

    async def set_result(self, content: str, *, is_error: bool = False) -> None:
        self.stop_elapsed_timer()
        self._stop_stream_refresh()
        self._output = normalize_tool_output(content)
        self._streamed = ""
        self._is_error = is_error
        if is_error and is_interrupt_payload(self._output):
            self._status = "approval"
        else:
            self._status = "error" if is_error else "success"
        await self._refresh()

    async def set_expanded(self, expanded: bool) -> None:
        """Show full output (verbose) or the compact per-tool preview."""
        if self._expanded == expanded:
            return
        self._expanded = expanded
        await self._refresh()

    async def _refresh(self) -> None:
        if self._body_widget is None:
            return

        # Only the bordered statuses get a CSS class; the others share the
        # default ``ToolBlock`` styling and don't need a marker.
        for css_class in ("approval", "rejected", "cancelled", "error", "responded", "compact"):
            self.remove_class(css_class)
        if self._status in {"approval", "rejected", "cancelled", "error", "responded"}:
            self.add_class(self._status)
        elif (
            self._status == "success"
            and uses_compact_display(
                self._tool_name,
                is_error=self._is_error,
                output=self._output,
            )
            and not self._expanded
        ):
            self.add_class("compact")
        self.border_subtitle = _status_text(self._status, self._reject_reason)

        body = _render_tool_body(
            self._tool_name,
            self._args,
            output=self._output,
            expanded=self._expanded,
            reject_reason=self._reject_reason,
            is_error=self._is_error,
        )
        await self._body_widget.update(body)
        self.refresh_for_width()


def _status_style(message: str) -> str:
    lowered = message.lower()
    if "approval" in lowered:
        return "bold yellow"
    if (
        "thinking" in lowered
        or "running" in lowered
        or "resuming" in lowered
        or "warming" in lowered
    ):
        return "yellow"
    if "error" in lowered or "failed" in lowered:
        return "red"
    return "green"


def _status_text(status: str, reject_reason: str | None) -> str:
    if status == "running":
        return "running"
    if status == "success":
        return "completed"
    if status == "approval":
        return "waiting for approval"
    if status == "rejected":
        if reject_reason:
            return f"rejected: {reject_reason}"
        return "rejected"
    if status == "cancelled":
        if reject_reason:
            return f"cancelled: {reject_reason}"
        return "cancelled"
    if status == "responded":
        if reject_reason:
            return f"answered by user: {shorten_single_line(reject_reason, 72)}"
        return "answered by user"
    if status == "error":
        return "tool error"
    return status


def _render_tool_body(
    tool_name: str,
    args: dict[str, Any],
    *,
    output: str,
    expanded: bool,
    reject_reason: str | None,
    is_error: bool = False,
) -> str:
    if output:
        return _render_tool_output(
            tool_name,
            output,
            expanded=expanded,
            args=args,
            is_error=is_error,
        )

    # Some tools render a running body before any output: julia_* show the Code
    # section, write/edit show the target file and a growing character count.
    if tool_name in {"run_julia", "plot_julia", "write_file", "edit_file"}:
        body = display_tool_body(
            tool_name,
            args,
            output=output,
            expanded=expanded,
            is_error=is_error,
        )
        if reject_reason:
            body += f"\n\n> Reject reason: {reject_reason}"
        return body

    body = _render_request_preview(tool_name, args)
    if reject_reason:
        body += f"\n\n> Reject reason: {reject_reason}"
    return body


def _render_tool_output(
    tool_name: str,
    output: str,
    *,
    expanded: bool,
    args: dict[str, Any] | None = None,
    is_error: bool = False,
) -> str:
    if tool_name == "write_todos":
        rendered = _render_todo_output(output, expanded=expanded)
        if rendered is not None:
            return rendered

    return display_tool_body(
        tool_name,
        args or {},
        output=output,
        expanded=expanded,
        is_error=is_error,
    )


def _render_request_preview(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "write_todos":
        rendered = _render_todo_request(args)
        if rendered is not None:
            return rendered
    return f"_{_format_tool_display(tool_name, args)}_"


def _format_tool_display(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name in {"write_file", "edit_file", "read_file"}:
        path_value = args.get("path") or args.get("file_path")
        if path_value is not None:
            path = display_path(str(path_value))
            return f"{tool_name}({path})"

    if tool_name == "execute" and args.get("command"):
        command = shorten_single_line(str(args["command"]), 80)
        return f'execute("{command}")'

    if tool_name == "run_julia" and args.get("code"):
        code = shorten_single_line(str(args["code"]), 80)
        return f'run_julia("{code}")'

    if tool_name == "write_todos":
        todos = args.get("todos")
        if isinstance(todos, list):
            count = len(todos)
            label = "item" if count == 1 else "items"
            return f"write_todos({count} {label})"

    if not args:
        return f"{tool_name}()"

    rendered_args = ", ".join(
        f"{key}={shorten_single_line(str(value), 30)}" for key, value in list(args.items())[:3]
    )
    if len(args) > 3:
        rendered_args += ", ..."
    return f"{tool_name}({rendered_args})"


def _render_todo_request(args: dict[str, Any]) -> str | None:
    items = _parse_todo_items(args.get("todos"))
    if items is None:
        return None
    return _render_todo_lines(items, preview=True, label="plan")


def _render_todo_output(output: str, *, expanded: bool) -> str | None:
    items = _parse_todo_items(output)
    if items is None:
        return None
    return _render_todo_lines(items, preview=not expanded, label="plan updated")


def _render_todo_lines(items: list[Any], *, preview: bool, label: str) -> str:
    total = len(items)
    summary = _summarize_todos(items)
    limit = _TODO_PREVIEW_ITEMS if preview else total

    lines = [f"_{label} · {summary}_", ""]
    for item in items[:limit]:
        marker = _todo_marker(item)
        text = _todo_text(item)
        short = shorten_single_line(text, _TODO_PREVIEW_TEXT if preview else _TODO_FULL_TEXT)
        lines.append(f"- [{marker}] {short}")

    hidden = total - limit
    if hidden > 0:
        item_label = "item" if hidden == 1 else "items"
        lines.append(f"- ... {hidden} more {item_label}")
    return "\n".join(lines)


def _parse_todo_items(value: Any) -> list[Any] | None:
    if isinstance(value, list):
        return value

    if isinstance(value, dict):
        todos = value.get("todos")
        if isinstance(todos, list):
            return todos
        return None

    if not isinstance(value, str):
        return None

    candidate = value.strip()
    if not candidate:
        return []

    for raw_candidate in (candidate, _extract_serialized_collection(candidate)):
        if raw_candidate is None:
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw_candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and isinstance(parsed.get("todos"), list):
                return parsed["todos"]
    return None


def _extract_serialized_collection(text: str) -> str | None:
    starts = [index for index in (text.find("["), text.find("{")) if index >= 0]
    if not starts:
        return None

    start = min(starts)
    opener = text[start]
    closer = "]" if opener == "[" else "}"
    end = text.rfind(closer)
    if end <= start:
        return None
    return text[start : end + 1]


def _summarize_todos(items: list[Any]) -> str:
    completed = sum(
        1 for item in items if isinstance(item, dict) and item.get("status") == "completed"
    )
    active = sum(
        1 for item in items if isinstance(item, dict) and item.get("status") == "in_progress"
    )
    pending = max(0, len(items) - completed - active)
    parts = [f"{len(items)} items"]
    if active:
        parts.append(f"{active} active")
    if pending:
        parts.append(f"{pending} pending")
    if completed:
        parts.append(f"{completed} done")
    return " · ".join(parts)


def _todo_marker(item: Any) -> str:
    status = item.get("status") if isinstance(item, dict) else None
    if status == "completed":
        return "x"
    if status == "in_progress":
        return "~"
    return " "


def _todo_text(item: Any) -> str:
    if isinstance(item, dict):
        content = item.get("content")
        if isinstance(content, str):
            return content
    return str(item)
