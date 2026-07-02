"""Interactive TUI for jutul-agent built on Textual.

The interface keeps the runtime seam in ``TurnRunner`` and focuses on local
presentation: a scrollable conversation log, grouped tool cards, inline
approval cards, and a compact input bar.

Slash commands typed into the input box:

- ``/transcript``: render the current session's trace to HTML on disk.
- ``/transcript md``: render the transcript as markdown instead.
- ``/add-dir <path>``: add an extra folder so the agent can read/edit it.
- ``/model [provider:model]``: open the model selector, or switch directly.
- ``/context``: show how much of the model's context window is used.
- ``/compact``: summarize older turns to free context space.
- ``/memory [note | edit [note]]``: view workspace memory, or open it in $EDITOR.
- ``/clear``: clear the visible log and restore the welcome card.
- ``/approve``: approve the currently pending tool actions.
- ``/reject [reason]``: reject the currently pending tool actions.
- ``/respond <message>``: answer on behalf of the pending tool actions.
- ``/approval-mode [ask|workspace|auto]``: cycle or set permission mode (Shift+Tab cycles).
- ``/quit``: quit the app (same as Ctrl+D).
- ``/help``: list available commands.
"""

from __future__ import annotations

import asyncio
import contextlib
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Markdown, Static

from jutul_agent.agent.added_dirs import AddDirError, add_dir, added_dirs
from jutul_agent.agent.approval import (
    ApprovalMode,
    ToolAllowlist,
    build_resume_payload,
    parse_approval_mode,
    pending_allowed_decisions,
    should_auto_approve_interrupt,
)
from jutul_agent.agent.turns import (
    TurnInterrupt,
    TurnReasoningDelta,
    TurnRunResult,
    TurnTextDelta,
    TurnTextEnd,
    TurnToolEvent,
    usage_from_messages,
)
from jutul_agent.approval_preview import render_interrupt_cards
from jutul_agent.context_panel import (
    context_component_estimates,
    render_context_panel,
    status_label,
    usage_alert,
)
from jutul_agent.interfaces.tui.approval_menu import ApprovalMenu, build_approval_options
from jutul_agent.interfaces.tui.commands import (
    ALL_COMMANDS,
    InputHistory,
    SlashCommandSpec,
    find_command,
    matching_specs,
)
from jutul_agent.interfaces.tui.model_menu import (
    ApiKeyModal,
    ModelChoice,
    ModelMenu,
    OllamaPullModal,
)
from jutul_agent.interfaces.tui.prompt import PromptTextArea
from jutul_agent.interfaces.tui.widgets import (
    ApprovalBlock,
    MessageBlock,
    PromptGuide,
    ReasoningBlock,
    StatusBar,
    ToolBlock,
    WelcomeBlock,
    display_session_id,
)
from jutul_agent.open_file import open_path
from jutul_agent.paths import workspace_root
from jutul_agent.recent_models import record_recent_model
from jutul_agent.session import Session
from jutul_agent.session_host import SessionHost
from jutul_agent.trace import TraceLog
from jutul_agent.transcript import render_html, render_markdown

if TYPE_CHECKING:
    from collections.abc import Callable

    from textual.timer import Timer

_RESIZE_DEBOUNCE_SECONDS = 0.05
# Most prior messages replayed into the log when resuming a session.
_REPLAY_MAX_MESSAGES = 40


def _resolve_editor() -> str | None:
    """The editor command for ``/memory edit``: ``$VISUAL``/``$EDITOR``, else
    a platform default, so the command works out of the box."""
    import os
    import shutil
    import sys

    for var in ("VISUAL", "EDITOR"):
        value = os.environ.get(var)
        if value:
            return value
    if sys.platform == "win32":
        return "notepad"
    return next((name for name in ("nano", "vim", "vi") if shutil.which(name)), None)


@dataclass
class _AssistantStream:
    """In-flight prose + reasoning blocks for one streamed assistant turn.

    Owns the two MessageBlocks the streaming path mounts so the call sites
    don't have to track them by hand. Call ``flush`` between
    streaming-eligible regions (e.g. before a tool call is mounted, after a
    ``message-finish`` event) to close the markdown stream and let a fresh
    block be mounted on the next chunk.

    Only genuine assistant text reaches here: the turn runner streams
    text/reasoning from the model node alone, so tool results and
    middleware-injected messages never arrive as prose (no content filtering
    is needed; see ``jutul_agent.agent.turns``).
    """

    prose: MessageBlock | None = None
    reasoning: ReasoningBlock | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def append_prose(self, log: VerticalScroll, text: str) -> None:
        if not text:
            return
        async with self._lock:
            block = self.prose
            if block is None:
                block = MessageBlock("Assistant", "assistant", "", markdown=True)
                self.prose = block
                await log.mount(block)
            elif not block.is_mounted:
                await block._mounted_event.wait()
            await block.append_content(text)

    async def append_reasoning(self, log: VerticalScroll, text: str) -> None:
        if not text:
            return
        async with self._lock:
            block = self.reasoning
            if block is None:
                block = ReasoningBlock()
                self.reasoning = block
                # Reasoning belongs above the answer it precedes. Text and
                # reasoning are drained concurrently, so a stray text delta can
                # mount the prose block first; anchor the reasoning above it
                # rather than letting arrival order decide.
                if self.prose is not None and self.prose.is_mounted:
                    await log.mount(block, before=self.prose)
                else:
                    await log.mount(block)
            elif not block.is_mounted:
                await block._mounted_event.wait()
            await block.append_content(text)

    async def flush(self, *, reasoning_expanded: bool = False) -> None:
        async with self._lock:
            if self.prose is not None:
                await self.prose.stop_stream()
            if self.reasoning is not None:
                await self.reasoning.finish(expanded=reasoning_expanded)
            self.prose = None
            self.reasoning = None


class TUIApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }
    #log {
        height: 1fr;
        padding: 0 1 0 1;
        background: $background;
        scrollbar-gutter: stable;
    }
    #input-panel {
        dock: bottom;
        height: auto;
        padding: 0 1 1 1;
        background: $background;
        border-top: solid $surface-lighten-1;
    }
    #input-row {
        height: auto;
        max-height: 10;
        border: solid $surface-lighten-1;
        padding: 0 1;
        background: $surface;
    }
    #prompt-glyph {
        width: 2;
        content-align: left top;
        color: $text-muted;
        padding-top: 0;
    }
    #prompt {
        width: 1fr;
        height: auto;
        min-height: 1;
        max-height: 8;
    }
    #prompt-help {
        height: auto;
        color: $text-muted;
        padding: 0 1;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "interrupt", "Interrupt / Copy / Quit", priority=True),
        Binding("ctrl+g", "cancel_turn", "Cancel", priority=True),
        Binding("ctrl+l", "clear_visible_log", "Clear Log", priority=True),
        Binding("ctrl+o", "toggle_tool_output", "Toggle Tool Output", priority=True),
        Binding("tab", "complete_prompt", "Complete Prompt", show=False, priority=True),
        Binding(
            "shift+tab",
            "cycle_approval_mode",
            "Cycle Permission Mode",
            show=False,
            priority=True,
        ),
    ]

    def __init__(
        self,
        *,
        agent: Any,
        session: Session,
        backend: Any | None = None,
        model_label: str | None = None,
        approval_mode: ApprovalMode | str | None = None,
        warmup_task: Any | None = None,
        agent_factory: Callable[[str, Any], tuple[Any, Any]] | None = None,
        host: SessionHost | None = None,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._session = session
        self._backend = backend
        self._model_label = model_label
        self._agent_factory = agent_factory
        self._warmup_task = warmup_task
        self._warming = warmup_task is not None and not warmup_task.done()
        self._approval_mode = (
            approval_mode
            if isinstance(approval_mode, ApprovalMode)
            else parse_approval_mode(approval_mode)
        )
        self._tool_allowlist = ToolAllowlist()
        # All turn driving goes through the host (the shared policy loop and
        # settle hooks). A caller that only has the raw pieces (tests) gets a
        # host wrapped around them.
        self._host = host or SessionHost(
            session=session,
            agent=agent,
            backend=backend,
            model=model_label,
            approval_mode=str(self._approval_mode),
            surface="tui",
        )
        # Held so the fire-and-forget titling task isn't garbage-collected mid-run.
        self._title_task: Any | None = None
        self._turn_worker: Any = None
        self._cancel_requested = False
        self._cancel_rendered = False
        self._julia_running_on_cancel = False
        self._pending_interrupts: list[TurnInterrupt] = []
        self._tool_blocks: list[ToolBlock] = []
        self._tools_expanded = False
        self._active_approval_blocks: list[ApprovalBlock] = []
        # The transient "approval is pending" log line, cleared on decision so
        # it never lingers as a stale instruction after the turn moves on.
        self._pending_approval_note: MessageBlock | None = None
        self._history = InputHistory()
        self._setting_prompt_value = False
        self._busy = False
        self._status_text = "ready"
        self._stream = _AssistantStream()
        # Whether the current turn has rendered any assistant prose; if not, the
        # turn end mounts a placeholder so an empty answer is not silent.
        self._assistant_text_rendered = False
        # Context accounting: usage_metadata from the newest/first model call
        # and the resolved context window (fetched in a background worker).
        self._last_usage: dict[str, Any] | None = None
        self._first_usage: dict[str, Any] | None = None
        self._model_calls = 0
        self._context_window_tokens: int | None = None
        self._resize_timer: Timer | None = None
        self._quit_armed = False
        self._quit_timer: Timer | None = None
        # Resolved in ``on_mount``; both widgets are always present.
        self._log: VerticalScroll = None  # type: ignore[assignment]
        self._prompt: PromptTextArea = None  # type: ignore[assignment]
        self._approval_menu: ApprovalMenu = None  # type: ignore[assignment]

    def compose(self) -> ComposeResult:
        yield StatusBar(
            simulator_label=self._session.simulator.display_name,
            session_id=self._session.session_id,
            model_label=self._model_label,
            id="status",
        )
        with VerticalScroll(id="log"):
            yield WelcomeBlock(simulator_label=self._session.simulator.display_name)
        with Vertical(id="input-panel"):
            yield ApprovalMenu(id="approval-menu")
            with Horizontal(id="input-row"):
                yield Static(">", id="prompt-glyph", markup=False)
                yield PromptTextArea(
                    placeholder="Ask a question or describe a task",
                    id="prompt",
                )
            yield PromptGuide(id="prompt-guide")

    def on_mount(self) -> None:
        self.title = "jutul-agent"
        self._refresh_subtitle()
        self._log = self.query_one("#log", VerticalScroll)
        # Streamed content grows the log continuously; an anchored container
        # follows it only while the user is at the bottom. Scrolling up releases
        # the anchor (reading is never interrupted) and returning to the bottom
        # re-engages it.
        self._log.anchor()
        self._prompt = self.query_one("#prompt", PromptTextArea)
        self._approval_menu = self.query_one("#approval-menu", ApprovalMenu)
        self._prompt.set_approval_nav_handler(self._handle_approval_nav)
        self._set_status("ready")
        self._prompt.focus()
        self._refresh_prompt_guide()
        if self._model_label:
            record_recent_model(self._model_label)
        if self._session.resumed:
            self._load_usage_from_trace()
            self.run_worker(self._replay_history(), name="replay")
            self.run_worker(self._resync_pending(), name="pending-resync")
        if self._warming:
            self.query_one("#status", StatusBar).set_warming(True)
            self.run_worker(self._watch_warmup(), name="warmup-watch")
        self.run_worker(self._fetch_context_window(), name="ctx-window")

    async def _resync_pending(self) -> None:
        """Re-surface an approval the session was paused on when it was left.

        A turn that pauses on an approval persists the interrupt in the graph
        state. Without this, resuming that session would orphan the paused
        turn: nothing would show, and a new prompt would be refused.
        """
        try:
            pending = await self._host.pending_interrupts()
        except Exception:
            return
        if not pending or self._busy or self._pending_interrupts:
            return
        self._pending_interrupts = list(pending)
        await self._render_interrupts(self._pending_interrupts)
        self._set_status("approval required")
        self._refresh_prompt_guide()

    async def _watch_warmup(self) -> None:
        """Clear the 'warming up' indicator once the background warm-up ends.

        ``asyncio.shield`` keeps the warm-up running if this watcher is torn
        down; ``run.py`` owns the task's lifecycle and cancels it on exit.
        """
        task = self._warmup_task
        if task is None:
            return
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.shield(task)
        self._warming = False
        if self.is_mounted:
            self.query_one("#status", StatusBar).set_warming(False)

    async def _fetch_context_window(self) -> None:
        """Resolve the active model's context window off the event loop."""
        if not self._model_label:
            return
        from jutul_agent.models import context_window

        label = self._model_label
        window = await asyncio.to_thread(context_window, label)
        if label != self._model_label:
            return  # switched while resolving; the new switch refetches
        self._context_window_tokens = window
        self._refresh_context_status()

    def _load_usage_from_trace(self) -> None:
        """Seed context figures from the trace (a resumed session's history)."""
        usages = [
            event.payload
            for event in self._session.trace.iter_events()
            if event.kind == "model_usage"
        ]
        if not usages:
            return
        self._model_calls = len(usages)
        self._first_usage = usages[0]
        self._last_usage = usages[-1]
        self._refresh_context_status()

    def _update_usage_from_messages(self, messages: list[Any]) -> None:
        usages = usage_from_messages(messages)
        if not usages:
            return
        self._model_calls = len(usages)
        self._first_usage = usages[0]
        self._last_usage = usages[-1]  # measured: clears any estimate flag
        self._refresh_context_status()

    def _apply_compaction_estimate(self, freed_tokens: int) -> None:
        """Drop the displayed usage by the tokens compaction freed.

        Compaction runs no model call, so the measured usage would otherwise
        sit stale until the next reply. This reduces the last measured input
        by the freed estimate (the fixed overhead is unchanged, so it
        cancels) and flags the figure estimated; the next real turn replaces
        it with a measurement.
        """
        if not self._last_usage or freed_tokens <= 0:
            return
        held = int(self._last_usage.get("input_tokens") or 0)
        self._last_usage = {
            "input_tokens": max(0, held - freed_tokens),
            "output_tokens": 0,
            "estimated": True,
        }
        self._refresh_context_status()

    def _refresh_context_status(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#status", StatusBar).set_context(
            status_label(self._last_usage, self._context_window_tokens),
            alert=usage_alert(self._last_usage, self._context_window_tokens),
        )

    def _memory_dir(self) -> Path:
        from jutul_agent.agent.memory import ensure_memory_dir
        from jutul_agent.paths import workspace_memory_dir

        return ensure_memory_dir(self._session.memory_dir(workspace_memory=workspace_memory_dir()))

    async def _command_memory(self, arg: str) -> None:
        """View workspace memory, one note, or open it in $EDITOR."""
        from jutul_agent.agent.memory import (
            memory_note_path,
            render_memory_overview,
        )

        memory_dir = self._memory_dir()
        head, _, rest = arg.partition(" ")
        head = head.strip()

        if head == "edit":
            await self._memory_edit(memory_dir, rest.strip())
            return
        if not head:
            body = render_memory_overview(memory_dir)
            await self._log.mount(MessageBlock("Memory", "system", body, markdown=True))
            self._jump_to_latest()
            return

        target = memory_note_path(memory_dir, head)
        if not target.exists():
            await self._note(f"no memory note named `{target.name}`. /memory lists them.")
            return
        body = target.read_text(encoding="utf-8")
        await self._log.mount(MessageBlock(target.name, "system", body, markdown=True))
        self._jump_to_latest()

    async def _memory_edit(self, memory_dir: Path, name: str) -> None:
        """Suspend the TUI and open a memory file in the user's editor."""
        import shlex
        import subprocess

        from jutul_agent.agent.memory import MEMORY_INDEX_FILENAME, memory_note_path

        target = memory_note_path(memory_dir, name) if name else memory_dir / MEMORY_INDEX_FILENAME
        if not target.exists():
            await self._note(f"no memory note named `{target.name}`. /memory lists them.")
            return
        editor = _resolve_editor()
        if not editor:
            await self._note(f"no editor found. Set $EDITOR, or open the file directly: {target}")
            return
        with self.suspend():
            subprocess.run([*shlex.split(editor), str(target)], check=False)
        await self._note(f"edited `{target.name}`; the agent sees it from its next turn.")

    async def _command_compact(self, _arg: str) -> None:
        """Summarize older turns of the live thread on request."""
        if self._busy or self._pending_interrupts:
            await self._note("finish or cancel the current turn before compacting.")
            return
        if not self._model_label or self._agent is None:
            await self._note("compacting needs an active model.")
            return
        self._prompt.disabled = True
        self._busy = True
        self._cancel_requested = False
        self._cancel_rendered = False
        self._turn_worker = self.run_worker(self._run_compaction(), exclusive=True, name="turn")

    async def _run_compaction(self) -> None:
        self._set_status("compacting…")
        try:
            note, result = await self._host.compact()
        except asyncio.CancelledError:
            await self._note("Compaction cancelled.")
            raise
        except Exception as exc:
            await self._note(f"compaction failed: {exc}")
        else:
            if result is not None:
                self._apply_compaction_estimate(result.freed_tokens)
                note += " The ctx figure now shows an estimate, refined on your next reply."
            await self._note(note)
        finally:
            await self._finish_turn()

    async def _command_context(self, _arg: str) -> None:
        from jutul_agent.agent.context_editing import clear_tool_uses_trigger_tokens
        from jutul_agent.agent.memory import list_memory_notes
        from jutul_agent.agent.summarization import auto_compact_trigger_tokens

        system_tokens, memory_tokens = context_component_estimates(
            self._session, self._memory_dir()
        )
        body = render_context_panel(
            model_label=self._model_label,
            usage=self._last_usage,
            window=self._context_window_tokens,
            first_usage=self._first_usage,
            model_calls=self._model_calls,
            system_prompt_tokens=system_tokens,
            memory_index_tokens=memory_tokens,
            memory_notes=len(list_memory_notes(self._memory_dir())),
            compact_trigger_tokens=auto_compact_trigger_tokens(self._context_window_tokens),
            clear_trigger_tokens=clear_tool_uses_trigger_tokens(self._context_window_tokens),
        )
        await self._log.mount(MessageBlock("Context", "system", body, markdown=True))
        self._jump_to_latest()

    async def _replay_history(self) -> None:
        """Mount the resumed conversation's prior exchanges from the trace.

        Only the user/assistant prose is replayed; tool calls and reasoning
        stay in the trace (``/transcript`` renders everything). Long sessions
        replay just the newest messages so startup stays instant.
        """
        events = [
            event
            for event in self._session.trace.iter_events()
            if event.kind in ("message_user", "message_assistant")
        ]
        shown = events[-_REPLAY_MAX_MESSAGES:]

        note = [f"Resumed session: {self._session.title or self._session.session_id}."]
        if len(shown) < len(events):
            note.append(
                f"Showing the last {len(shown)} of {len(events)} messages; "
                "/transcript has the full record."
            )
        note.append(
            "The Julia REPL restarted with this session: variables and loaded "
            "packages from earlier turns are gone (files on disk remain)."
        )
        blocks: list[MessageBlock] = [MessageBlock("System", "system", "\n".join(note))]
        for event in shown:
            content = str(event.payload.get("content") or "")
            if not content.strip():
                continue
            if event.kind == "message_user":
                blocks.append(MessageBlock("You", "user", content, markdown="\n" in content))
            else:
                blocks.append(MessageBlock("Assistant", "assistant", content, markdown=True))
        await self._log.mount_all(blocks)
        self._jump_to_latest()

    async def _handle_approval_nav(self, key: str) -> bool:
        if not self._approval_menu.visible or self._busy:
            return False
        if key == "up":
            self._approval_menu.action_move_up()
            return True
        if key == "down":
            self._approval_menu.action_move_down()
            return True
        if key == "enter":
            self._approval_menu.action_confirm()
            return True
        if key == "escape":
            self._approval_menu.action_select_reject()
            return True
        return False

    async def action_cycle_approval_mode(self) -> None:
        self._approval_mode = self._approval_mode.cycle_next()
        label = self._approval_mode.display_label()
        await self._note(f"Permission mode: {label}")
        self._refresh_prompt_guide()
        await self._auto_approve_pending_if_allowed()
        self._rebuild_for_approval_mode()

    def _rebuild_for_approval_mode(self) -> None:
        """Bake the new mode into the agent graph when it is safe to rebuild.

        Mid-turn or under a pending approval the client-side policy loop
        already applies the new mode; the rebuild happens on the next safe
        occasion a mode change lands while idle.
        """
        if self._busy or self._pending_interrupts or self._agent is None:
            return
        try:
            self._host.reconfigure(approval_mode=str(self._approval_mode))
        except Exception:
            return  # the client-side loop still applies the mode
        self._agent = self._host.agent
        self._backend = self._host.backend

    async def action_interrupt(self) -> None:
        """Ctrl+C: interrupt → copy → double-press to exit.

        Priority:
        1. A turn is running → interrupt it (same as Ctrl+G).
        2. Idle with a text selection → copy it (OSC 52). Textual turns mouse
           drags into a selection; this is what lets the user actually copy it
           instead of quitting. (No mouse? Use ``/copy`` for the last reply.)
        3. Idle, nothing selected → first press arms exit and shows a hint, a
           second press within the window quits.
        """
        if self._busy:
            self._disarm_quit()
            await self.action_cancel_turn()
            return

        selected: str | None = None
        with contextlib.suppress(Exception):
            selected = self.screen.get_selected_text()
        if selected:
            self.copy_to_clipboard(selected)
            with contextlib.suppress(Exception):
                self.screen.selections = {}
            self._disarm_quit()
            self.notify("Copied selection to clipboard.", timeout=2.0)
            return

        if self._quit_armed:
            self.exit()
            return
        self._arm_quit()

    def _arm_quit(self) -> None:
        self._quit_armed = True
        self.notify("Press Ctrl+C again to exit.", timeout=2.0)
        if self._quit_timer is not None:
            self._quit_timer.stop()
        self._quit_timer = self.set_timer(2.0, self._disarm_quit)

    def _disarm_quit(self) -> None:
        self._quit_armed = False
        if self._quit_timer is not None:
            self._quit_timer.stop()
            self._quit_timer = None

    async def _auto_approve_pending_if_allowed(self) -> None:
        """Resume an already-shown approval once a mode change clears it.

        Mirrors the menu-driven path: hide the menu and drop the transient
        approval cards before resuming so nothing stale lingers in the log.
        """
        if not (self._pending_interrupts and self._should_auto_approve_pending()):
            return
        self._hide_approval_menu()
        await self._clear_approval_blocks()
        resume_payload = self._build_resume_payload({"type": "approve"})
        # Decided: the interrupts are consumed by the payload. Clearing now
        # keeps the status bar and activity label from reporting an approval
        # as pending for the whole resumed turn.
        self._pending_interrupts = []
        await self._resume_turn(resume_payload)

    async def on_approval_menu_selected(self, event: ApprovalMenu.Selected) -> None:
        if not self._pending_interrupts or self._busy:
            return
        option = event.option
        for category in option.allowlist_categories:
            self._tool_allowlist.add(category)
            label = category.replace("_", " ")
            await self._note(f"Always allow {label} for this session.")
        self._hide_approval_menu()
        await self._resume_pending(option.decision)

    def on_markdown_link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Open file:// links in the OS default application."""
        href = event.href
        if href.startswith("file://"):
            path = Path(urllib.request.url2pathname(urllib.parse.urlparse(href).path))
            open_path(path)
            event.prevent_default()

    def on_resize(self, _event: events.Resize) -> None:
        if self._resize_timer is not None:
            self._resize_timer.stop()
        self._resize_timer = self.set_timer(
            _RESIZE_DEBOUNCE_SECONDS,
            self._refresh_after_resize,
        )

    def _refresh_after_resize(self) -> None:
        # Textual handles widget layout on its own, but our markdown bodies
        # don't re-wrap until their content is re-rendered, so nudge each
        # wrapping widget explicitly. A whole-screen refresh covers the rest.
        self._resize_timer = None
        # This is a debounced timer callback; it can fire while the app is tearing
        # down (e.g. the test harness exiting), when there is no active screen.
        # Touching `self.screen` then raises ScreenStackError, so bail out.
        if not self.screen_stack:
            return
        for widget in (
            *self.query(MessageBlock),
            *self.query(ApprovalBlock),
            *self.query(ToolBlock),
        ):
            widget.refresh_for_width()
        self.screen.refresh(layout=True, repaint=True)

    def _jump_to_latest(self) -> None:
        """Scroll to the newest content and re-engage the bottom anchor.

        For user-initiated events (a submitted prompt, a command's response, a
        pending approval) the newest content is what the user acted for, so it
        jumps even if they had scrolled up. Streamed deltas never call this;
        they rely on the anchor alone, which respects a reader's position.
        """
        if not self.is_mounted:
            return
        self._log.scroll_end(animate=False)

    def _set_status(self, text: str) -> None:
        self._status_text = text
        self.query_one("#status", StatusBar).set_state(
            pending_count=len(self._pending_interrupts),
            tool_toggle_available=any(block.expandable for block in self._tool_blocks),
            tools_expanded=self._tools_expanded,
            approval_mode_label=self._approval_mode.display_label(),
        )
        self._refresh_prompt_guide()

    def on_text_area_changed(self, event) -> None:
        if event.text_area.id != "prompt":
            return
        if getattr(event.text_area, "consume_changed_suppression", lambda: False)():
            return
        if self._setting_prompt_value:
            self._setting_prompt_value = False
        elif self._history.is_navigating:
            self._history.reset()
        self._refresh_prompt_guide()
        if self._pending_interrupts and self._approval_menu.visible:
            self._approval_menu.set_reply_preview(self._prompt.value)

    async def on_prompt_text_area_submitted(self, event: PromptTextArea.Submitted) -> None:
        if self._busy:
            return
        text = event.value.strip()
        if not text:
            return

        self._history.record(text)
        self._history.reset()
        self._prompt.clear()

        if text.startswith("/"):
            await self._handle_command(text)
            return

        if self._pending_interrupts:
            # Typing a message while approval is pending is the "No, and tell the
            # agent what to do differently" path: decline the action and send the
            # text as the reason the model sees. (respond is a synthetic success,
            # used only as a fallback when the request disallows reject.)
            allowed = self._pending_allowed_decisions()
            if "reject" in allowed:
                await self._resume_pending({"type": "reject", "message": text})
            elif "respond" in allowed:
                await self._resume_pending({"type": "respond", "message": text})
            else:
                await self._note("This request can't take a reply. Choose an option above.")
            return

        if self._agent is None:
            await self._note(
                "No model is ready yet. Open the selector with `/model` to choose one and "
                "add the provider's API key (or pick a local Ollama model that does not need one)."
            )
            return

        self._session.adopt_title(text)
        self._prompt.disabled = True
        self._busy = True
        self._cancel_requested = False
        self._cancel_rendered = False
        self._julia_running_on_cancel = False

        await self._mount_welcome_if_empty()
        user_block = MessageBlock("You", "user", text, markdown="\n" in text)
        await self._log.mount(user_block)
        self._jump_to_latest()
        self._set_status("thinking…")

        self._turn_worker = self.run_worker(self._run_turn(text), exclusive=True, name="turn")

    def on_prompt_text_area_history_previous(self, event: PromptTextArea.HistoryPrevious) -> None:
        if self._prompt.disabled:
            return
        new_value = self._history.up(event.current_text)
        if new_value is None:
            return
        self._set_prompt_value(new_value)
        self._refresh_prompt_guide()

    def on_prompt_text_area_history_next(self, event: PromptTextArea.HistoryNext) -> None:
        if self._prompt.disabled:
            return
        new_value = self._history.down()
        if new_value is None:
            return
        self._set_prompt_value(new_value)
        self._refresh_prompt_guide()

    def action_complete_prompt(self) -> None:
        prompt_input = self._prompt
        if self.focused is not prompt_input or prompt_input.disabled:
            return

        command, separator, rest = prompt_input.value.partition(" ")
        if not command.startswith("/") or (separator and rest):
            return

        matches = matching_specs(command, self._command_specs())
        if len(matches) != 1:
            self._refresh_prompt_guide()
            return

        completed = matches[0].name
        if not separator and matches[0].argument_hint:
            completed += " "
        self._set_prompt_value(completed)
        self._refresh_prompt_guide()

    async def _handle_command(self, command: str) -> None:
        """Dispatch a slash command to its ``_command_<name>`` handler.

        The handler method is named by the spec (``SlashCommandSpec.handler_attr``),
        so declaring a spec in ``commands.py`` and defining the method here is all
        it takes to add a command.
        """
        head, _, tail = command.partition(" ")
        spec = find_command(head.lower())
        if spec is None:
            await self._note(f"unknown command: {head}. Try /help.")
            return
        await getattr(self, spec.handler_attr)(tail.strip())

    async def _command_transcript(self, arg: str) -> None:
        fmt = arg.lower()
        if fmt in ("", "html"):
            target = self._session.output_dir / "transcript.html"
            with TraceLog(self._session.state_dir / "trace.sqlite") as log_db:
                content = render_html(log_db.iter_events())
        elif fmt in ("md", "markdown"):
            target = self._session.output_dir / "transcript.md"
            with TraceLog(self._session.state_dir / "trace.sqlite") as log_db:
                content = render_markdown(log_db.iter_events())
        else:
            await self._note("usage: /transcript [md]")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        open_path(target)
        await self._note(f"transcript written to {target}")

    async def _command_copy(self, _arg: str) -> None:
        await self._copy_last_assistant_message()

    async def _command_clear(self, _arg: str) -> None:
        await self._clear_visible_log()

    async def _command_quit(self, _arg: str) -> None:
        self.exit()

    async def _command_help(self, _arg: str) -> None:
        lines = [f"{spec.name:<16}{spec.description}" for spec in self._command_specs()]
        await self._note("\n".join(lines))

    async def _command_approval_mode(self, arg: str) -> None:
        if not arg:
            await self._note(
                f"approval mode is `{self._approval_mode.value}`. "
                "Usage: /approval-mode ask|workspace|auto"
            )
            return
        try:
            self._approval_mode = parse_approval_mode(arg)
        except ValueError as exc:
            await self._note(str(exc))
            return
        await self._note(
            f"approval mode set to `{self._approval_mode.value}` for this session. "
            "Restart jutul-agent to change which tools interrupt at build time."
        )
        await self._auto_approve_pending_if_allowed()
        self._refresh_prompt_guide()

    async def _command_key(self, arg: str) -> None:
        """View provider API keys, or set one (``/key openai``).

        Setting a key here works even when the provider's key is already present,
        so it is the way to replace a wrong or expired one without leaving the app.
        """
        from jutul_agent.credentials import key_status, provider_by_name, user_env_path

        if not arg:
            lines = [f"API keys are saved in {user_env_path()}", ""]
            for st in key_status():
                if not st.is_set:
                    detail = "not set"
                elif st.source == "environment":
                    detail = f"set in your environment ({st.masked})"
                elif st.shadowed:
                    detail = f"saved, overridden by your environment ({st.masked})"
                else:
                    detail = f"saved ({st.masked})"
                lines.append(f"  {st.label:<10} {st.env_var:<20} {detail}")
            lines.append("")
            lines.append("Set or replace one with `/key <provider>`, e.g. `/key openai`.")
            await self._note("\n".join(lines))
            return

        info = provider_by_name(arg)
        if info is None or info.key_env_var is None:
            from jutul_agent.credentials import key_providers

            known = ", ".join(p.name for p in key_providers())
            await self._note(f"unknown provider `{arg}`. Known: {known}.")
            return

        env_var = info.key_env_var

        def _on_key(key: str | None) -> None:
            if not key:
                self.run_worker(self._note(f"`{info.label}` key unchanged."), name="key-note")
                return
            from jutul_agent.credentials import store_credential

            store_credential(env_var, key)
            self.run_worker(self._note(f"Saved {env_var} to {user_env_path()}."), name="key-note")

        self.push_screen(ApiKeyModal(env_var=env_var, provider_label=info.label), _on_key)

    async def _note(self, text: str) -> None:
        await self._log.mount(MessageBlock("System", "system", text))
        self._jump_to_latest()

    async def _copy_last_assistant_message(self) -> None:
        """Copy the most recent assistant reply to the clipboard (OSC 52).

        A reliable alternative to mouse-selection when the terminal doesn't
        cooperate with Textual's text selection.
        """
        assistant_blocks = [
            block for block in self._log.query(MessageBlock) if block.has_class("assistant")
        ]
        if not assistant_blocks:
            await self._note("no assistant message to copy yet")
            return
        self.copy_to_clipboard(assistant_blocks[-1].content_text)
        await self._note("copied the last assistant message to the clipboard")

    async def _command_add_dir(self, raw: str) -> None:
        """Add a folder to the agent filesystem, or list the folders already added.

        With no argument, lists the folders added this session; with a path,
        records it so the file tools reach it. The real-path backend already
        reaches any absolute path, so the folder is usable on the next tool call.
        """
        if self._backend is None:
            await self._note("adding folders isn't available in this session.")
            return

        path = raw.strip().strip("\"'")
        if not path:
            dirs = added_dirs(self._backend)
            if dirs:
                lines = ["Added folders:"]
                lines += [f"  `{entry.path}`" for entry in dirs]
                await self._note("\n".join(lines))
            else:
                await self._note(
                    "usage: /add-dir <path>. Adds a folder the agent can read and edit."
                )
            return

        try:
            entry = add_dir(self._backend, path, workspace=workspace_root())
        except AddDirError as exc:
            await self._note(f"could not add folder: {exc}")
            return

        await self._note(
            f"Added `{entry.path}`. The agent reads, greps, writes, and edits it "
            f"by its real absolute path in every tool (file tools, Julia, and shell)."
        )

    async def _command_model(self, raw: str) -> None:
        """Open the selector, or switch directly to a given ``provider:model``."""
        target = raw.strip().strip("\"'")
        if not target:
            self.push_screen(ModelMenu(current=self._model_label), self._after_model_menu)
            return
        await self._switch_model(target)

    def _after_model_menu(self, choice: ModelChoice | None) -> None:
        if choice is None:
            return
        model_id, scope = choice
        self.run_worker(self._switch_model(model_id, scope=scope), name="model-switch")

    async def _switch_model(self, model_id: str, *, scope: str = "workspace") -> None:
        """Switch to ``model_id``, prompting for a missing provider key first.

        Refused while a turn is running or an approval is pending. Local models
        go through ``_prepare_local_model`` instead of the key prompt.
        """
        if self._agent_factory is None:
            await self._note("switching models isn't available in this session.")
            return
        if self._busy or self._pending_interrupts:
            await self._note("finish or cancel the current turn before switching models.")
            return
        if model_id == self._model_label and self._agent is not None:
            await self._note(f"already using `{model_id}`.")
            return

        from jutul_agent.models import is_local

        if is_local(model_id):
            await self._prepare_local_model(model_id, scope)
            return

        from jutul_agent.credentials import missing_credential

        env_var = missing_credential(model_id)
        if env_var is not None:
            self._prompt_api_key(model_id, scope, env_var)
            return
        await self._apply_model(model_id, scope)

    async def _prepare_local_model(self, model_id: str, scope: str) -> None:
        """Check a local (Ollama) model is reachable, pulled, and tool-capable,
        then switch. Reports if the server is down; pulls the model if missing.
        """
        from jutul_agent import ollama_client
        from jutul_agent.models import is_ollama_cloud

        if not await ollama_client.is_reachable():
            await self._note(
                f"Ollama isn't reachable at {ollama_client.host()}. Start it with "
                "`ollama serve` (or install it from https://ollama.com), then try again."
            )
            return
        if is_ollama_cloud(model_id):
            # Hosted by Ollama (needs `ollama signin`); nothing to pull locally.
            await self._apply_model(model_id, scope)
            return
        name = ollama_client.model_name(model_id)
        if await ollama_client.is_installed(name):
            await self._apply_local_if_tool_capable(model_id, scope, name)
            return

        def _after_pull(ok: bool | None) -> None:
            if ok:
                self.run_worker(
                    self._apply_local_if_tool_capable(model_id, scope, name), name="model-switch"
                )
            else:
                self.run_worker(
                    self._note(f"`{name}` was not pulled; switch cancelled."),
                    name="model-note",
                )

        self.push_screen(OllamaPullModal(model_name=name), _after_pull)

    async def _apply_local_if_tool_capable(self, model_id: str, scope: str, name: str) -> None:
        """Switch only if the daemon exposes tool calling for the model; the
        agent is tool-driven, so a tool-less model can't run here."""
        from jutul_agent import ollama_client

        if not await ollama_client.supports_tools(name):
            await self._note(
                f"`{model_id}` doesn't support tool calling, which this agent requires. "
                "Pick a tool-capable model, or update Ollama and re-pull if it should "
                "support tools."
            )
            return
        await self._apply_model(model_id, scope)

    def _prompt_api_key(self, model_id: str, scope: str, env_var: str) -> None:
        """Collect the provider key in a modal, store it, then resume the switch."""
        from jutul_agent.models import provider_info

        info = provider_info(model_id)
        label = info.label if info else model_id

        def _on_key(key: str | None) -> None:
            if not key:
                self.run_worker(
                    self._note(f"`{model_id}` needs {env_var}; switch cancelled."),
                    name="model-note",
                )
                return
            from jutul_agent.credentials import store_credential

            store_credential(env_var, key)
            self.run_worker(self._apply_model(model_id, scope), name="model-switch")

        self.push_screen(ApiKeyModal(env_var=env_var, provider_label=label), _on_key)

    async def _apply_model(self, model_id: str, scope: str) -> None:
        """Rebuild the agent on the new model and persist the choice.

        The rebuilt agent shares the session's checkpointer and thread id, so
        the conversation continues; ``/add-dir`` folders are re-applied.
        """
        dirs = [entry.path for entry in added_dirs(self._backend)] if self._backend else []
        try:
            agent, backend = self._agent_factory(model_id, dirs)
        except Exception as exc:
            await self._note(f"could not switch to `{model_id}`: {exc}")
            return

        self._agent = agent
        self._backend = backend
        self._model_label = model_id
        self._host.adopt_agent(agent, backend, model=model_id)
        self.query_one("#status", StatusBar).set_model(model_id)
        self._context_window_tokens = None
        self._refresh_context_status()
        self.run_worker(self._fetch_context_window(), name="ctx-window")
        self._refresh_subtitle()
        record_recent_model(model_id)
        where = self._persist_model(model_id, scope)
        await self._note(f"model changed to `{model_id}` (saved to {where}).")

    def _persist_model(self, model_id: str, scope: str) -> str:
        """Save the model to the workspace or user-global config.

        Returns where it was saved, for the confirmation note.
        """
        from dataclasses import replace

        try:
            if scope == "global":
                from jutul_agent.user_config import load_user_config, write_user_config

                write_user_config(replace(load_user_config(), model=model_id))
                return "user config"
            ws = workspace_root()
            from jutul_agent.workspace import load_workspace_config, write_workspace_config

            write_workspace_config(replace(load_workspace_config(ws), model=model_id), workspace=ws)
            return "this workspace"
        except OSError:
            return "this session only (could not write config)"

    def _refresh_subtitle(self) -> None:
        parts = [self._session.simulator.display_name]
        if self._model_label:
            parts.append(self._model_label)
        parts.append(display_session_id(self._session.session_id))
        self.sub_title = " · ".join(parts)

    async def _run_turn(self, prompt: str) -> None:
        self._set_status("thinking…")
        self._assistant_text_rendered = False
        try:
            result = await self._host.drive_turn(
                lambda: self._host.runner.run_prompt(prompt, on_message=self._render_message),
                approval_mode=self._approval_mode,
                allowlist=self._tool_allowlist,
                on_message=self._render_message,
            )
            await self._apply_turn_result(result)
        except asyncio.CancelledError:
            await self._render_turn_cancelled()
            raise
        except Exception as exc:
            await self._render_turn_error(exc)
        finally:
            await self._finish_turn()

    async def _resume_turn(
        self,
        resume_payload: dict[str, dict[str, list[dict[str, str]]]],
    ) -> None:
        self._set_status("resuming…")
        try:
            result = await self._host.drive_turn(
                lambda: self._host.runner.resume(resume_payload, on_message=self._render_message),
                approval_mode=self._approval_mode,
                allowlist=self._tool_allowlist,
                on_message=self._render_message,
            )
            await self._apply_turn_result(result)
        except asyncio.CancelledError:
            await self._render_turn_cancelled()
            raise
        except Exception as exc:
            await self._render_turn_error(exc)
        finally:
            await self._finish_turn()

    async def _render_turn_cancelled(self) -> None:
        # One cancel propagates CancelledError up through every nested
        # _resume_turn/_run_turn in an auto-approve chain, and each would render.
        # Render only the first (it carries the Julia-running detail, before the
        # inner _finish_turn clears the flag); swallow the rest.
        if self._cancel_rendered:
            return
        self._cancel_rendered = True
        await self._flush_stream()
        await self._mark_running_tools_cancelled()
        self._reset_approval_state()
        await self._clear_approval_note()
        if self._julia_running_on_cancel:
            # The kernel interrupts the running eval (SIGINT) and keeps the session,
            # unless the eval was wedged and had to be restarted.
            preserved = getattr(self._session.julia, "cancel_preserved_state", True)
            message = (
                "Turn cancelled. Interrupted the running Julia command; loaded "
                "packages and variables are intact."
                if preserved
                else "Turn cancelled. The Julia command wouldn't interrupt, so the "
                "session was restarted and its state cleared."
            )
        else:
            message = "Turn cancelled."
        await self._log.mount(MessageBlock("System", "system", message))
        self._jump_to_latest()

    async def action_cancel_turn(self) -> None:
        if not self._busy or self._cancel_requested:
            return
        self._cancel_requested = True
        self._julia_running_on_cancel = self._has_running_julia_tool()
        self._set_status(
            "cancelling… (interrupting Julia)" if self._julia_running_on_cancel else "cancelling…"
        )
        worker = self._turn_worker
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.cancel()

    def _has_running_julia_tool(self) -> bool:
        return any(
            block.status == "running" and block.tool_name in {"run_julia", "plot_julia"}
            for block in self._tool_blocks
        )

    async def _mark_running_tools_cancelled(self) -> None:
        for block in self._tool_blocks:
            if block.status == "running":
                await block.set_cancelled("turn cancelled")

    def _reset_approval_state(self) -> None:
        self._pending_interrupts = []
        self._active_approval_blocks = []
        self._hide_approval_menu()

    async def _apply_turn_result(self, result: TurnRunResult) -> None:
        await self._flush_stream()
        self._update_usage_from_messages(result.messages)
        self._pending_interrupts = result.interrupts
        if self._pending_interrupts:
            await self._render_interrupts(result.interrupts)
            return
        self._active_approval_blocks = []
        if not self._assistant_text_rendered:
            # The turn finished without any assistant prose (e.g. an empty model
            # reply). Show a muted placeholder so the user sees the turn completed.
            await self._log.mount(
                MessageBlock("Assistant", "assistant", "_(no response)_", markdown=True)
            )

    def _should_auto_approve_pending(self) -> bool:
        if not self._pending_interrupts:
            return False
        return all(
            should_auto_approve_interrupt(
                interrupt.value,
                self._approval_mode,
                allowlist=self._tool_allowlist,
            )
            for interrupt in self._pending_interrupts
        )

    async def action_approve_pending(self) -> None:
        if not self._pending_interrupts or self._busy:
            return
        if "approve" not in self._pending_allowed_decisions():
            return
        await self._resume_pending({"type": "approve"})

    async def action_reject_pending(self) -> None:
        if not self._pending_interrupts or self._busy:
            return
        if "reject" not in self._pending_allowed_decisions():
            return
        await self._resume_pending({"type": "reject"})

    async def _render_turn_error(self, exc: Exception) -> None:
        await self._flush_stream()
        await self._mark_running_tools_cancelled()
        self._reset_approval_state()
        await self._clear_approval_note()
        await self._log.mount(MessageBlock("Error", "error", str(exc)))
        await self._note(
            "The turn stopped early. You can retry your last message or continue from here."
        )

    async def _finish_turn(self) -> None:
        # The kernel self-heals a cancelled eval (SIGINT + drain, keeping state), so
        # there's no force-restart here anymore; see ``JuliaKernel.eval``.
        self._cancel_requested = False
        self._julia_running_on_cancel = False
        self._busy = False
        self._set_status("approval required" if self._pending_interrupts else "ready")
        if not self._pending_interrupts:
            # Settle duties (report sidecar refresh) run in the host's turn
            # loop; here only the UI reset and the one-time title upgrade.
            task = self._host.maybe_title()
            if task is not None:
                self._title_task = task
            self._hide_approval_menu()
            self._prompt.focus()
        self._prompt.disabled = False
        self._refresh_prompt_guide()

    def _show_approval_menu(self) -> None:
        allowed = self._pending_allowed_decisions()
        tool_names = self._pending_tool_names()
        interrupt_values = [
            interrupt.value if isinstance(interrupt.value, dict) else {}
            for interrupt in self._pending_interrupts
        ]
        options = build_approval_options(
            allowed_decisions=allowed,
            tool_names=tool_names,
            interrupt_values=interrupt_values,
        )
        self._approval_menu.set_options(options)
        self._approval_menu.show_menu()
        self._prompt.focus()
        self._refresh_prompt_guide()

    def _hide_approval_menu(self) -> None:
        if self._approval_menu is not None:
            self._approval_menu.hide_menu()

    def _pending_tool_names(self) -> list[str]:
        names: list[str] = []
        for interrupt in self._pending_interrupts:
            value = interrupt.value if isinstance(interrupt.value, dict) else {}
            action_requests = value.get("action_requests")
            if not isinstance(action_requests, list):
                continue
            for action in action_requests:
                if isinstance(action, dict):
                    name = str(action.get("name") or "")
                    if name:
                        names.append(name)
        return names

    async def _render_message(self, msg: Any) -> None:
        # TurnRunner emits exactly these four event types (see agent/turns.py);
        # tool results arrive as TurnToolEvents, never as raw ToolMessages.
        if isinstance(msg, TurnTextDelta):
            if msg.text:
                self._assistant_text_rendered = True
                await self._stream.append_prose(self._log, msg.text)
            return

        if isinstance(msg, TurnTextEnd):
            await self._flush_stream()
            return

        if isinstance(msg, TurnReasoningDelta):
            await self._render_reasoning_delta(msg)
            return

        if isinstance(msg, TurnToolEvent):
            await self._render_tool_event(msg)

    async def _render_reasoning_delta(self, msg: TurnReasoningDelta) -> None:
        await self._stream.append_reasoning(self._log, msg.text)
        if msg.text:
            self._set_status("thinking…")

    async def _render_tool_event(self, msg: TurnToolEvent) -> None:
        await self._flush_stream()
        if msg.event == "delta":
            block = self._matching_tool_block(
                tool_call_id=msg.tool_call_id,
                tool_name=msg.tool_name,
            )
            if block is None:
                block = self._new_tool_block(msg.tool_name, tool_call_id=msg.tool_call_id)
                self._tool_blocks.append(block)
                await self._log.mount(block)
            await block.append_output(msg.content)
            return

        if msg.event in {"requested", "started"}:
            await self._mount_tool_call(
                {"name": msg.tool_name, "args": msg.args or {}, "id": msg.tool_call_id}
            )
            return

        block = self._matching_tool_block(
            tool_call_id=msg.tool_call_id,
            tool_name=msg.tool_name,
        )
        if block is None:
            block = self._new_tool_block(msg.tool_name, tool_call_id=msg.tool_call_id)
            await self._log.mount(block)
            self._tool_blocks.append(block)
        await block.set_result(msg.content, is_error=msg.event == "error")
        self._set_status("thinking…")

    async def _flush_stream(self) -> None:
        await self._stream.flush(reasoning_expanded=self._tools_expanded)

    async def _mount_tool_call(self, call: dict[str, Any]) -> None:
        name = call.get("name") or "tool"
        tool_call_id = str(call.get("id")) if call.get("id") else None
        existing = self._matching_tool_block(tool_call_id=tool_call_id, tool_name=name)
        if existing is not None:
            return

        block = self._new_tool_block(
            name,
            call.get("args") if isinstance(call.get("args"), dict) else None,
            tool_call_id=tool_call_id,
        )
        # Append before awaiting mount so concurrent emitters (e.g. the
        # ``requested`` event from the message stream racing the ``started``
        # event from the tool-call projection) see the block and dedupe.
        self._tool_blocks.append(block)
        await self._log.mount(block)
        if name in {"run_julia", "plot_julia"}:
            block.start_elapsed_timer()
        self._set_status("updating plan…" if name == "write_todos" else f"running {name}…")

    async def _render_interrupts(self, interrupts: list[TurnInterrupt]) -> None:
        self._active_approval_blocks = []
        rendered_cards = 0
        for interrupt in interrupts:
            await self._mark_tool_blocks_pending_approval(interrupt)
            for card in render_interrupt_cards(
                interrupt.interrupt_id,
                interrupt.value,
                workspace_root=workspace_root(),
            ):
                block = ApprovalBlock(card)
                await self._log.mount(block)
                self._active_approval_blocks.append(block)
                rendered_cards += 1

        # The card shows the request and the menu below shows the options, so no
        # extra "approval is pending" line is needed, only a note for the rare
        # request the TUI can't resolve at all.
        if self._pending_allowed_decisions():
            await self._clear_approval_note()
        else:
            await self._set_approval_note(
                "This request needs a decision the TUI can't make; resolve it elsewhere."
            )
        self._show_approval_menu()
        self._set_status("approval required")
        self._refresh_prompt_guide()
        self._jump_to_latest()

    async def _resume_pending(self, decision: dict[str, str]) -> None:
        if not self._pending_interrupts:
            await self._note("no approval is pending")
            return

        allowed = self._pending_allowed_decisions()
        decision_type = decision["type"]
        if decision_type not in allowed:
            rendered = ", ".join(sorted(allowed)) or "(none)"
            await self._note(f"`{decision_type}` is not allowed here. Allowed: {rendered}.")
            return

        self._hide_approval_menu()
        resume_payload = self._build_resume_payload(decision)
        # Decided: the interrupts are consumed by the payload. Clearing now
        # keeps the status bar and activity label from reporting an approval
        # as pending for the whole resumed turn.
        self._pending_interrupts = []
        self._prompt.disabled = True
        self._busy = True
        self._cancel_requested = False
        self._cancel_rendered = False
        self._julia_running_on_cancel = False
        await self._preview_pending_decision(decision)
        self._set_status("resuming…")
        self._turn_worker = self.run_worker(
            self._resume_turn(resume_payload), exclusive=True, name="turn"
        )

    async def action_clear_visible_log(self) -> None:
        await self._clear_visible_log()

    async def action_toggle_tool_output(self) -> None:
        """Ctrl+O: toggle verbose output for every tool and reasoning card."""
        self._tools_expanded = not self._tools_expanded
        for block in list(self._tool_blocks):
            await block.set_expanded(self._tools_expanded)
        for reasoning in self._log.query(ReasoningBlock):
            await reasoning.set_expanded(self._tools_expanded)
        self._set_status(self._status_text)

    async def _clear_visible_log(self) -> None:
        if self._busy:
            await self._note("cannot clear the log while a turn is running")
            return
        if self._pending_interrupts:
            await self._note("cannot clear the log while approval is pending")
            return

        await self._log.remove_children()
        self._stream = _AssistantStream()
        self._tool_blocks.clear()
        self._active_approval_blocks.clear()
        self._pending_approval_note = None
        await self._mount_welcome_if_empty()
        self._set_status("ready")
        self._refresh_prompt_guide()

    async def _mount_welcome_if_empty(self) -> None:
        if self._log.children:
            return
        await self._log.mount(WelcomeBlock(simulator_label=self._session.simulator.display_name))
        self._jump_to_latest()

    def _command_specs(self) -> list[SlashCommandSpec]:
        return list(ALL_COMMANDS)

    def _refresh_prompt_guide(self) -> None:
        guide = self.query_one("#prompt-guide", PromptGuide)
        guide.set_message(self._compute_prompt_guide())
        guide.set_activity(self._activity_label())

    def _activity_label(self) -> str:
        # Lives in the bottom bar next to the input. The Julia warm-up state
        # is a separate chip in the top status bar, so neither line ever
        # replaces the other.
        if self._pending_interrupts:
            return "approval required"
        if self._busy:
            return self._status_text
        return "ready"

    def _compute_prompt_guide(self) -> str:
        position = self._history.position
        if position is not None:
            idx, total = position
            return f"History {idx}/{total} · Ctrl+P/↑ prev · Ctrl+N/↓ next"
        if self._pending_interrupts:
            value = self._prompt.value
            if value.startswith("/"):
                return self._command_guide(value)
            if value.strip():
                return "Enter sends this to the agent as what to do instead"
            # The approval menu beneath the prompt owns the key map (↑/↓ move,
            # enter confirm, type to reply); don't repeat it in the footer.
            return ""
        if self._busy:
            return "Ctrl+G cancel"
        value = self._prompt.value
        if value.startswith("/"):
            return self._command_guide(value)
        if self._tools_expanded:
            tool_hint = " · Ctrl+O collapse tool output"
        elif any(block.expandable for block in self._tool_blocks):
            tool_hint = " · Ctrl+O expand tool output"
        else:
            tool_hint = ""
        # The current mode lives in the top status bar (always visible); the
        # footer only needs to say how to change it, not repeat its value.
        return (
            f"Enter send · Shift+Enter newline · Ctrl+P/↑ history · Shift+Tab cycle mode{tool_hint}"
        )

    def _command_guide(self, value: str) -> str:
        command, separator, rest = value.partition(" ")
        specs = self._command_specs()
        exact = next((spec for spec in specs if spec.name == command), None)

        if separator and not rest and exact and exact.argument_hint:
            return f"{exact.name} {exact.argument_hint} · Enter send · Up/Down history"

        matches = matching_specs(command, specs)
        if len(matches) == 1 and exact is not None:
            detail = exact.description
            if exact.argument_hint and not separator:
                detail += f" · {exact.argument_hint}"
            return f"{exact.name} · {detail}"

        if matches:
            rendered = " · ".join(spec.name for spec in matches[:5])
            if len(matches) > 5:
                rendered += " · ..."
            return f"Tab completes unique match · {rendered}"

        return "Unknown command · /help lists available commands"

    def _set_prompt_value(self, value: str) -> None:
        prompt_input = self._prompt
        self._setting_prompt_value = True
        prompt_input.suppress_next_changed()
        prompt_input.value = value

    def _pending_allowed_decisions(self) -> frozenset[str]:
        """Decisions every pending interrupt accepts, scoped to what the TUI can do."""
        return pending_allowed_decisions(self._pending_interrupts)

    def _build_resume_payload(
        self,
        decision: dict[str, str],
    ) -> dict[str, dict[str, list[dict[str, str]]]]:
        return build_resume_payload(self._pending_interrupts, decision)

    def _new_tool_block(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        tool_call_id: str | None = None,
    ) -> ToolBlock:
        """Construct a tool card that honours the current verbose mode."""
        return ToolBlock(name, args, tool_call_id=tool_call_id, expanded=self._tools_expanded)

    def _matching_tool_block(
        self,
        *,
        tool_call_id: str | None,
        tool_name: str,
    ) -> ToolBlock | None:
        if tool_call_id is not None:
            for block in reversed(self._tool_blocks):
                if block.tool_call_id == tool_call_id:
                    return block
        for block in reversed(self._tool_blocks):
            if block.tool_name == tool_name and not block.has_output:
                return block
        return None

    async def _mark_tool_blocks_pending_approval(self, interrupt: TurnInterrupt) -> None:
        value = interrupt.value if isinstance(interrupt.value, dict) else {}
        action_requests = value.get("action_requests")
        if not isinstance(action_requests, list):
            return

        used: set[int] = set()
        for action in action_requests:
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("name") or "tool")
            for block in reversed(self._tool_blocks):
                if id(block) in used or block.tool_name != tool_name or block.has_output:
                    continue
                await block.set_pending_approval()
                used.add(id(block))
                break

    async def _set_approval_note(self, text: str) -> None:
        """Mount the (single) transient "approval is pending" line, tracked so
        it can be removed once the user decides."""
        await self._clear_approval_note()
        block = MessageBlock("System", "system", text)
        self._pending_approval_note = block
        await self._log.mount(block)
        self._jump_to_latest()

    async def _clear_approval_note(self) -> None:
        if self._pending_approval_note is not None:
            with contextlib.suppress(Exception):
                await self._pending_approval_note.remove()
            self._pending_approval_note = None

    async def _clear_approval_blocks(self) -> None:
        """Remove the pending approval cards and the pending note from the log.

        Both are transient: their only job is to show the diff/command and the
        instruction while the user decides. Once decided, the outcome is
        carried by the tool card (running / rejected / responded) and the
        durable record lives in the SQLite trace, so leaving a stale colored
        card or a "approval is pending" line in the log just misleads.
        """
        for block in self._active_approval_blocks:
            with contextlib.suppress(Exception):
                await block.remove()
        self._active_approval_blocks = []
        await self._clear_approval_note()

    async def _preview_pending_decision(self, decision: dict[str, str]) -> None:
        decision_type = decision.get("type") or "pending"
        reason = decision.get("message")

        await self._clear_approval_blocks()

        if decision_type == "approve":
            for block in self._tool_blocks:
                if not block.has_output:
                    await block.set_running()
            return

        if decision_type == "reject":
            for block in self._tool_blocks:
                if not block.has_output:
                    await block.set_rejected(reason)
            return

        if decision_type == "respond":
            # The tool does not run; the user's message goes back as its
            # result. Without this the cards stay "waiting for approval".
            for block in self._tool_blocks:
                if not block.has_output:
                    await block.set_responded(reason)
