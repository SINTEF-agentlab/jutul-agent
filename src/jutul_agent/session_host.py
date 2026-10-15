"""One running session, with everything it needs to take a turn.

A ``SessionHost`` owns a ``Session`` (its Julia kernel, trace, and directories),
the agent built for it, and the ``TurnRunner`` that drives a turn. ``start`` is
the single session bootstrap every front end uses: the TUI, the headless CLI,
the web server, and the bench solver all stand up sessions here, so the kernel
environment, capability discovery, and the checkpointer cannot drift between
them. The kernel half is ``session_core.start_core``, shared with front ends
that bring their own agent. ``aclose`` tears everything down.

The constructor takes a ready-made session and agent so tests can wrap fakes;
``start`` is the production path that stands up a real kernel.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

from jutul_agent.agent.turns import TurnRunner

if TYPE_CHECKING:
    from contextlib import AsyncExitStack
    from pathlib import Path

    from jutul_agent.agent.capabilities import Capability
    from jutul_agent.session import Session
    from jutul_agent.session_core import SessionCore
    from jutul_agent.simulators.base import SimulatorAdapter


class SessionHost:
    """A live session plus its agent and turn runner."""

    def __init__(
        self,
        *,
        session: Session,
        agent: Any,
        backend: Any | None = None,
        exit_stack: AsyncExitStack | None = None,
        checkpointer: Any | None = None,
        model: str | None = None,
        approval_mode: str | None = None,
        surface: str = "web",
        extensions: Sequence[Capability] = (),
        workspace: Path | None = None,
        package_sources: Sequence[Any] = (),
        add_dirs: Sequence[Path] = (),
    ) -> None:
        self.session = session
        self.agent = agent
        self.backend = backend
        self.workspace = workspace
        # Launch-time extra folders; the rebuild fallback when there is no live
        # backend to read them from (e.g. the first build after a key arrives).
        self._add_dirs = list(add_dirs)
        self._exit_stack = exit_stack
        self._runner: TurnRunner | None = None
        # Kept so the agent can be rebuilt in place (e.g. /model, /approval-mode)
        # without restarting the kernel; the same checkpointer keeps the history
        # and the same session keeps the live Julia state.
        self._checkpointer = checkpointer
        self._model = model
        self._approval_mode = approval_mode
        self._surface = surface
        self._extensions = list(extensions)
        # The resolved env package sources: kept so an in-place rebuild keeps the
        # read-only depot guard and the simulator's source-path prompt note.
        self._package_sources = list(package_sources)
        # Set once a content-aware (LLM) title has been generated for this session,
        # so the server only attempts it on the first turn.
        self.titled = False
        # At most one live WebSocket drives a session at a time: two would run
        # turns against the one kernel concurrently and corrupt its state. ``attach``
        # claims the session for a connection; ``detach`` releases it on disconnect.
        self._attached = False
        # Whether a turn is running; see the ``busy`` property.
        self._busy = False
        self._active_turn: asyncio.Task[None] | None = None
        # The kernel half this host was started on. It owns the background Julia
        # warm-up (load warm package, set GLMakie offscreen), which runs again after
        # each restart, so teardown cancels whichever is current. Set by ``start``.
        self._core: SessionCore | None = None

    def attach(self) -> bool:
        """Claim this session for a connection; ``False`` if one already holds it."""
        if self._attached:
            return False
        self._attached = True
        return True

    def detach(self) -> None:
        """Release the session so a later connection can attach."""
        self._attached = False

    @property
    def attached(self) -> bool:
        """Whether a live connection currently holds this session."""
        return self._attached

    def set_busy(self, busy: bool) -> None:
        """Mark whether a turn is running, whoever is (or is no longer) watching."""
        self._busy = busy

    def claim_turn(self, task: asyncio.Task[None]) -> bool:
        """Reserve this host for one WebSocket turn, including after disconnect."""
        if self.busy:
            return False
        self._active_turn = task
        task.add_done_callback(self.release_turn)
        return True

    def release_turn(self, task: asyncio.Task[None]) -> None:
        if self._active_turn is task:
            self._active_turn = None

    @property
    def active_turn(self) -> asyncio.Task[None] | None:
        return self._active_turn

    @property
    def busy(self) -> bool:
        """Whether a turn is in flight.

        Separate from ``attached`` because a turn outlives the connection that
        started it: switching sessions closes the socket but leaves the work
        running, and nothing may tear the session down until it finishes.
        """
        return self._busy or (self._active_turn is not None and not self._active_turn.done())

    @property
    def model(self) -> str | None:
        """The session's model spec (``provider:model``), or ``None`` for the default."""
        return self._model

    @property
    def approval_mode(self) -> str | None:
        """The session's human-in-the-loop policy, or ``None`` for the default."""
        return self._approval_mode

    @property
    def branding(self):
        """What this session's capabilities call it, or ``None`` when unbranded.

        Only the layers that reach this surface count, so a capability that
        renames the web UI does not rename a TUI it never runs in.
        """
        from jutul_agent.agent.capabilities import collect_branding, select_for_surface

        return collect_branding(select_for_surface(self._extensions, self._surface))

    def adopt_agent(self, agent: Any, backend: Any | None, *, model: str | None = None) -> None:
        """Point the host at an externally rebuilt agent.

        ``reconfigure`` is the normal rebuild path; this is the escape hatch
        for a front end whose factory built the agent itself (test harnesses).
        The runner is rebuilt lazily against the new agent.
        """
        self.agent, self.backend = agent, backend
        if model is not None:
            self._model = model
        self._runner = None

    def reconfigure(self, *, model: str | None = None, approval_mode: str | None = None) -> None:
        """Rebuild the agent in place with a new model and/or approval policy.

        The kernel, checkpointer, and session are untouched, so conversation
        history and the live Julia REPL survive the switch; only the agent graph
        and its turn runner are replaced."""
        from jutul_agent.agent.added_dirs import added_dirs
        from jutul_agent.agent.builder import build_agent

        new_model = model if model is not None else self._model
        new_approval = approval_mode if approval_mode is not None else self._approval_mode
        # Rebuilding makes a fresh backend, so carry the added folders (launch
        # --add-dir and any runtime /add-dir) over or the switch would silently
        # drop the agent's access to them. Without a live backend (a session
        # started without a model), the launch-time folders are the source.
        carried = (
            [entry.path for entry in added_dirs(self.backend)]
            if self.backend
            else list(self._add_dirs)
        )
        # Build first, then commit the new values, only once it succeeds. A value
        # build_agent rejects (e.g. an unknown approval mode) must leave the host
        # consistent with the agent still running, not reporting a model/approval it
        # never applied (which a same-value reattach would then take as "unchanged").
        agent, backend = build_agent(
            self.session,
            model=new_model,
            checkpointer=self._checkpointer,
            approval_mode=new_approval,
            surface=self._surface,
            extensions=self._extensions,
            added_dirs=carried or None,
            package_sources=self._package_sources or None,
        )
        self.agent, self.backend = agent, backend
        self._model, self._approval_mode = new_model, new_approval
        self._runner = None  # rebuilt lazily against the new agent

    def set_host_context(self, context: dict[str, Any] | None, *, rebuild: bool = True) -> bool:
        """Adopt the host application's selection; return whether it changed.

        Keeps three things in step: the session (which persists the value and
        records the change in the trace, and is what capability tools read), the
        stored capability list (so a later rebuild for an unrelated reason, such
        as a model switch, does not resurrect the previous selection), and the live
        agent's system prompt.

        ``rebuild=False`` is for the moment just after ``start``, where the
        capability list the agent was built from already carries this value and a
        rebuild would only repeat the work. Every later change wants the default:
        the fragment is baked into the agent at build time, so an update reaches
        the model only by building a new one. That is safe mid-session (the
        kernel, the checkpointed history, and the live Julia state all survive)
        but *not* mid-turn, so callers must be idle before calling.
        """
        from jutul_agent.agent.capabilities import replace_host_context

        changed = self.session.adopt_host_context(context)
        # Rewritten even when nothing changed, so the stored layer is the one
        # invariant that always holds: what the agent is told matches what the
        # session knows. Only the rebuild is conditional, because that is the
        # expensive half.
        self._extensions = replace_host_context(self._extensions, context)
        if changed and rebuild and self.agent is not None:
            self.reconfigure()
        return changed

    @property
    def memory_dir(self):
        """The workspace memory directory for this session (created if needed)."""
        from jutul_agent.agent.memory import ensure_memory_dir
        from jutul_agent.paths import workspace_memory_dir

        return ensure_memory_dir(self.session.memory_dir(workspace_memory=workspace_memory_dir()))

    async def compact(self) -> tuple[str, Any | None]:
        """Summarize older turns to free context now.

        Returns a human-readable outcome plus the ``CompactResult`` (``None``
        when there was nothing to compact) so a front end can also adjust its
        live context estimate by the freed tokens.
        """
        from jutul_agent.agent.summarization import MANUAL_KEEP_MESSAGES, compact_thread
        from jutul_agent.models import DEFAULT_MODEL

        result = await compact_thread(
            self.agent,
            thread_id=self.session.session_id,
            model=self._model or DEFAULT_MODEL,
            backend=self.backend,
            trace=self.session.trace,
        )
        if result is None:
            return (
                f"Nothing to compact yet: compaction keeps the newest "
                f"{MANUAL_KEEP_MESSAGES} messages and the conversation isn't longer than that.",
                None,
            )
        extra = " The summarized turns were saved and can be reopened." if result.offloaded else ""
        message = (
            f"Compacted: summarized {result.messages_summarized} older messages and kept the "
            f"{result.messages_kept} most recent.{extra}"
        )
        return message, result

    def add_dir(self, path: str) -> str:
        """Give the agent read/write access to another folder; return a result note."""
        from jutul_agent.agent.added_dirs import AddDirError, add_dir, added_dirs
        from jutul_agent.paths import workspace_root

        if not path:
            dirs = added_dirs(self.backend)
            if not dirs:
                return "Usage: /add-dir <path>. Adds a folder the agent can read and edit."
            return "Added folders:\n" + "\n".join(f"  {e.path}" for e in dirs)
        try:
            entry = add_dir(self.backend, path, workspace=workspace_root())
        except AddDirError as exc:
            return f"Could not add folder: {exc}"
        return f"Added folder: {entry.path}. The agent can read and edit it from its next turn."

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def warmup_task(self) -> Any | None:
        """The background Julia warm-up, for a front end that shows warm status."""
        return self._core.warmup_task if self._core is not None else None

    @property
    def package_sources(self) -> list[Any]:
        """The env's resolved package sources (read-only depot guard inputs)."""
        return list(self._package_sources)

    async def pending_interrupts(self) -> list[Any]:
        """Approvals persisted in the graph state, awaiting a decision.

        A turn that pauses on an approval completes with the interrupt recorded
        in the checkpointer. A front end (re)connecting to the session reads
        these back and re-surfaces them, so the paused turn is never orphaned.
        """
        return await self.runner.pending_interrupts()

    async def drive_turn(
        self,
        start_turn: Callable[[], Awaitable[Any]],
        *,
        approval_mode: Any = None,
        allowlist: Any = None,
        on_message: Any = None,
    ) -> Any:
        """Run one turn to its resting point, applying the approval policy.

        Starts the turn, then keeps resuming past interrupts the policy already
        allows (the mode, or a category the user marked "always allow" this
        session) until the turn completes or an interrupt genuinely needs a
        human. When the turn settles with nothing pending, the end-of-turn
        duties run. Every front end drives turns through here so the policy
        loop and the settle hooks cannot drift between them.
        """
        from jutul_agent.agent.approval import (
            build_resume_payload,
            parse_approval_mode,
            should_auto_approve_interrupt,
        )

        mode = parse_approval_mode(str(approval_mode) if approval_mode is not None else None)
        result = await start_turn()
        while result.interrupts and all(
            should_auto_approve_interrupt(
                interrupt.value,
                mode,
                allowlist=allowlist,
            )
            for interrupt in result.interrupts
        ):
            payload = build_resume_payload(result.interrupts, {"type": "approve"})
            result = await self.runner.resume(payload, on_message=on_message)
        if not result.interrupts:
            self.turn_settled()
        return result

    def turn_settled(self) -> None:
        """End-of-turn duties once no approval is pending (best-effort).

        Rewrites any report's sidecar transcript so it includes the model's
        closing message (``write_report`` runs mid-turn).
        """
        with contextlib.suppress(Exception):
            self.session.refresh_report_transcripts()

    def maybe_title(self, on_titled: Any = None) -> asyncio.Task[None] | None:
        """After the first turn, upgrade the first-prompt title to an LLM one.

        Fire-and-forget and once per session: the first-prompt title already
        shows in listings, so this only improves it from what the exchange was
        actually about. Runs only when exactly one user message is recorded and
        is wholly best-effort. ``on_titled`` (sync or async) is called with the
        new title so a front end can refresh its display. Returns the task so
        the caller can hold a reference.
        """
        from jutul_agent.trace import schema

        if self.titled:
            return None
        events = self.session.trace.iter_events()
        user_msgs = [e for e in events if e.kind == schema.MESSAGE_USER]
        if len(user_msgs) != 1:
            return None
        self.titled = True
        first_user = str(user_msgs[0].payload.get("content", "")).strip()
        if not first_user:
            return None
        first_reply = next(
            (
                str(e.payload.get("content", "")).strip()
                for e in events
                if e.kind == schema.MESSAGE_ASSISTANT
            ),
            "",
        )
        conversation = f"User: {first_user}\n\nAssistant: {first_reply}"
        return asyncio.create_task(self._retitle(conversation, on_titled))

    async def _retitle(self, conversation: str, on_titled: Any) -> None:
        import inspect

        from jutul_agent.agent.titling import generate_session_title

        title = await generate_session_title(self._model, conversation)
        if not title:
            return
        with contextlib.suppress(Exception):  # session may be closing; never raise here
            self.session.retitle(title)
        if on_titled is not None:
            outcome = on_titled(title)
            if inspect.isawaitable(outcome):
                await outcome

    @property
    def runner(self) -> TurnRunner:
        """The turn runner for this session, built once and reused."""
        if self._runner is None:
            self._runner = TurnRunner(
                self.agent,
                thread_id=self.session.session_id,
                trace=self.session.trace,
            )
        return self._runner

    async def aclose(self) -> None:
        """Tear down the kernel and checkpointer, then close the session."""
        from jutul_agent.session_core import close_session

        stack, self._exit_stack = self._exit_stack, None
        await close_session(self.session, stack, self.warmup_task)

    @classmethod
    async def start(
        cls,
        *,
        simulator: SimulatorAdapter,
        surface: str,
        model: str | None = None,
        session_id: str | None = None,
        resume: bool = False,
        approval_mode: str | None = None,
        workspace: Path | None = None,
        state_root: Path | None = None,
        julia_project: Path | None = None,
        threads: str | None = None,
        add_dirs: Sequence[Path] = (),
        ephemeral_memory: bool = False,
        open_windows: bool = False,
        prepare_env: bool = True,
        discover: bool = True,
        warmup: bool = True,
        virtual_display: bool = True,
        allow_missing_credential: bool = False,
        extensions: Sequence[Capability] = (),
        sysimage: bool | None = None,
    ) -> SessionHost:
        """Stand up a real session: prepare the env, start the kernel, build the agent.

        The one bootstrap for every front end that runs the agent. ``surface``
        names the front end driving the session (``tui``/``cli``/``web``) and
        selects which capabilities and surface-tuned tools apply. The kernel half
        is ``session_core.start_core`` (shared with front ends that bring their own
        agent); this adds the SQLite checkpointer and the agent on top. Both are
        entered on an ``AsyncExitStack`` held by the host, so they stay open until
        ``aclose``.

        ``discover=False`` keeps a run hermetic (the bench must not compose
        ambient entry-point capabilities). ``warmup=False`` skips the background
        Julia warm-up. ``virtual_display=False`` skips the per-session Xvfb.
        ``allow_missing_credential=True`` lets an interactive front end come up
        with no agent when the model's provider key is missing, so the user can
        supply it in-app and rebuild via ``reconfigure``. ``sysimage`` overrides
        the workspace's own on/off setting for this session.
        """

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from jutul_agent.agent.builder import build_agent
        from jutul_agent.credentials import missing_credential
        from jutul_agent.models import resolve_model
        from jutul_agent.paths import workspace_root
        from jutul_agent.session_core import start_core
        from jutul_agent.user_config import load_user_config
        from jutul_agent.workspace import load_workspace_config

        # Model precedence is resolved here, once, for every front end, and before
        # the (possibly minutes-long) env preparation so a bad spec fails fast:
        # explicit request > workspace config > user config > env > default.
        resolved_model = resolve_model(
            model,
            workspace_model=load_workspace_config(workspace or workspace_root()).model,
            user_model=load_user_config().model,
        )
        core = await start_core(
            simulator=simulator,
            surface=surface,
            session_id=session_id,
            resume=resume,
            workspace=workspace,
            state_root=state_root,
            julia_project=julia_project,
            threads=threads,
            ephemeral_memory=ephemeral_memory,
            open_windows=open_windows,
            prepare_env=prepare_env,
            discover=discover,
            virtual_display=virtual_display,
            extensions=extensions,
            sysimage=sysimage,
        )
        session, stack = core.session, core.exit_stack
        try:
            ckpt_path = session.state_dir / "checkpoints.sqlite"
            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(ckpt_path))
            )
            if allow_missing_credential and missing_credential(resolved_model) is not None:
                # The provider key isn't set, so building the model would crash.
                # Come up without an agent; the front end collects the key and
                # rebuilds through ``reconfigure``.
                agent, backend = None, None
            else:
                agent, backend = build_agent(
                    session,
                    model=resolved_model,
                    checkpointer=checkpointer,
                    approval_mode=approval_mode,
                    surface=surface,
                    extensions=core.extensions,
                    added_dirs=add_dirs or None,
                    package_sources=core.package_sources,
                )
        except BaseException:
            await stack.aclose()
            raise

        host = cls(
            session=session,
            agent=agent,
            backend=backend,
            exit_stack=stack,
            checkpointer=checkpointer,
            model=resolved_model,
            approval_mode=approval_mode,
            surface=surface,
            extensions=core.extensions,
            workspace=core.workspace,
            package_sources=core.package_sources,
            add_dirs=add_dirs,
        )
        host._core = core
        if warmup:
            core.start_warmup()
        return host
