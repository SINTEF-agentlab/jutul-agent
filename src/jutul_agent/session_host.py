"""One running session, with everything it needs to take a turn.

A ``SessionHost`` owns a ``Session`` (its Julia kernel, trace, and directories),
the agent built for it, and the ``TurnRunner`` that drives a turn. ``start`` is
the single session bootstrap every front end uses: the TUI, the headless CLI,
the web server, and the bench solver all stand up sessions here, so the kernel
environment, capability discovery, and the checkpointer cannot drift between
them. ``aclose`` tears everything down.

The constructor takes a ready-made session and agent so tests can wrap fakes;
``start`` is the production path that stands up a real kernel.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

from jutul_agent.agent.turns import TurnRunner

if TYPE_CHECKING:
    from contextlib import AsyncExitStack
    from pathlib import Path

    from jutul_agent.agent.capabilities import Capability
    from jutul_agent.session import Session
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
        # Background Julia warm-up (load warm package, set GLMakie offscreen); held
        # so it can be cancelled on teardown. Set by ``start``.
        self._warmup_task: Any | None = None

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

    @property
    def model(self) -> str | None:
        """The session's model spec (``provider:model``), or ``None`` for the default."""
        return self._model

    @property
    def approval_mode(self) -> str | None:
        """The session's human-in-the-loop policy, or ``None`` for the default."""
        return self._approval_mode

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
        return self._warmup_task

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
        if self._warmup_task is not None and not self._warmup_task.done():
            self._warmup_task.cancel()
            # Await the cancellation before closing the kernel: warm-up may be mid-eval,
            # whose shielded interrupt/restart recovery would otherwise race the kernel
            # teardown below (the CLI path awaits here too).
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._warmup_task
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        with contextlib.suppress(Exception):
            self.session.finalize()

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
    ) -> SessionHost:
        """Stand up a real session: prepare the env, start the kernel, build the agent.

        The one bootstrap for every front end. ``surface`` names the front end
        driving the session (``tui``/``cli``/``web``) and selects which
        capabilities and surface-tuned tools apply. The Julia kernel and the
        SQLite checkpointer are entered on an ``AsyncExitStack`` held by the
        host, so they stay open until ``aclose``.

        ``discover=False`` keeps a run hermetic (the bench must not compose
        ambient entry-point capabilities). ``warmup=False`` skips the background
        Julia warm-up. ``virtual_display=False`` skips the per-session Xvfb.
        ``allow_missing_credential=True`` lets an interactive front end come up
        with no agent when the model's provider key is missing, so the user can
        supply it in-app and rebuild via ``reconfigure``.
        """

        from contextlib import AsyncExitStack

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        from jutul_agent.agent.builder import build_agent
        from jutul_agent.agent.capabilities import collect_dependency_paths, discover_extensions
        from jutul_agent.credentials import missing_credential
        from jutul_agent.julia.requirements import require_julia
        from jutul_agent.julia.threads import (
            HYPRE_THREADS_ENV_VAR,
            blas_thread_env,
            resolve_compute_threads,
            resolve_hypre_threads,
        )
        from jutul_agent.juliakernel import JuliaKernel, KernelConfig
        from jutul_agent.models import resolve_model
        from jutul_agent.paths import workspace_root
        from jutul_agent.session import Session, default_session_id, session_dir
        from jutul_agent.simulators.env_setup import prepare_workspace_env
        from jutul_agent.user_config import load_user_config
        from jutul_agent.workspace import load_workspace_config, resolve_julia_project

        require_julia()
        ws = workspace or workspace_root()
        project = julia_project or resolve_julia_project(ws)
        # Capabilities are composed before the env is prepared, because their Julia
        # dependencies have to be in the project the kernel starts against. Every
        # capability contributes them, not just the ones this surface ends up
        # offering tools from: the env is per workspace, not per surface, and it
        # would otherwise be re-resolved each time the workspace is opened
        # differently.
        all_extensions = [*discover_extensions(), *extensions] if discover else list(extensions)
        dependency_paths = collect_dependency_paths(all_extensions)
        # Model precedence is resolved here, once, for every front end:
        # explicit request > workspace config > user config > env > default.
        resolved_model = resolve_model(
            model,
            workspace_model=load_workspace_config(ws).model,
            user_model=load_user_config().model,
        )
        # A caller can supply a pre-provisioned env (and skip preparation); the
        # default path prepares the workspace env from the simulator template.
        # Run it off the event loop: the first session for a simulator can spend
        # minutes precompiling, and a blocking call here would freeze the whole
        # server (it serves every session from one event loop) so even the page
        # would stop loading. Threaded, the server stays responsive while the
        # creating request waits.
        if prepare_env:
            import asyncio as _asyncio

            await _asyncio.to_thread(
                prepare_workspace_env,
                simulator,
                workspace=ws,
                julia_project=project,
                sim_name=simulator.name,
                dependencies=dependency_paths,
            )

        sid = session_id or default_session_id()
        sdir = session_dir(sid, state_root=state_root)
        sdir.mkdir(parents=True, exist_ok=True)

        compute_threads = resolve_compute_threads(threads)
        env = {
            **blas_thread_env(compute_threads),
            HYPRE_THREADS_ENV_VAR: str(resolve_hypre_threads()),
        }

        stack = AsyncExitStack()
        try:
            # On a headless box GLMakie needs a virtual display just to load, on
            # every surface (the TUI renders PNGs, the web serves WGLMakie, and
            # both go through the native plotters). Best-effort: without it the
            # session still runs and plotting errors at first use. WGLMakie and
            # Bonito need nothing further here; they are ordinary deps of the
            # workspace env, resolved together with the simulator's own.
            if virtual_display:
                _add_headless_display(stack, env)

            kernel_config = KernelConfig(
                julia_project=project,
                stderr_file=sdir / "julia-startup.log",
                cwd=ws,
                env=env,
                threads=str(compute_threads),
            )
            kernel = await stack.enter_async_context(JuliaKernel(kernel_config))
            # The session's tools go through the wrapper, the warm-up gets the kernel
            # itself. That is what tells the two apart: an eval arriving at the wrapper
            # is real work, and drops whatever is left of the warm-up before taking the
            # kernel, so a user who acts early never waits behind it.
            from jutul_agent.simulators.warmup import YieldsToWork

            julia = YieldsToWork(kernel)
            if resume:
                session = Session.resume(
                    julia=julia,
                    simulator=simulator,
                    session_id=sid,
                    state_root=state_root,
                    ephemeral_memory=ephemeral_memory,
                    open_windows=open_windows,
                    surface=surface,
                )
            else:
                session = Session.create(
                    julia=julia,
                    simulator=simulator,
                    session_id=sid,
                    state_root=state_root,
                    ephemeral_memory=ephemeral_memory,
                    open_windows=open_windows,
                    surface=surface,
                )
            ckpt_path = session.state_dir / "checkpoints.sqlite"
            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(ckpt_path))
            )
            # Resolve the env's package source dirs (one fast, no-compile call): it
            # gives the agent the simulator's source path up front, so it needn't
            # `using <Sim>; pkgdir(<Sim>)` to find it, and guards the read-only depot.
            from jutul_agent.agent.builder import resolve_package_sources

            package_sources = await asyncio.to_thread(resolve_package_sources, project)
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
                    extensions=all_extensions,
                    added_dirs=add_dirs or None,
                    package_sources=package_sources,
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
            extensions=all_extensions,
            workspace=ws,
            package_sources=package_sources,
            add_dirs=add_dirs,
        )
        if warmup:
            # Warm the kernel in the background: load the warm packages and set
            # GLMakie offscreen so a native plotter can't pop an OS window on a
            # machine with a display. Best-effort and cancelled on teardown.
            #
            # Every discovered capability's package is loaded, not just this
            # surface's, for the same reason their Julia dependencies are: the
            # kernel is per workspace, and a package left unloaded here would be
            # loaded by the first tool call that needs it, at the user's expense.
            from jutul_agent.agent.capabilities import collect_warm_code, collect_warm_packages
            from jutul_agent.simulators.warmup import start_warmup

            # Set once the warm-up is past loading; before that, cancelling it
            # would interrupt a `using` and take the kernel down with it.
            abandonable = asyncio.Event()
            host._warmup_task = start_warmup(
                kernel,
                simulator.warm_package,
                collect_warm_packages(all_extensions),
                collect_warm_code(all_extensions),
                abandonable=abandonable,
            )
            julia.set_warmup(host._warmup_task, abandonable)
        return host


def _add_headless_display(stack: AsyncExitStack, env: dict[str, str]) -> None:
    """Start an Xvfb for this session (headless boxes) and set ``DISPLAY`` in ``env``.

    GLMakie needs a GL context just to load, which is what makes the native
    plotters' methods available for WGLMakie to render. On a machine with a real
    display this is a no-op; the Xvfb is tied to the session's exit stack.
    """
    from jutul_agent.display import managed_display, should_wrap_xvfb

    if not should_wrap_xvfb():
        return
    try:
        display = stack.enter_context(managed_display())
    except Exception as exc:  # missing/slow Xvfb must not break the session
        print(f"warning: could not start a virtual display for plotting ({exc}).", file=sys.stderr)
        return
    env["DISPLAY"] = display
