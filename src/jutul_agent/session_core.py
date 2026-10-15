"""A session's computational core: the prepared env, the Julia kernel, and the
``Session`` around it, with no agent.

``start_core`` is the kernel half of the session bootstrap. ``SessionHost.start``
builds the agent on top of a core; a front end that brings its own agent (the
MCP server, whose host application runs the model) uses a core on its own. Both
therefore prepare the env, pick the system image, start the virtual display and
the kernel, and warm it the same way, so none of that can drift between them.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextlib import AsyncExitStack
    from pathlib import Path

    from jutul_agent.agent.capabilities import Capability
    from jutul_agent.session import Session
    from jutul_agent.simulators.base import SimulatorAdapter
    from jutul_agent.workspace import WorkspaceConfig


@dataclass
class SessionCore:
    """A started session's kernel and ``Session``, ready for tools to use.

    ``kernel`` is the raw ``JuliaKernel``; ``session.julia`` wraps it so real
    work pre-empts the background warm-up. ``extensions`` is every capability
    composed for the session (discovered plus passed in), ``package_sources``
    the env's resolved package directories. The kernel and anything else the
    core started live on ``exit_stack``; ``aclose`` tears them down.
    """

    session: Session
    kernel: Any
    workspace: Path
    julia_project: Path
    workspace_config: WorkspaceConfig
    extensions: list[Capability]
    package_sources: list[Any]
    exit_stack: AsyncExitStack
    warmup_task: Any | None = None
    _warm_started: bool = field(default=False, init=False, repr=False)

    def start_warmup(self) -> None:
        """Warm the kernel in the background (idempotent), and again after each restart.

        Loads the warm packages and sets GLMakie offscreen, so a native plotter
        can't pop an OS window on a machine with a display. Best-effort and
        cancelled on teardown.

        Every capability's package is loaded, not just the ones whose tools this
        surface offers, for the same reason their Julia dependencies are in the
        env: the kernel is per workspace, and a package left unloaded here would
        be loaded by the first tool call that needs it, at the user's expense.

        A restart (``reset_julia``, or a cancelled call that ignored its
        interrupt) replaces the process, and with it all the warm-up did: the
        packages, HYPRE's thread count, the ``using GLMakie`` that plotting code
        calls ``Figure`` through. So the fresh process is warmed the same way,
        with other calls held until it is loaded, and a restart returns the
        session to how it started, not to a bare Julia.
        """
        if self._warm_started:
            return
        self._warm_started = True
        self._warm()
        on_restart = getattr(self.kernel, "on_restart", None)
        if on_restart is not None:
            # The stack unwinds in reverse, so this stops listening before the
            # kernel closes: a restart during teardown can't start a warm-up.
            self.exit_stack.callback(on_restart(self._rewarm))

    def _warm(self) -> asyncio.Event:
        from jutul_agent.agent.capabilities import collect_warm_code, collect_warm_packages
        from jutul_agent.simulators.warmup import start_warmup

        # Set once the warm-up is past loading; before that, cancelling it would
        # interrupt a `using` and take the kernel down with it.
        abandonable = asyncio.Event()
        self.warmup_task = start_warmup(
            self.kernel,
            self.session.simulator.warm_package,
            collect_warm_packages(self.extensions),
            collect_warm_code(self.extensions),
            abandonable=abandonable,
        )
        self.session.julia.set_warmup(self.warmup_task, abandonable)
        return abandonable

    def _rewarm(self, _restart: object) -> asyncio.Future[None] | None:
        # What is left of the last warm-up was meant for the process the restart
        # replaced, and would otherwise run its remaining steps on the new one.
        stale = self.warmup_task
        if stale is not None and not stale.done():
            stale.cancel()
        abandonable = self._warm()
        if self.warmup_task is None:
            return None
        # The kernel holds other calls until this is done, even one queued through
        # the restart, which would otherwise run first, on a bare process.
        return asyncio.ensure_future(_loaded(self.warmup_task, abandonable))

    async def aclose(self) -> None:
        """Stop the warm-up, close the kernel and the rest of the stack, finalize."""
        await close_session(self.session, self.exit_stack, self.warmup_task)


async def _loaded(warmup: asyncio.Task[None], abandonable: asyncio.Event) -> None:
    """Return once ``warmup`` is past loading, or has ended without getting there."""
    loaded = asyncio.ensure_future(abandonable.wait())
    try:
        await asyncio.wait((warmup, loaded), return_when=asyncio.FIRST_COMPLETED)
    finally:
        loaded.cancel()


async def close_session(
    session: Session, exit_stack: AsyncExitStack | None, warmup_task: Any | None
) -> None:
    """Tear down what ``start_core`` stood up, in the one safe order.

    The warm-up is cancelled and awaited before the kernel closes: it may be
    mid-eval, and its shielded interrupt/restart recovery would otherwise race
    the kernel teardown.
    """
    if warmup_task is not None and not warmup_task.done():
        warmup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await warmup_task
    if exit_stack is not None:
        await exit_stack.aclose()
    with contextlib.suppress(Exception):
        session.finalize()


async def start_core(
    *,
    simulator: SimulatorAdapter,
    surface: str,
    session_id: str | None = None,
    resume: bool = False,
    workspace: Path | None = None,
    state_root: Path | None = None,
    julia_project: Path | None = None,
    threads: str | None = None,
    ephemeral_memory: bool = False,
    open_windows: bool = False,
    prepare_env: bool = True,
    discover: bool = True,
    virtual_display: bool = True,
    extensions: Sequence[Capability] = (),
    sysimage: bool | None = None,
) -> SessionCore:
    """Stand up a session's kernel: prepare the env, start Julia, open the session.

    ``surface`` names the front end driving the session; it is recorded on the
    ``Session`` (tools read it) and names the command a sysimage refusal points
    at. The kernel is entered on the returned core's ``exit_stack``, so it stays
    up until ``SessionCore.aclose`` (or the owner closes the stack). The warm-up
    is not started here: call ``start_warmup`` once the caller is ready.

    ``discover=False`` keeps a run hermetic (no ambient entry-point
    capabilities). ``virtual_display=False`` skips the per-session Xvfb.
    ``sysimage`` overrides the workspace's own on/off setting.
    """
    from contextlib import AsyncExitStack

    from jutul_agent import sysimage as sysimage_mod
    from jutul_agent.agent.capabilities import collect_dependency_paths, discover_extensions
    from jutul_agent.julia.requirements import require_julia
    from jutul_agent.julia.threads import (
        HYPRE_THREADS_ENV_VAR,
        blas_thread_env,
        resolve_compute_threads,
        resolve_hypre_threads,
    )
    from jutul_agent.juliakernel import JuliaKernel, KernelConfig
    from jutul_agent.paths import workspace_root
    from jutul_agent.session import Session, default_session_id, session_dir
    from jutul_agent.simulators.env_setup import prepare_workspace_env
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
    workspace_config = load_workspace_config(ws)
    # A caller can supply a pre-provisioned env (and skip preparation); the
    # default path prepares the workspace env from the simulator template.
    # Run it off the event loop: the first session for a simulator can spend
    # minutes precompiling, and a blocking call here would freeze the whole
    # server (it serves every session from one event loop) so even the page
    # would stop loading. Threaded, the server stays responsive while the
    # creating request waits.
    if prepare_env:
        await asyncio.to_thread(
            prepare_workspace_env,
            simulator,
            workspace=ws,
            julia_project=project,
            sim_name=simulator.name,
            dependencies=dependency_paths,
        )

    # After the env is prepared, never before: preparation is what re-copies the
    # in-env JutulAgent runtime after an upgrade and installs anything a
    # capability added, so asking earlier would compare the image against an
    # environment that is about to change.
    sysimage_decision = sysimage_mod.decide(
        ws,
        project,
        enabled=sysimage_mod.resolve_enabled(sysimage, workspace_enabled=workspace_config.sysimage),
    )
    if sysimage_decision.blocks:
        raise sysimage_mod.SysimageUnavailable(
            sysimage_decision,
            command=sysimage_mod.SURFACE_COMMANDS.get(surface, "jutul-agent web"),
        )
    for note in sysimage_decision.notes:
        print(f"note: {note}", file=sys.stderr)

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
            sysimage=sysimage_decision.path,
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
        open_session = Session.resume if resume else Session.create
        session = open_session(
            julia=julia,
            simulator=simulator,
            session_id=sid,
            state_root=state_root,
            ephemeral_memory=ephemeral_memory,
            open_windows=open_windows,
            surface=surface,
        )
        # Resolve the env's package source dirs (one fast, no-compile call): it
        # gives the agent the simulator's source path up front, so it needn't
        # `using <Sim>; pkgdir(<Sim>)` to find it, and guards the read-only depot.
        from jutul_agent.agent.builder import resolve_package_sources

        package_sources = await asyncio.to_thread(resolve_package_sources, project)
    except BaseException:
        await stack.aclose()
        raise

    return SessionCore(
        session=session,
        kernel=kernel,
        workspace=ws,
        julia_project=project,
        workspace_config=workspace_config,
        extensions=all_extensions,
        package_sources=list(package_sources),
        exit_stack=stack,
    )


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
