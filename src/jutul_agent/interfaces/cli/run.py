"""``jutul-agent`` (run / TUI / headless turn) subcommand."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jutul_agent import __version__
from jutul_agent.interfaces.cli._helpers import (
    add_session_flags,
    add_workspace_flags,
    known_packages_map,
    resolve_add_dirs,
)
from jutul_agent.paths import workspace_root
from jutul_agent.session import (
    default_session_id,
    list_sessions,
    read_last_session,
    resolve_session_id,
    session_dir,
    write_last_session,
)
from jutul_agent.simulators import registry
from jutul_agent.workspace import (
    auto_detect_simulator,
    load_workspace_config,
    resolve_julia_project,
)

if TYPE_CHECKING:
    # Imported for typing only; the runtime import stays inside _run_session so the
    # capability entry points are not scanned on an unrelated subcommand.
    from jutul_agent.agent.capabilities import Capability


def build_parser(prog: str = "jutul-agent tui") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Specialized scientific agent for AD-enabled simulators built on the Jutul framework."
        ),
        epilog=(
            "Interfaces: `jutul-agent web` (browser), `jutul-agent tui` (terminal), "
            '`jutul-agent run "<prompt>"` (one-shot). Setup: '
            "`jutul-agent init|setup [--sim <name>]`. Other commands: `doctor`, "
            "`upgrade`, `transcript [<id>]`, `sessions`, `review`, `eval`."
        ),
    )
    parser.add_argument("--version", action="version", version=f"jutul-agent {__version__}")
    parser.add_argument(
        "--sim",
        choices=registry.names(),
        required=False,
        help="Active simulator. Required if not set in workspace config and not auto-detectable.",
    )
    add_session_flags(parser)
    add_workspace_flags(parser)
    parser.add_argument(
        "--continue",
        dest="continue_last",
        action="store_true",
        help="Continue the most recent session in this workspace.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="SESSION",
        help=(
            "Resume an earlier session by id (or unique prefix). With no "
            "value, pick from a list of recent sessions."
        ),
    )
    parser.add_argument(
        "--approval-mode",
        choices=["ask", "workspace", "auto"],
        default=None,
        help=(
            "Human-in-the-loop policy: ask (default) prompts before shell and "
            "file edits; workspace auto-allows write_file/edit_file; auto "
            "allows all side-effecting tools."
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt for a single headless turn (used by `jutul-agent run`).",
    )
    return parser


def dispatch(args: argparse.Namespace) -> int:
    import asyncio

    from jutul_agent.update_check import notify_at_launch

    notify_at_launch()

    ws = workspace_root()
    config = load_workspace_config(ws)
    sim_name = args.sim or config.simulator or auto_detect_simulator(known_packages_map(), ws)
    if sim_name is None:
        print(
            "error: --sim is required (or set [workspace].simulator in "
            ".jutul-agent/config.toml). Known: " + ", ".join(registry.names()) + ".",
            file=sys.stderr,
        )
        return 2

    try:
        adapter = registry.get(sim_name)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        resume_id = _resolve_resume_id(args)
    except _ResumeCancelled:
        return 0
    except _ResumeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        return asyncio.run(_run_session(args, adapter, config, resume_id=resume_id))
    except KeyboardInterrupt:
        # Ctrl+C during the synchronous startup (Julia kernel, env bootstrap,
        # warm-up), before the TUI takes over input. Exit cleanly instead of
        # dumping a traceback.
        print("\nStartup interrupted.", file=sys.stderr)
        return 130


class _ResumeError(Exception):
    """A --continue/--resume request that cannot be satisfied."""


class _ResumeCancelled(Exception):
    """The user declined to pick a session from the resume list."""


def _resolve_resume_id(args: argparse.Namespace) -> str | None:
    """The session id to resume, or ``None`` for a fresh session."""
    if args.continue_last and args.resume is not None:
        raise _ResumeError("--continue and --resume are mutually exclusive.")

    if args.continue_last:
        sid = read_last_session()
        if sid is None or not (session_dir(sid) / "trace.sqlite").exists():
            raise _ResumeError("no previous session found in this workspace.")
        return sid

    if args.resume is None:
        return None
    if args.resume:
        sid = resolve_session_id(args.resume)
        if sid is None:
            raise _ResumeError(
                f"no unique session matches {args.resume!r}. "
                "Run `jutul-agent sessions` to list them."
            )
        return sid
    return _pick_session()


def _pick_session(limit: int = 15) -> str:
    """Interactive resume picker: list recent sessions, read one choice."""
    from jutul_agent.interfaces.cli.sessions import format_session_line

    sessions = list_sessions()[:limit]
    if not sessions:
        raise _ResumeError("no previous sessions found in this workspace.")
    if not sys.stdin.isatty():
        raise _ResumeError("--resume needs a session id when stdin is not a terminal.")

    print("Recent sessions:", file=sys.stderr)
    for index, info in enumerate(sessions, start=1):
        print(f"  {index:2}. {format_session_line(info)}", file=sys.stderr)
    try:
        answer = input("Resume which session? [number, or Enter to cancel] ").strip()
    except (EOFError, KeyboardInterrupt):
        raise _ResumeCancelled() from None
    if not answer:
        raise _ResumeCancelled()
    try:
        index = int(answer)
    except ValueError:
        sid = resolve_session_id(answer)
        if sid is None:
            raise _ResumeError(f"no unique session matches {answer!r}.") from None
        return sid
    if not 1 <= index <= len(sessions):
        raise _ResumeError(f"pick a number between 1 and {len(sessions)}.")
    return sessions[index - 1].session_id


async def _run_session(
    args: argparse.Namespace,
    adapter: Any,
    config: Any,
    *,
    resume_id: str | None = None,
) -> int:
    from jutul_agent.agent.capabilities import collect_dependency_paths, discover_extensions
    from jutul_agent.julia.requirements import JuliaRequirementError, require_julia
    from jutul_agent.juliakernel import JuliaStartupError, KernelConfig
    from jutul_agent.simulators.env_setup import EnvSetupError, prepare_workspace_env

    try:
        require_julia()
    except JuliaRequirementError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ws = workspace_root()
    julia_project = args.julia_project or resolve_julia_project(ws)
    extensions = discover_extensions()

    if args.julia_project is not None:
        # An explicit project override is used as-is; the user owns it.
        if not (julia_project / "Project.toml").exists():
            print(
                f"error: --julia-project {julia_project} has no Project.toml.",
                file=sys.stderr,
            )
            return 2
    else:
        try:
            prepare_workspace_env(
                adapter,
                workspace=ws,
                julia_project=julia_project,
                sim_name=args.sim or config.simulator,
                dependencies=collect_dependency_paths(extensions),
            )
        except EnvSetupError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    session_id = resume_id or default_session_id()
    state_dir = session_dir(session_id)
    state_dir.mkdir(parents=True, exist_ok=True)

    from jutul_agent.julia.threads import (
        HYPRE_THREADS_ENV_VAR,
        blas_thread_env,
        resolve_compute_threads,
        resolve_hypre_threads,
    )

    compute_threads = resolve_compute_threads(args.threads)

    print(f"Workspace:     {ws}", file=sys.stderr)
    if resume_id:
        print(f"Resuming:      {session_id}", file=sys.stderr)
    print(f"Julia project: {julia_project}", file=sys.stderr)
    print(f"Julia threads: {compute_threads} compute + 1 interactive", file=sys.stderr)
    _warn_if_plotting_unavailable()

    # On headless Linux, plotting needs a virtual display. We manage Xvfb directly
    # rather than via `xvfb-run`, whose `2>&1` would merge the stdout/stderr the
    # kernel keeps on separate pipes. The Julia process inherits it through DISPLAY.
    from contextlib import ExitStack

    with ExitStack() as display_stack:
        kernel_env = _open_headless_display(display_stack) or {}
        # Reserve BLAS so N compute threads don't oversubscribe (sparse solves don't
        # need threaded BLAS); merged over any DISPLAY the headless path set.
        kernel_env = {**kernel_env, **blas_thread_env(compute_threads)}
        # HYPRE's own (OpenMP) thread count; read by the warm-up snippet.
        kernel_env[HYPRE_THREADS_ENV_VAR] = str(resolve_hypre_threads())
        kernel_config = KernelConfig(
            julia_project=julia_project,
            stderr_file=state_dir / "julia-startup.log",
            cwd=ws,
            env=kernel_env or None,
            threads=str(compute_threads),
        )
        try:
            return await _run_with_backend(
                kernel_config,
                args,
                adapter,
                config,
                session_id,
                state_dir,
                extensions=extensions,
                resuming=bool(resume_id),
            )
        except JuliaStartupError as exc:
            print(f"\nerror: {exc}", file=sys.stderr)
            print("Run `jutul-agent doctor` to check your setup.", file=sys.stderr)
            return 1


def _open_headless_display(stack: Any) -> dict[str, str] | None:
    """Start a virtual display for headless plotting, returning ``{DISPLAY: ...}``.

    Returns ``None`` when no virtual display is needed (a real display is present,
    non-Linux, or the user opted out) or when Xvfb can't start; the Julia process
    then simply inherits the ambient environment.
    """

    from jutul_agent.display import managed_display, should_wrap_xvfb

    if not should_wrap_xvfb():
        return None
    try:
        display = stack.enter_context(managed_display())
    except Exception as exc:  # Xvfb missing or slow to start; don't block the session.
        print(
            f"warning: could not start a virtual display for plotting ({exc}); "
            "GLMakie plotting will be unavailable. Run `jutul-agent doctor` for help.",
            file=sys.stderr,
        )
        return None
    return {"DISPLAY": display}


def _warn_if_plotting_unavailable() -> None:
    """One-line heads-up at launch when GLMakie has no display here.

    On headless Linux without ``xvfb-run`` (or with it opted out), the native
    plotters can't render and ``plot_julia`` errors at use-time; but simulation,
    eval, and the file tools all still work, so this is a warning, not a failure.
    Surfacing it at launch means the user learns before their first plot, not
    mid-session when a ``plot_reservoir`` call fails.
    """

    from jutul_agent.display import (
        plotting_display_available,
        xvfb_opted_out,
    )

    if plotting_display_available():
        return
    hint = (
        "unset JUTUL_AGENT_NO_XVFB and install xvfb"
        if xvfb_opted_out()
        else "install xvfb (e.g. `sudo apt-get install -y xvfb`)"
    )
    print(
        "warning: no display and xvfb not available, so plotting (GLMakie) is "
        f"unavailable; simulation still works. To enable plots, {hint}. "
        "Run `jutul-agent doctor` for details.",
        file=sys.stderr,
    )


async def _run_with_backend(
    kernel_config: Any,
    args: argparse.Namespace,
    adapter: Any,
    config: Any,
    session_id: str,
    state_dir: Path,
    *,
    extensions: Sequence[Capability] = (),
    resuming: bool = False,
) -> int:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    from jutul_agent.agent.added_dirs import added_dirs
    from jutul_agent.agent.approval import parse_approval_mode
    from jutul_agent.agent.builder import build_agent, resolve_model, resolve_package_sources
    from jutul_agent.display import can_open_windows
    from jutul_agent.juliakernel import JuliaKernel
    from jutul_agent.session import Session
    from jutul_agent.user_config import load_user_config

    # A live window is shown only for an interactive session with a display; a
    # one-shot `--prompt` run and any headless box render offscreen to a PNG.
    open_windows = can_open_windows(interactive_session=not args.prompt)

    async with JuliaKernel(kernel_config) as julia:
        if resuming:
            session = Session.resume(
                julia=julia,
                simulator=adapter,
                session_id=session_id,
                ephemeral_memory=args.ephemeral_memory,
                open_windows=open_windows,
            )
        else:
            session = Session.create(
                julia=julia,
                simulator=adapter,
                session_id=session_id,
                ephemeral_memory=args.ephemeral_memory,
                open_windows=open_windows,
            )
        write_last_session(session.session_id)
        warmup_task = _start_warmup(julia, adapter.warm_package)
        try:
            ckpt_path = session.state_dir / "checkpoints.sqlite"
            async with AsyncSqliteSaver.from_conn_string(str(ckpt_path)) as checkpointer:
                user_config = load_user_config()
                model_label = resolve_model(
                    args.model,
                    workspace_model=config.model,
                    user_model=user_config.model,
                )
                approval_mode = parse_approval_mode(args.approval_mode or config.approval_mode)
                package_sources = await asyncio.to_thread(
                    resolve_package_sources, kernel_config.julia_project
                )
                extra_dirs = resolve_add_dirs(args.add_dir, kernel_config.cwd)

                def build(model_id: str, dirs: Any) -> Any:
                    # Rebuilds with the same checkpointer/session so the TUI can
                    # switch models without losing the conversation; ``dirs`` keeps
                    # any /add-dir folders across the rebuild.
                    return build_agent(
                        session,
                        model=model_id,
                        checkpointer=checkpointer,
                        approval_mode=approval_mode,
                        package_sources=package_sources,
                        added_dirs=dirs,
                        extensions=extensions,
                    )

                from jutul_agent.credentials import missing_credential

                env_var = missing_credential(model_label)
                if env_var is not None:
                    # The provider key isn't set, so building the model would
                    # crash. Launch without an agent instead: the user reaches
                    # the model selector to paste the key (or pick a local Ollama
                    # model that needs none), and selecting one rebuilds the agent.
                    agent, backend = None, None
                    print(
                        f"note: {model_label} needs {env_var}, which isn't set. "
                        "Starting without a model. Open the selector with `/model` "
                        "to enter the key or pick a local Ollama model.",
                        file=sys.stderr,
                    )
                else:
                    agent, backend = build(model_label, extra_dirs)
                if backend is not None:
                    if package_sources:
                        writable = [src.name for src in package_sources if src.writable]
                        summary = f"Packages: {len(package_sources)} installed (read-only source)"
                        if writable:
                            summary += f"; writable dev checkout(s): {', '.join(writable)}"
                        print(summary, file=sys.stderr)
                    for entry in added_dirs(backend):
                        print(f"Added folder:  {entry.path}", file=sys.stderr)
                if args.prompt:
                    if agent is None:
                        print(
                            f"error: {model_label} needs {env_var}, which isn't set. "
                            "Set it (shell env, .env, or `jutul-agent init`) before a "
                            "headless `--prompt` run.",
                            file=sys.stderr,
                        )
                        return 1
                    return await _headless_turn(agent, session, args.prompt)
                from jutul_agent.interfaces.tui import TUIApp

                await TUIApp(
                    agent=agent,
                    session=session,
                    backend=backend,
                    model_label=model_label,
                    approval_mode=approval_mode,
                    warmup_task=warmup_task,
                    agent_factory=build,
                ).run_async()
                # Review the whole session once, after the TUI exits (cheaper than
                # per-turn, and the natural "we finished" point).
                await _maybe_review(session)
        finally:
            if warmup_task is not None and not warmup_task.done():
                warmup_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await warmup_task
            session.finalize()
    return 0


def _start_warmup(julia: Any, warm_package: str) -> asyncio.Task[Any] | None:
    """Background warm-up; the logic is shared with the server in ``simulators.warmup``."""
    from jutul_agent.simulators.warmup import start_warmup

    return start_warmup(julia, warm_package)


async def _headless_turn(agent: Any, session: Any, prompt: str) -> int:
    from jutul_agent.agent.turns import TurnRunner

    session.adopt_title(prompt)
    runner = TurnRunner(agent, thread_id=session.session_id, trace=session.trace)
    result = await runner.run_prompt(prompt)
    if result.interrupts:
        print(
            "error: this turn paused for approval, but headless mode can't prompt for it yet.\n"
            "       Re-run with `--approval-mode auto` to let the agent run tools without "
            "approval,\n"
            "       or launch the interactive TUI (`jutul-agent`) to approve steps as "
            "they come up.",
            file=sys.stderr,
        )
        print(f"\n[session {session.session_id}]", file=sys.stderr)
        return 3

    _print_final_message(result.messages)
    await _maybe_review(session)
    print(f"\n[session {session.session_id}]", file=sys.stderr)
    return 0


async def _maybe_review(session: Any) -> None:
    """Run the session reviewer when review mode is on (best-effort, dev-only)."""
    from jutul_agent.review import maybe_review_session, review_enabled

    if not review_enabled():
        return
    print("Reviewing the session…", file=sys.stderr)
    report = await maybe_review_session(session)
    if report is None:
        return
    n = len(report.findings)
    print(
        f"Review: {n} finding{'s' if n != 1 else ''}; see `jutul-agent review`.",
        file=sys.stderr,
    )


def _print_final_message(messages: list[Any]) -> None:
    from jutul_agent.agent.turns import final_assistant_text

    text = final_assistant_text(messages)
    if text:
        print(text)
