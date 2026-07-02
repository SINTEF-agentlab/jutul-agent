"""``jutul-agent`` (run / TUI / headless turn) subcommand."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from jutul_agent import __version__
from jutul_agent.interfaces.cli._helpers import (
    add_session_flags,
    add_workspace_flags,
    known_packages_map,
    resolve_add_dirs,
)
from jutul_agent.paths import workspace_root
from jutul_agent.session import (
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
    from jutul_agent.agent.approval import parse_approval_mode
    from jutul_agent.agent.builder import resolve_model
    from jutul_agent.credentials import missing_credential
    from jutul_agent.display import can_open_windows
    from jutul_agent.julia.requirements import JuliaRequirementError, require_julia
    from jutul_agent.julia.threads import resolve_compute_threads
    from jutul_agent.juliakernel import JuliaStartupError
    from jutul_agent.session_host import SessionHost
    from jutul_agent.simulators.env_setup import EnvSetupError
    from jutul_agent.user_config import load_user_config

    try:
        require_julia()
    except JuliaRequirementError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    ws = workspace_root()
    julia_project = args.julia_project or resolve_julia_project(ws)
    if args.julia_project is not None and not (julia_project / "Project.toml").exists():
        # An explicit project override is used as-is; the user owns it.
        print(f"error: --julia-project {julia_project} has no Project.toml.", file=sys.stderr)
        return 2

    print(f"Workspace:     {ws}", file=sys.stderr)
    if resume_id:
        print(f"Resuming:      {resume_id}", file=sys.stderr)
    print(f"Julia project: {julia_project}", file=sys.stderr)
    print(
        f"Julia threads: {resolve_compute_threads(args.threads)} compute + 1 interactive",
        file=sys.stderr,
    )
    _warn_if_plotting_unavailable()

    headless = bool(args.prompt)
    model_label = resolve_model(
        args.model, workspace_model=config.model, user_model=load_user_config().model
    )
    env_var = missing_credential(model_label)
    if headless and env_var is not None:
        print(
            f"error: {model_label} needs {env_var}, which isn't set. "
            "Set it (shell env, .env, or `jutul-agent init`) before a "
            "headless `--prompt` run.",
            file=sys.stderr,
        )
        return 1

    approval_mode = parse_approval_mode(args.approval_mode or config.approval_mode)
    try:
        host = await SessionHost.start(
            simulator=adapter,
            surface="cli" if headless else "tui",
            model=args.model,
            session_id=resume_id,
            resume=bool(resume_id),
            approval_mode=approval_mode,
            julia_project=args.julia_project,
            threads=args.threads,
            add_dirs=resolve_add_dirs(args.add_dir, ws),
            ephemeral_memory=args.ephemeral_memory,
            open_windows=can_open_windows(interactive_session=not headless),
            prepare_env=args.julia_project is None,
            allow_missing_credential=not headless,
        )
    except EnvSetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except JuliaStartupError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        print("Run `jutul-agent doctor` to check your setup.", file=sys.stderr)
        return 1

    try:
        write_last_session(host.session.session_id)
        if host.agent is None:
            # The provider key isn't set, so the host came up without a model.
            # The model selector collects the key (or a local model) and rebuilds.
            print(
                f"note: {host.model} needs {env_var}, which isn't set. "
                "Starting without a model. Open the selector with `/model` "
                "to enter the key or pick a local Ollama model.",
                file=sys.stderr,
            )
        if host.backend is not None:
            from jutul_agent.agent.added_dirs import added_dirs

            sources = host.package_sources
            if sources:
                writable = [src.name for src in sources if src.writable]
                summary = f"Packages: {len(sources)} installed (read-only source)"
                if writable:
                    summary += f"; writable dev checkout(s): {', '.join(writable)}"
                print(summary, file=sys.stderr)
            for entry in added_dirs(host.backend):
                print(f"Added folder:  {entry.path}", file=sys.stderr)

        if headless:
            return await _headless_turn(host.agent, host.session, args.prompt)

        from jutul_agent.interfaces.tui import TUIApp

        def build(model_id: str, dirs: Any) -> Any:
            # One rebuild path for every surface: the host carries the
            # checkpointer, session, added folders, and package sources.
            host.reconfigure(model=model_id)
            return host.agent, host.backend

        await TUIApp(
            agent=host.agent,
            session=host.session,
            backend=host.backend,
            model_label=host.model,
            approval_mode=approval_mode,
            warmup_task=host.warmup_task,
            agent_factory=build,
        ).run_async()
        # Review the whole session once, after the TUI exits (cheaper than
        # per-turn, and the natural "we finished" point).
        await _maybe_review(host.session)
        return 0
    finally:
        await host.aclose()


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
