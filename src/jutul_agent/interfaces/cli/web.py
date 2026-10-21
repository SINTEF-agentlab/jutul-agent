"""``jutul-agent`` (or ``jutul-agent web``): run the HTTP + WebSocket server.

Front ends talk to this server to drive sessions; the wire contract is in
docs/server-interface.md. The web stack (FastAPI + uvicorn) ships in the core
install, so there is nothing extra to add.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from typing import TYPE_CHECKING

from jutul_agent import __version__
from jutul_agent.interfaces.cli._helpers import add_session_flags, add_workspace_flags

if TYPE_CHECKING:
    from pathlib import Path

    from jutul_agent.workspace import WorkspaceConfig


def build_parser(prog: str = "jutul-agent web") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run the jutul-agent web UI (HTTP + WebSocket server + browser app). "
            "`jutul-agent` and `jutul-agent web` are the same command and take the same "
            "options."
        ),
    )
    from jutul_agent.simulators import registry

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Address to bind (default 127.0.0.1, localhost only).",
    )
    parser.add_argument("--port", type=int, default=8742, help="Port to bind (default 8742).")
    parser.add_argument(
        "--base-path",
        default="",
        help=(
            "Public URL prefix the browser sees this server under (e.g. /restricted when "
            "an SSO wrapper reverse-proxies the UI). Empty means the server is at /. "
            "API routes still bind at /; generated browser URLs and Bonito proxy_url "
            "include the prefix."
        ),
    )
    parser.add_argument(
        "--sim",
        default=None,
        choices=registry.names(),
        help=(
            "Simulator this folder's sessions use (e.g. jutuldarcy). One folder is "
            "bound to one simulator; defaults to the folder's saved choice or "
            "auto-detection. Use another simulator by serving from another folder."
        ),
    )
    parser.add_argument(
        "--approval-mode",
        choices=["ask", "workspace", "auto"],
        default=None,
        help=(
            "Default human-in-the-loop policy for new sessions: ask (default) prompts "
            "before shell and file edits; workspace auto-allows write_file/edit_file; "
            "auto allows all side-effecting tools. Change it per session in the UI "
            "with /approval-mode."
        ),
    )
    # The same per-session knobs the TUI/run take; here they set the default for
    # every session this server creates (each is one folder bound to one env).
    add_session_flags(parser)
    add_workspace_flags(parser)
    return parser


def _resolve_simulator(args: argparse.Namespace) -> str | None:
    """The one simulator this folder is bound to, or ``None`` if none can be found.

    Same precedence the CLI uses everywhere: the ``--sim`` flag, then the folder's
    saved ``[workspace] simulator``, then auto-detection from its packages. The
    resolved choice is saved back so a later launch here needs no flag. The web UI
    deliberately does not switch simulators in place — a different simulator means
    a different folder, each with its own Julia environment.
    """
    from dataclasses import replace as dc_replace

    from jutul_agent.interfaces.cli._helpers import known_packages_map
    from jutul_agent.paths import workspace_root
    from jutul_agent.simulators import registry
    from jutul_agent.workspace import (
        auto_detect_simulator,
        load_workspace_config,
        write_workspace_config,
    )

    ws = workspace_root()
    config = load_workspace_config(ws)
    sim = args.sim or config.simulator or auto_detect_simulator(known_packages_map(), ws)
    if sim is None:
        print(
            "error: no simulator for this folder. Pass --sim <name>, or run "
            "`jutul-agent init --sim <name>` here first. Known: "
            + ", ".join(registry.names())
            + ".",
            file=sys.stderr,
        )
        return None
    try:
        registry.get(sim)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None
    if config.simulator != sim:  # remember the folder's simulator for next time
        with contextlib.suppress(OSError):
            write_workspace_config(dc_replace(config, simulator=sim), workspace=ws)
    return sim


def _env_versions_ok(args: argparse.Namespace, ws: Path, *, sim: str) -> bool:
    """Refuse a workspace env older than this install supports, before binding.

    Without a usable system image the env is prepared per session, so the same
    check inside that preparation would only fail once the browser is open.
    """
    from jutul_agent.interfaces.cli._helpers import env_versions_ok
    from jutul_agent.simulators import registry

    return env_versions_ok(registry.get(sim), ws, args.julia_project, sim)


def _sysimage_ready(
    args: argparse.Namespace, ws: Path, config: WorkspaceConfig, *, sim: str
) -> bool:
    """Check the system image before binding the port, and explain if it fails.

    Every session this server creates goes through the same check, so a bad
    image would otherwise surface as a session that refuses to start once the
    browser is already open. Checking here puts the explanation in the terminal
    the user launched from. The check inside the session is still the
    authoritative one: it runs after the environment is prepared, when the
    manifest reflects what the session will really load.
    """
    from jutul_agent import sysimage as sysimage_mod
    from jutul_agent.workspace import resolve_julia_project

    project = args.julia_project or resolve_julia_project(ws)
    enabled = sysimage_mod.resolve_enabled(args.sysimage, workspace_enabled=config.sysimage)
    decision = sysimage_mod.decide(ws, project, enabled=enabled)

    # A missing or already-stale image needs no potentially expensive environment
    # preparation to explain it. A currently usable image does: preparation syncs
    # bundled path packages into the managed environment, and that sync can be the
    # very operation that makes the image stale after an upgrade. Do it before the
    # server binds its port so the terminal refuses the launch instead of serving a
    # browser application in which every session creation is doomed to fail.
    if decision.usable and args.julia_project is None:
        from jutul_agent.agent.capabilities import collect_dependency_paths, discover_extensions
        from jutul_agent.simulators import registry
        from jutul_agent.simulators.env_setup import EnvSetupError, prepare_workspace_env

        try:
            prepare_workspace_env(
                registry.get(sim),
                workspace=ws,
                julia_project=project,
                sim_name=sim,
                dependencies=collect_dependency_paths(discover_extensions()),
            )
        except EnvSetupError as exc:
            print(f"error: could not prepare the Julia environment:\n{exc}", file=sys.stderr)
            return False
        decision = sysimage_mod.decide(ws, project, enabled=enabled)

    if decision.blocks:
        print(
            f"\n{sysimage_mod.refusal(decision, command='jutul-agent')}",
            file=sys.stderr,
        )
        return False
    for note in decision.notes:
        print(f"note: {note}", file=sys.stderr)
    if decision.usable:
        print(f"System image: {decision.path}", file=sys.stderr)
    return True


def run(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ModuleNotFoundError:
        # uvicorn ships in the core install, so this only fires on a broken
        # environment; reinstalling restores the web stack.
        print(
            "error: the web stack (uvicorn/fastapi) is missing from this install. "
            "Reinstall jutul-agent, or run `uv sync` from a checkout.",
            file=sys.stderr,
        )
        return 1

    from jutul_agent.interfaces.cli._helpers import apply_workspace_flags

    apply_workspace_flags(args)  # bind the server to its --workspace folder (or the cwd)

    sim = _resolve_simulator(args)
    if sim is None:
        return 2

    from jutul_agent.interfaces.cli._helpers import resolve_add_dirs
    from jutul_agent.interfaces.server.app import create_app
    from jutul_agent.paths import workspace_root
    from jutul_agent.workspace import load_workspace_config

    ws = workspace_root()
    config = load_workspace_config(ws)
    # Defaults for every session this server creates (same precedence as the
    # TUI/run: flag, then the folder's saved config). The UI can still change the
    # model and approval policy per session; the rest are fixed to the folder.
    approval_mode = args.approval_mode or config.approval_mode
    model = args.model or config.model
    add_dirs = resolve_add_dirs(args.add_dir, ws)

    from jutul_agent.interfaces.server.updates import (
        RESTARTED_ENV_VAR,
        Updater,
        is_loopback_host,
        restart_server,
    )

    if os.environ.pop(RESTARTED_ENV_VAR, None) is not None:
        # Restarted by an update from the browser, which is waiting for this server:
        # start, and let the page report anything the update left to fix.
        print("Restarted after updating.", file=sys.stderr)
        if not _sysimage_ready(args, ws, config, sim=sim):
            print("Starting without the system image.", file=sys.stderr)
            args.sysimage = False
    else:
        from jutul_agent.update_check import notify_at_launch

        notify_at_launch()
        if not _env_versions_ok(args, ws, sim=sim) or not _sysimage_ready(
            args, ws, config, sim=sim
        ):
            return 1

    print(
        f"jutul-agent {__version__.split('+')[0]} server on http://{args.host}:{args.port} "
        f"(simulator: {sim}"
        + (f", model: {model}" if model else "")
        + (f", approval: {approval_mode}" if approval_mode else "")
        + (f", +{len(add_dirs)} dir(s)" if add_dirs else "")
        + (", ephemeral memory" if args.ephemeral_memory else "")
        + ")",
        file=sys.stderr,
    )
    updater = Updater(ws, local=is_loopback_host(args.host))
    app = create_app(
        default_sim=sim,
        default_approval_mode=approval_mode,
        default_model=model,
        julia_project=args.julia_project,
        threads=args.threads,
        add_dirs=add_dirs,
        ephemeral_memory=args.ephemeral_memory,
        sysimage=args.sysimage,
        base_path=args.base_path,
        updater=updater,
    )
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, timeout_graceful_shutdown=5)
    )
    updater.bind_server(lambda: setattr(server, "should_exit", True))
    try:
        server.run()
    except KeyboardInterrupt:
        # Ctrl+C: uvicorn re-raises it once it has shut down. Stop here, as
        # uvicorn.run does, without a traceback or a restart.
        return 0
    if not server.started:  # the port was taken, say; uvicorn said why
        return 1
    if updater.restart_requested:
        restart_server()
    return 0
