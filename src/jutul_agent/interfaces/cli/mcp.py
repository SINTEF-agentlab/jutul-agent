"""``jutul-agent mcp`` subcommand: serve this folder's Julia session to an MCP client.

The client (an IDE or a desktop agent) launches this command and speaks MCP over
its stdin/stdout; see docs/mcp.md. The client brings its own model, so no
provider key is needed here, and none is passed on to Julia.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from jutul_agent.interfaces.cli._helpers import add_sysimage_flags, add_workspace_flags

if TYPE_CHECKING:
    from jutul_agent.interfaces.mcp.runtime import McpLaunch


def build_parser(prog: str = "jutul-agent mcp") -> argparse.ArgumentParser:
    from jutul_agent.julia.threads import THREADS_ENV_VAR
    from jutul_agent.simulators import registry

    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Serve this folder's Julia session to an MCP client (an IDE or desktop "
            "agent) over stdio. The client runs the chat; this runs the tools."
        ),
    )
    parser.add_argument(
        "--sim",
        default=None,
        choices=registry.names(),
        help=(
            "Simulator this folder uses (e.g. jutuldarcy). Defaults to the folder's "
            "saved choice or auto-detection, like `jutul-agent web`."
        ),
    )
    parser.add_argument(
        "--julia-project",
        type=Path,
        default=None,
        help="Override the resolved workspace Julia project.",
    )
    parser.add_argument(
        "--threads",
        default=None,
        metavar="N",
        help=(
            "Julia compute threads: an integer, or 'auto' for all logical cores. "
            f"Precedence: --threads > ${THREADS_ENV_VAR} > default (physical cores "
            "minus one)."
        ),
    )
    parser.add_argument(
        "--viewer-port",
        type=int,
        default=0,
        metavar="PORT",
        help="Loopback port for the plot viewer (default: any free port).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Serve no viewer; tool results name the saved files instead.",
    )
    add_sysimage_flags(parser)
    add_workspace_flags(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    from jutul_agent.interfaces.mcp.server import serve_stdio
    from jutul_agent.paths import set_state_home, set_workspace_root

    # The workspace flags, without loading the user's saved credentials: this
    # server calls no model.
    set_workspace_root(args.workspace)
    set_state_home(args.state_home)
    _drop_provider_keys()

    launch = build_launch(args)
    if launch is None:
        return 2
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve_stdio(launch))
    return 0


def build_launch(args: argparse.Namespace) -> McpLaunch | None:
    """What to serve for these flags, or ``None`` if the folder has no simulator."""
    from jutul_agent.agent.capabilities import discover_extensions
    from jutul_agent.interfaces.cli._helpers import resolve_folder_simulator
    from jutul_agent.interfaces.mcp.runtime import McpLaunch
    from jutul_agent.paths import workspace_root
    from jutul_agent.simulators import registry

    sim = resolve_folder_simulator(args.sim)
    if sim is None:
        return None
    return McpLaunch(
        simulator=registry.get(sim),
        workspace=workspace_root(),
        julia_project=args.julia_project,
        threads=args.threads,
        sysimage=args.sysimage,
        # A project passed in is prepared already, as for every interface.
        prepare_env=args.julia_project is None,
        viewer=not args.no_viewer,
        viewer_port=args.viewer_port,
        capabilities=tuple(discover_extensions()),
    )


def _drop_provider_keys() -> None:
    """Remove model-provider keys from this process's environment.

    Nothing here calls a model, and the Julia kernel inherits this environment,
    so a key inherited from the client's shell would otherwise be readable from
    any code the session runs.
    """
    from jutul_agent.models import PROVIDERS

    for info in PROVIDERS.values():
        if info.key_env_var:
            os.environ.pop(info.key_env_var, None)
