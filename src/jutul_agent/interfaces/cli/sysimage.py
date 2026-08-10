"""``jutul-agent sysimage`` subcommand: build, inspect and remove a system image.

Building one is the slowest thing jutul-agent does, and the folder that has one
is expected to keep it. So the two commands that change what is on disk also
change the folder's setting to match: a successful build turns the image on, and
clearing it turns it off. Leaving those apart is how a workspace ends up either
refusing to start over an image it no longer has, or quietly taking the slow path
over an image it does.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

from jutul_agent.interfaces.cli._helpers import add_workspace_flags, known_packages_map
from jutul_agent.paths import workspace_root
from jutul_agent.simulators import registry
from jutul_agent.workspace import (
    auto_detect_simulator,
    load_workspace_config,
    resolve_julia_project,
    write_workspace_config,
)

ACTIONS = ("build", "status", "clear")


def build_parser(prog: str = "jutul-agent sysimage") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Manage this folder's Julia system image: every package it uses, "
            "pre-linked into one file the session starts from."
        ),
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="status",
        choices=ACTIONS,
        help="build a new image, show the current one (default), or remove it.",
    )
    parser.add_argument(
        "--sim",
        choices=registry.names(),
        default=None,
        help="Simulator to build for. Defaults to the folder's own.",
    )
    parser.add_argument(
        "--cpu-target",
        default="native",
        help=(
            "CPU the image is compiled for. The default, 'native', is the fastest "
            "code for this machine and will not run on another. Pass a portable "
            "target (e.g. 'generic') to build an image for other hardware."
        ),
    )
    add_workspace_flags(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    from jutul_agent import sysimage as sysimage_mod
    from jutul_agent import sysimage_build

    ws = workspace_root()
    config = load_workspace_config(ws)
    project = resolve_julia_project(ws)

    if args.action == "status":
        print(sysimage_build.describe(ws, project))
        return 0

    if args.action == "clear":
        print(
            "Removed the system image." if sysimage_mod.clear(ws) else "No system image to remove."
        )
        # Turned off in the same breath. A folder still set to use an image it no
        # longer has would refuse every launch until someone worked out why.
        if config.sysimage:
            write_workspace_config(dc_replace(config, sysimage=False), workspace=ws)
            print("This folder no longer starts from one.")
        return 0

    return _build(args, ws, project, config)


def _build(args: argparse.Namespace, ws: Path, project: Path, config) -> int:
    from jutul_agent import sysimage_build
    from jutul_agent.julia.requirements import JuliaRequirementError, require_julia
    from jutul_agent.simulators.env_setup import EnvSetupError

    try:
        require_julia()
    except JuliaRequirementError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sim_name = args.sim or config.simulator or auto_detect_simulator(known_packages_map(), ws)
    if sim_name is None:
        print(
            "error: no simulator for this folder. Pass --sim <name>, or run "
            "`jutul-agent init --sim <name>` here first. Known: "
            + ", ".join(registry.names())
            + ".",
            file=sys.stderr,
        )
        return 2
    try:
        adapter = registry.get(sim_name)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        prepare_environment(adapter, workspace=ws, julia_project=project)
        result = sysimage_build.build(
            workspace=ws, julia_project=project, cpu_target=args.cpu_target
        )
    except (EnvSetupError, sysimage_build.SysimageBuildError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Nothing was installed: the image is built under a temporary name and
        # only moved into place once it has passed verification.
        print("\nBuild interrupted; the existing image (if any) is untouched.", file=sys.stderr)
        return 130

    print(f"\nSystem image built in {result.seconds / 60:.0f} min: {result.path}")
    print(
        f"  contains: {result.contained} packages "
        f"({len(result.packages)} direct dependencies and everything they load)"
    )
    if config.sysimage is not True:
        write_workspace_config(dc_replace(config, sysimage=True), workspace=ws)
        print("  this folder now starts from it; `--no-sysimage` skips it for one run.")
    return 0


def prepare_environment(adapter, *, workspace: Path, julia_project: Path) -> None:
    """Bring the environment to exactly what a session would run against.

    Capabilities are composed here for the same reason a session composes them
    before preparing its environment: their Julia dependencies belong to the
    project, and an image built before they are installed is missing precisely
    the packages an extended workspace exists to use.
    """

    from jutul_agent.agent.capabilities import collect_dependency_paths, discover_extensions
    from jutul_agent.simulators.env_setup import prepare_workspace_env

    prepare_workspace_env(
        adapter,
        workspace=workspace,
        julia_project=julia_project,
        sim_name=adapter.name,
        dependencies=collect_dependency_paths(discover_extensions()),
    )
