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
    from jutul_agent.julia.requirements import JuliaRequirementError, require_julia

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

    return 0 if build_for_workspace(adapter, ws, project, config, cpu_target=args.cpu_target) else 1


def build_for_workspace(
    adapter,
    ws: Path,
    project: Path,
    config,
    *,
    cpu_target: str = "native",
    skip_current: bool = False,
) -> bool:
    """Prepare the env, build and install the image, and record the folder's opt-in.

    The one build path, shared by the explicit ``sysimage build`` and by ``init
    --sysimage``. ``skip_current`` is for callers like ``init`` that build as one
    step of a larger command: an image that still matches the environment just
    prepared is left alone rather than rebuilt identically, since a rebuild costs
    tens of minutes and changes nothing. The explicit command never skips: asking
    for a build gets one.
    """

    from jutul_agent import sysimage as sysimage_mod
    from jutul_agent import sysimage_build
    from jutul_agent.agent.capabilities import collect_warm_code, discover_extensions
    from jutul_agent.simulators.env_setup import EnvSetupError

    result = None
    try:
        # Discovered once, for both halves of the build: the same capabilities
        # whose Julia dependencies the environment is prepared with also carry
        # the warm-up code the image bakes.
        extensions = discover_extensions()
        prepare_environment(adapter, workspace=ws, julia_project=project, extensions=extensions)
        current = (
            skip_current
            and sysimage_mod.decide(ws, project, enabled=True).status == sysimage_mod.CURRENT
        )
        if current:
            print("\nSystem image already matches this environment; skipping the rebuild.")
        else:
            result = sysimage_build.build(
                workspace=ws,
                julia_project=project,
                cpu_target=cpu_target,
                warmup_code=collect_warm_code(extensions),
            )
    except (EnvSetupError, sysimage_build.SysimageBuildError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return False
    except KeyboardInterrupt:
        # Nothing was installed: the image is built under a temporary name and
        # only moved into place once it has passed verification.
        print("\nBuild interrupted; the existing image (if any) is untouched.", file=sys.stderr)
        return False

    if result is not None:
        print(f"\nSystem image built in {result.seconds / 60:.0f} min: {result.path}")
        print(
            f"  contains: {result.contained} packages "
            f"({len(result.packages)} direct dependencies and everything they load)"
        )
        if result.warmup_failures:
            # Said again, at the end, where it cannot be missed: the warning
            # itself went by thousands of lines ago, and the image looks
            # perfectly healthy afterwards: it is only slower at the work the
            # snippet was there to bake, which is usually the first plot.
            print(
                f"  warning: {len(result.warmup_failures)} warm-up snippet(s) failed, so "
                "what they cover is not baked in and a session compiles it at first use:",
                file=sys.stderr,
            )
            for failure in result.warmup_failures:
                print(f"    {failure}", file=sys.stderr)
    if config.sysimage is not True:
        write_workspace_config(dc_replace(config, sysimage=True), workspace=ws)
        print("  this folder now starts from it; `--no-sysimage` skips it for one run.")
    return True


def prepare_environment(adapter, *, workspace: Path, julia_project: Path, extensions=None) -> None:
    """Bring the environment to exactly what a session would run against.

    Capabilities are composed here for the same reason a session composes them
    before preparing its environment: their Julia dependencies belong to the
    project, and an image built before they are installed is missing precisely
    the packages an extended workspace exists to use. ``extensions`` lets a
    caller that has already discovered them (to read their warm-up code, say)
    pass the same list rather than discover twice.
    """

    from jutul_agent.agent.capabilities import collect_dependency_paths, discover_extensions
    from jutul_agent.simulators.env_setup import prepare_workspace_env
    from jutul_agent.sysimage_build import apply_windows_preferences

    if extensions is None:
        extensions = discover_extensions()
    # Before the environment precompiles, so a workload turned off here is one
    # nothing compiles twice. The build sets them again anyway.
    apply_windows_preferences(julia_project)
    prepare_workspace_env(
        adapter,
        workspace=workspace,
        julia_project=julia_project,
        sim_name=adapter.name,
        dependencies=collect_dependency_paths(extensions),
    )
