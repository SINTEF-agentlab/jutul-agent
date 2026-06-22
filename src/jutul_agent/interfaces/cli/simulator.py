"""``jutul-agent simulator`` subcommand: add, list, and remove simulators.

Usage
-----
  jutul-agent simulator add <name>         [--display-name X] [--packages Pkg1,Pkg2]
  jutul-agent simulator add <path-to-dir>
  jutul-agent simulator list
  jutul-agent simulator remove <name>

``add`` with a bare name creates a scaffold simulator folder (``adapter.py`` +
``skills/``) in the user simulators directory.  ``add`` with a path to an
existing directory that contains an ``adapter.py`` registers it by creating a
symlink at the conventional location (falls back to copying on platforms where
symlinks are unavailable).

``list`` shows both built-in and user-defined simulators.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from jutul_agent.paths import user_simulators_dir
from jutul_agent.simulators import registry

_ADAPTER_TEMPLATE = """\
from pathlib import Path

from jutul_agent.simulators.base import SimulatorAdapter

{name_upper} = SimulatorAdapter(
    name="{name}",
    display_name="{display_name}",
    module_dir=Path(__file__).resolve().parent,
    package_imports=({packages}),
    primary_package="{primary_package}",
    domain_hints=(
        "Describe {display_name}'s core concepts and typical workflow here."
    ),
)
"""

_SKILL_MD_TEMPLATE = """\
---
name: {name}-overview
description: Overview of {display_name} concepts and typical workflow
---

# {display_name} overview

## When to use

Use this skill for any task involving {display_name}.

## Details

Add domain knowledge about {display_name} here.
"""


def build_parser(prog: str = "jutul-agent simulator") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Manage simulators.")
    sub = parser.add_subparsers(dest="action", required=True)

    add_p = sub.add_parser("add", help="Add a new simulator or register an existing one.")
    add_p.add_argument(
        "name_or_path",
        metavar="name-or-path",
        help=(
            "Bare simulator name (creates scaffold) or path to an existing "
            "simulator directory containing an adapter.py (registers by symlink/copy)."
        ),
    )
    add_p.add_argument(
        "--display-name",
        metavar="NAME",
        default=None,
        help="Human-readable name for the simulator (default: title-cased name).",
    )
    add_p.add_argument(
        "--packages",
        metavar="PKG1,PKG2",
        default=None,
        help="Comma-separated Julia package names the agent should know about.",
    )

    sub.add_parser("list", help="List all simulators (built-in and user-defined).")

    remove_p = sub.add_parser("remove", help="Remove a user-defined simulator.")
    remove_p.add_argument("name", help="Name of the simulator to remove.")

    return parser


def _register_path(src: Path, dest: Path) -> None:
    """Point ``dest`` at ``src`` via symlink, falling back to a directory copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        print(f"Already registered: {dest}", file=sys.stderr)
        sys.exit(1)
    try:
        dest.symlink_to(src.resolve())
        print(f"Registered (symlink): {dest} → {src.resolve()}")
    except (OSError, NotImplementedError):
        shutil.copytree(src, dest)
        print(f"Registered (copy): {dest}")
        print(
            "Note: symlinks unavailable on this platform — edits to the original "
            "directory will not be reflected automatically.",
            file=sys.stderr,
        )


def _cmd_add(args: argparse.Namespace) -> int:
    candidate = Path(args.name_or_path)

    if candidate.is_dir():
        if not (candidate / "adapter.py").exists():
            print(
                f"{candidate} does not contain an adapter.py — not a valid simulator directory.",
                file=sys.stderr,
            )
            return 1
        dest = user_simulators_dir() / candidate.name
        _register_path(candidate, dest)
        return 0

    name = args.name_or_path
    if "/" in name or "\\" in name:
        print(
            f"{name!r} looks like a path but does not exist as a directory.",
            file=sys.stderr,
        )
        return 1

    display_name = args.display_name or name.title()
    raw_packages = args.packages or name
    packages_list = [p.strip() for p in raw_packages.split(",") if p.strip()]
    primary_package = packages_list[0] if packages_list else name
    packages_repr = ", ".join(f'"{p}"' for p in packages_list)

    sim_dir = user_simulators_dir() / name
    if sim_dir.exists():
        print(f"Simulator directory already exists: {sim_dir}", file=sys.stderr)
        return 1
    sim_dir.mkdir(parents=True, exist_ok=True)
    skills_dir = sim_dir / "skills" / f"{name}-overview"
    skills_dir.mkdir(parents=True, exist_ok=True)

    adapter_text = _ADAPTER_TEMPLATE.format(
        name=name,
        name_upper=name.upper().replace("-", "_"),
        display_name=display_name,
        packages=packages_repr,
        primary_package=primary_package,
    )
    (sim_dir / "adapter.py").write_text(adapter_text, encoding="utf-8")
    (skills_dir / "SKILL.md").write_text(
        _SKILL_MD_TEMPLATE.format(name=name, display_name=display_name),
        encoding="utf-8",
    )

    print(f"Created simulator scaffold: {sim_dir}")
    print(f"  adapter.py  — fill in domain_hints and Julia package details")
    print(f"  skills/{name}-overview/SKILL.md  — add domain knowledge")
    return 0


def _cmd_list() -> int:

    built_in = registry.names()
    sims_dir = user_simulators_dir()
    user_sims = []
    if sims_dir.is_dir():
        for d in sorted(sims_dir.iterdir()):
            if d.is_dir() and (d / "adapter.py").exists():
                user_sims.append(d)

    print("Built-in simulators:")
    for name in built_in:
        if name not in [d.name for d in user_sims]:
            adapter = registry.get(name)
            print(f"  {name}  ({adapter.display_name})")

    if user_sims:
        print("User simulators:")
        for d in user_sims:
            print(f"  {d.name}  ({d})")
    else:
        print("User simulators: (none)")
        print(f"Add one with: jutul-agent simulator add <name>")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    name = args.name
    if name in registry.names():
        builtin_dir = user_simulators_dir() / name
        if not builtin_dir.exists() and not builtin_dir.is_symlink():
            print(
                f"{name!r} is a built-in simulator and cannot be removed.",
                file=sys.stderr,
            )
            return 1
    target = user_simulators_dir() / name
    if not target.exists() and not target.is_symlink():
        print(f"User simulator {name!r} not found.", file=sys.stderr)
        return 1
    if target.is_symlink():
        target.unlink()
    else:
        shutil.rmtree(target)
    print(f"Removed: {target}")
    return 0


def run(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "add":
        return _cmd_add(args)
    if args.action == "list":
        return _cmd_list()
    if args.action == "remove":
        return _cmd_remove(args)
    parser.print_help()
    return 1
