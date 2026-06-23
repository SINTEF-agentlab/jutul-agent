"""``jutul-agent skill`` subcommand: add, list, and remove user skills.

Usage
-----
  jutul-agent skill add <name>          [--sim <simulator>]
  jutul-agent skill add <path-to-dir>   [--sim <simulator>]
  jutul-agent skill list
  jutul-agent skill remove <name>       [--sim <simulator>]

``add`` with a bare name creates a scaffold skill folder (and ``SKILL.md``) in
the conventional user skills directory.  ``add`` with a path to an existing
directory that contains a ``SKILL.md`` registers it by creating a symlink at
the conventional location (falls back to copying when symlinks are unavailable,
e.g. on Windows without elevated rights).

Skills placed without ``--sim`` are global and loaded for every simulator.
Skills placed with ``--sim`` are loaded only when that simulator is active.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from jutul_agent.paths import user_simulators_dir, user_skills_dir
from jutul_agent.simulators import registry

_SKILL_MD_TEMPLATE = """\
---
name: {name}
description: One-line summary — used to decide relevance
---

# {name}

## When to use

Describe when the agent should consult this skill.

## Details

Add your domain knowledge here.
"""


def build_parser(prog: str = "jutul-agent skill") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Manage user-defined skills.")
    sub = parser.add_subparsers(dest="action", required=True)

    add_p = sub.add_parser("add", help="Add a new skill or register an existing one.")
    add_p.add_argument(
        "name_or_path",
        metavar="name-or-path",
        help=(
            "Bare skill name (creates scaffold) or path to an existing skill "
            "directory containing a SKILL.md (registers it by symlink/copy)."
        ),
    )
    add_p.add_argument(
        "--sim",
        metavar="SIMULATOR",
        default=None,
        help=(
            "Scope the skill to a specific simulator.  Without this flag the "
            "skill is global (loaded for every simulator)."
        ),
    )

    sub.add_parser("list", help="List all user-defined skills.")

    remove_p = sub.add_parser("remove", help="Remove a user-defined skill.")
    remove_p.add_argument("name", help="Name of the skill to remove.")
    remove_p.add_argument(
        "--sim",
        metavar="SIMULATOR",
        default=None,
        help="Simulator the skill is scoped to (omit for global skills).",
    )

    return parser


def _target_dir(name: str, sim: str | None) -> Path:
    if sim:
        return user_simulators_dir() / sim / "skills" / name
    return user_skills_dir() / name


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
    sim: str | None = args.sim

    if sim and sim not in registry.names():
        print(
            f"Unknown simulator {sim!r}. Known: {', '.join(registry.names())}",
            file=sys.stderr,
        )
        return 1

    if candidate.is_dir():
        if not (candidate / "SKILL.md").exists():
            print(
                f"{candidate} does not contain a SKILL.md — not a valid skill directory.",
                file=sys.stderr,
            )
            return 1
        dest = _target_dir(candidate.name, sim)
        _register_path(candidate, dest)
    else:
        name = args.name_or_path
        if "/" in name or "\\" in name:
            print(
                f"{name!r} looks like a path but does not exist as a directory.",
                file=sys.stderr,
            )
            return 1
        dest = _target_dir(name, sim)
        if dest.exists():
            print(f"Skill already exists: {dest}", file=sys.stderr)
            return 1
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "SKILL.md").write_text(_SKILL_MD_TEMPLATE.format(name=name), encoding="utf-8")
        print(f"Created skill scaffold: {dest / 'SKILL.md'}")
        print("Edit SKILL.md to add your domain knowledge.")
    return 0


def _cmd_list() -> int:
    found_any = False

    global_dir = user_skills_dir()
    if global_dir.is_dir():
        for skill_dir in sorted(global_dir.iterdir()):
            if (skill_dir / "SKILL.md").exists():
                print(f"  [global]  {skill_dir.name}  ({skill_dir})")
                found_any = True

    sims_dir = user_simulators_dir()
    if sims_dir.is_dir():
        for sim_dir in sorted(sims_dir.iterdir()):
            skills_subdir = sim_dir / "skills"
            if skills_subdir.is_dir():
                for skill_dir in sorted(skills_subdir.iterdir()):
                    if (skill_dir / "SKILL.md").exists():
                        print(f"  [{sim_dir.name}]  {skill_dir.name}  ({skill_dir})")
                        found_any = True

    if not found_any:
        print("No user skills found.")
        print("Add one with: jutul-agent skill add <name>")
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    sim: str | None = args.sim
    if sim and sim not in registry.names():
        print(
            f"Unknown simulator {sim!r}. Known: {', '.join(registry.names())}",
            file=sys.stderr,
        )
        return 1
    target = _target_dir(args.name, sim)
    if not target.exists() and not target.is_symlink():
        scope = f" (simulator: {sim})" if sim else " (global)"
        print(f"Skill {args.name!r}{scope} not found.", file=sys.stderr)
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
