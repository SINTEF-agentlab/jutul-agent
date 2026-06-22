"""``jutul-agent tool`` subcommand: add and list tools.

Usage
-----
  jutul-agent tool add <name>         [--sim <simulator>]
  jutul-agent tool add <path-to-file> [--sim <simulator>]
  jutul-agent tool list
  jutul-agent tool remove <name>      [--sim <simulator>]

``add`` with a bare name creates a scaffold tool file in the user tools
directory.  ``add`` with a path to an existing tool file registers it by
creating a symlink at the conventional location (falls back to copying on
platforms where symlinks are unavailable).

``list`` shows user-defined tools.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

from jutul_agent.paths import user_simulators_dir, user_tools_dir
from jutul_agent.simulators import registry

_TOOL_TEMPLATE = '''\
from langchain_core.tools import tool

from jutul_agent.session import Session

def make_{tool_name}_tool(session: Session):
    @tool
    async def {tool_name}({tool_args}) -> str:
        """{tool_description}
        """
        {tool_body}

    return {tool_name}
'''


def build_parser(prog: str = "jutul-agent tool") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Manage tools.")
    sub = parser.add_subparsers(dest="action", required=True)

    add_p = sub.add_parser("add", help="Add a new tool.")
    add_p.add_argument(
        "name_or_path",
        metavar="name-or-path",
        help=("Bare tool name (creates scaffold) or path to existing tool file."),
    )
    add_p.add_argument(
        "--sim",
        metavar="SIMULATOR",
        default=None,
        help=(
            "Scope the tool to a specific simulator.  Without this flag the "
            "tool is global (loaded for every simulator)."
        ),
    )

    sub.add_parser("list", help="List all user-defined tools.")

    remove_p = sub.add_parser("remove", help="Remove a user-defined tool.")
    remove_p.add_argument("name", help="Name of the tool to remove.")
    remove_p.add_argument(
        "--sim",
        metavar="SIMULATOR",
        default=None,
        help="Simulator the tool is scoped to (omit for global tools).",
    )

    return parser


def _target_file(name: str, sim: str | None) -> Path:
    if not name.endswith(".py"):
        name += ".py"
    if sim:
        return user_simulators_dir() / sim / "tools" / name
    return user_tools_dir() / name


def _defines_factory(path: Path, factory_name: str) -> bool:
    """Whether ``path`` defines a top-level ``factory_name`` function.

    Uses ``ast`` rather than importing the module so a bad registration
    candidate can't run arbitrary code as a side effect of the check.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == factory_name
        for node in tree.body
    )


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
        shutil.copy2(src, dest)
        print(f"Registered (copy): {dest}")
        print(
            "Note: symlinks unavailable on this platform — edits to the original "
            "file will not be reflected automatically.",
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

    if not candidate.exists():
        tool_name = candidate.name
        display_name = tool_name.replace("_", " ").title()
        tool_code = _TOOL_TEMPLATE.format(
            tool_name=tool_name,
            tool_args="",
            tool_description=f"Description of {display_name}.",
            tool_body="pass  # TODO: implement the tool's functionality here.",
        )
        dest = _target_file(tool_name, sim)
        if dest.exists():
            print(f"Tool already exists: {dest}", file=sys.stderr)
            return 1
        if not dest.parent.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(tool_code)
        print(f"Created new tool scaffold at: {dest}")
        print("Edit the file to implement the tool's functionality.")
    else:
        if not candidate.is_file():
            print(f"Not a file: {candidate}", file=sys.stderr)
            return 1
        dest = _target_file(candidate.name, sim)
        factory_name = f"make_{dest.stem}_tool"
        if not _defines_factory(candidate, factory_name):
            print(
                f"{candidate} does not define `{factory_name}()` — jutul-agent "
                "would not be able to load it as a tool.",
                file=sys.stderr,
            )
            return 1
        _register_path(candidate, dest)
    return 0


def _cmd_list() -> int:
    found_any = False

    global_dir = user_tools_dir()
    if global_dir.is_dir():
        for tool_file in sorted(global_dir.glob("*.py")):
            print(f"User tool: {tool_file.name}  ({tool_file})")
            found_any = True

    sims_dir = user_simulators_dir()
    if sims_dir.is_dir():
        for sim_dir in sorted(sims_dir.iterdir()):
            tools_subdir = sim_dir / "tools"
            if tools_subdir.is_dir():
                for tool_file in sorted(tools_subdir.glob("*.py")):
                    print(f"User tool for {sim_dir.name}: {tool_file.name}  ({tool_file})")
                    found_any = True

    if not found_any:
        print("No user-defined tools found.")
        print(
            "Use 'jutul-agent tool add <name>' to create a new tool scaffold, "
            "or 'jutul-agent tool add <path-to-file>' to register an existing tool file."
        )
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    sim: str | None = args.sim
    if sim and sim not in registry.names():
        print(
            f"Unknown simulator {sim!r}. Known: {', '.join(registry.names())}",
            file=sys.stderr,
        )
        return 1
    target = _target_file(args.name, sim)
    if not target.exists() and not target.is_symlink():
        scope = f" (simulator: {sim})" if sim else " (global)"
        print(f"Tool {args.name!r}{scope} not found.", file=sys.stderr)
        return 1
    target.unlink()
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
