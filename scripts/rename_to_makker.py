"""Rename jutul-agent to makker across the repository.

A one-off, mechanical rewrite. Run it from the repository root on a clean
working tree::

    python scripts/rename_to_makker.py

It moves ``src/jutul_agent`` to ``src/makker`` and rewrites the three spellings
of the old name in every tracked text file. Running it twice changes nothing
the second time, so a branch that predates the rename can be brought across by
running it there and then merging ``main``.

Left alone on purpose:

- the Julia packages (``JutulAgent``, ``JutulAgent<Sim>``), which are renamed
  separately;
- the built web UI under ``web_dist``, which is rebuilt from source;
- this file.

Afterwards, refresh what is generated: ``uv sync``, ``uv run ruff format .``,
``uv run ruff check --fix .`` (the new name sorts differently in imports), and
``npm ci && npm run build`` in the webapp folder.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

OLD_PACKAGE_DIR = "src/jutul_agent"
NEW_PACKAGE_DIR = "src/makker"

# Longest-first is not needed: the three spellings never overlap.
REPLACEMENTS = (
    ("JUTUL_AGENT", "MAKKER"),
    ("jutul_agent", "makker"),
    ("jutul-agent", "makker"),
)

SKIPPED_DIRS = (f"{NEW_PACKAGE_DIR}/interfaces/server/web_dist/",)
SKIPPED_FILES = ("scripts/rename_to_makker.py",)

# Workspaces set up before the rename keep their folder until they are next
# opened, so a checkout that has one must go on ignoring it.
GITIGNORE_NEW_LINE = ".makker/"
GITIGNORE_OLD_LINE = "." + "jutul" + "-agent/"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout


def tracked_files() -> list[str]:
    return [name for name in git("ls-files", "-z").split("\0") if name]


def move_package() -> bool:
    if not any(name.startswith(f"{OLD_PACKAGE_DIR}/") for name in tracked_files()):
        return False
    if Path(NEW_PACKAGE_DIR).exists():
        sys.exit(f"{NEW_PACKAGE_DIR} already exists next to {OLD_PACKAGE_DIR}; resolve that first.")
    git("mv", OLD_PACKAGE_DIR, NEW_PACKAGE_DIR)
    return True


def rewrite(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def rewrite_gitignore(text: str) -> str:
    lines = text.split("\n")
    if GITIGNORE_NEW_LINE in lines:
        # Already converted: the old folder's line is there to stay as it is.
        return "\n".join(line if line == GITIGNORE_OLD_LINE else rewrite(line) for line in lines)
    lines = rewrite(text).split("\n")
    if GITIGNORE_NEW_LINE in lines:
        lines.insert(lines.index(GITIGNORE_NEW_LINE) + 1, GITIGNORE_OLD_LINE)
    return "\n".join(lines)


def rewrite_files() -> list[str]:
    changed = []
    for name in tracked_files():
        if name in SKIPPED_FILES or name.startswith(SKIPPED_DIRS):
            continue
        path = Path(name)
        if path.is_symlink() or not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new = rewrite_gitignore(text) if name == ".gitignore" else rewrite(text)
        if new != text:
            path.write_bytes(new.encode("utf-8"))
            changed.append(name)
    return changed


def main() -> int:
    if not Path(".git").exists():
        sys.exit("Run this from the repository root.")
    moved = move_package()
    changed = rewrite_files()
    if moved:
        print(f"Moved {OLD_PACKAGE_DIR} to {NEW_PACKAGE_DIR}.")
    print(f"Rewrote {len(changed)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
