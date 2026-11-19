"""Rename jutul-agent to makker across the repository.

A one-off, mechanical rewrite. Run it from the repository root on a clean
working tree::

    python scripts/rename_to_makker.py

It moves ``src/jutul_agent`` to ``src/makker``, renames the Julia packages
(``JutulAgent`` to ``Makker``, with new UUIDs), and rewrites the old names in
every tracked text file. Running it twice changes nothing the second time, so a
branch that predates the rename can be brought across by running it there and
then merging ``main``.

Left alone on purpose:

- the built web UI under ``web_dist``, which is rebuilt from source;
- ``bridge/``, the last jutul-agent release, which is meant to carry the old name;
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

# Applied in order. No entry's output contains a later entry's input.
REPLACEMENTS = (
    # The three spellings of the product name.
    ("JUTUL_AGENT", "MAKKER"),
    ("jutul_agent", "makker"),
    ("jutul-agent", "makker"),
    # The Julia packages: a new name is a new package, so each gets a new UUID.
    ("79b12aa6-73c9-4fa8-9c54-e97745f42687", "0d1d1ec9-c0b9-4de0-a260-d13dbbd400b2"),  # shared
    ("f9f14f6a-5be2-4b55-b75d-847ac964f0ff", "93c9a128-c61c-47ba-a3ed-ee8197bf8a0b"),  # BattMo
    ("46917057-ee61-4ce0-8807-c8cab0e0431f", "6ae4a4f4-c201-4215-9553-998d03f1d36a"),  # Fimbul
    ("69df87d8-8b4b-4157-81d2-8b93ff139141", "4b91d94b-bab6-4ec5-a649-083be1735a47"),  # JutulDarcy
    ("118bae70-9d54-4e00-8665-eefd5efc28fa", "2c594c4c-ba15-48a6-abd4-ceee9b009a5c"),  # Mocca
    ("JutulAgent", "Makker"),
    ("jutulagent", "makker"),
    # Internal names that carried the old prefix. Names that belong to the Jutul
    # framework itself (``Jutul``, ``JutulDarcy``, ``jutul_output``) are not here.
    ("__JUTUL_", "__MAKKER_"),
    ("JUTUL_MAKIE_MATCH", "MAKKER_MAKIE_MATCH"),
    ("JUTUL_LAB_UPDATE", "MAKKER_LAB_UPDATE"),
    ("JUTUL_MEMORY_SYSTEM_PROMPT", "MAKKER_MEMORY_SYSTEM_PROMPT"),
    ("JutulDebug", "MakkerDebug"),
    ("jutulDebug", "makkerDebug"),
    ("onJutulUi", "onMakkerUi"),
    ("_run_jutul_session", "_run_makker_session"),
    ("_jutul_tolerant_b64", "_makker_tolerant_b64"),
    ("jutul-base-path", "makker-base-path"),
    ("jutul-review", "makker-review"),
    ("jutul-bench", "makker-bench"),
    ("jutul-state", "makker-state"),
    ("jutul-live-", "makker-live-"),
    ("jutul-profile-", "makker-profile-"),
    ("jutul-profturn-", "makker-profturn-"),
    ("jutul-lab-", "makker-lab-"),
)

# Tracked paths are renamed by the same rule as the Julia package names.
OLD_JULIA_NAME = "JutulAgent"
NEW_JULIA_NAME = "Makker"

SKIPPED_DIRS = (f"{NEW_PACKAGE_DIR}/interfaces/server/web_dist/", "bridge/")
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


def move_julia_packages() -> int:
    moved = 0
    for name in tracked_files():
        if OLD_JULIA_NAME not in name or name.startswith(SKIPPED_DIRS):
            continue
        target = Path(name.replace(OLD_JULIA_NAME, NEW_JULIA_NAME))
        target.parent.mkdir(parents=True, exist_ok=True)
        git("mv", name, str(target))
        moved += 1
        _remove_if_empty(Path(name).parent)
    return moved


def _remove_if_empty(folder: Path) -> None:
    """Drop the folders a moved package leaves behind, as far up as they are empty."""
    while OLD_JULIA_NAME in folder.as_posix():
        try:
            folder.rmdir()
        except OSError:
            return
        folder = folder.parent


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
    julia_files = move_julia_packages()
    changed = rewrite_files()
    if moved:
        print(f"Moved {OLD_PACKAGE_DIR} to {NEW_PACKAGE_DIR}.")
    if julia_files:
        print(f"Moved {julia_files} Julia package files.")
    print(f"Rewrote {len(changed)} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
