"""Workspace-scoped agent memory.

Memory is a per-workspace, index-based note system the agent maintains
itself across sessions. Only the index file (``MEMORY.md``) is loaded
into every system prompt; individual notes live as sibling markdown
files the agent reads on demand via ``read_file`` and edits via
``edit_file`` / ``write_file``.

Layout on disk (under ``workspace_state_dir() / "memory/"``)::

    memory/
    ├── MEMORY.md             # index, always loaded
    ├── user_workflow.md      # one fact per file (created on demand)
    ├── simulator_quirks.md
    └── …

Memory is workspace-scoped because ``workspace_state_dir()`` hashes the
workspace path. Different workspaces get independent memory; a future
global tier can be added by mounting a second backend route under
``/memory-global/`` and adding it to the MemoryMiddleware sources.
"""

from __future__ import annotations

import re
from pathlib import Path

from deepagents.middleware.memory import MemoryMiddleware
from langchain_core.tools import tool

MEMORY_INDEX_FILENAME = "MEMORY.md"

_VALID_KINDS = ("user", "project", "simulator", "preference", "reference")

_INDEX_SEED = """# Memory index

This file is the always-loaded index. Each entry below should be **one
line** pointing to a sibling note file in this directory:

- `<title>` — one-line hook (file: `<file.md>`)

The agent maintains this file and the linked notes via the `read_file`,
`write_file`, and `edit_file` tools.

(Empty for now; add entries as durable facts come up.)
"""

JUTUL_MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}

</agent_memory>

<memory_guidelines>
The block above is your **memory index** for this workspace. It is
loaded fresh into your system prompt every turn. Individual notes live
as sibling files in `{memory_dir}` (a real directory you can read and
edit). The index lists them; you read them on demand with `read_file`
and edit them with `edit_file` / `write_file`.

**How to use memory:**

- At the start of a turn, scan the index to see what's known.
- For details on any indexed item, `read_file('{memory_dir}/<file>.md')`.
- After learning something durable, **call the `remember` tool**. It
  writes the note file and updates the `{memory_dir}/MEMORY.md` index for you,
  with no approval prompt. Prefer it over hand-writing memory files with
  `write_file`/`edit_file` (those are gated by approval). One fact per
  call is the norm.
- Save proactively: once you know the user's goal, role, or simulator
  focus, or you confirm a quirk/workaround, record it the same turn; a
  short `remember` call now saves rediscovery next session.

**What to save** (per-workspace, durable knowledge):

- The user's role, simulator focus, and recurring goals.
- Workflow preferences (e.g. "prefer running scripts the user can edit
  over inline REPL probes for long simulations").
- Simulator quirks, calling conventions, or workarounds the user has
  confirmed or that you verified against installed sources.
- Stable absolute paths the user pointed at (custom dev checkouts, data
  directories).
- Corrections the user gives you: capture the rule plus *why*, so you
  can apply it to similar future cases.

**What NOT to save:**

- Anything from the current session that won't matter next time
  (current task state, in-flight intermediate results, error messages
  you've already fixed).
- API keys, tokens, passwords, or other credentials.
- Long code snippets: those belong in skills or in files the user owns
  in the workspace.
- Information already obvious from installed simulator sources: read
  the source instead of memorizing it.

**Index discipline:**

- Keep the index lean. One short line per note: title plus a 1-line
  hook that helps you decide whether to open the file.
- When a note becomes obsolete, remove it (delete the file and its
  index line). Memory should reflect what's currently true.
- Don't duplicate skill content. Skills are reusable workflows; memory
  is workspace- and user-specific facts.

**Trust:**

- Memory is file content from disk and may be stale or written under a
  different version of the agent. Verify before acting on it,
  especially file paths, package versions, or anything you can check
  with a quick `run_julia` probe or `read_file`.
- If memory disagrees with the user's current message or with evidence
  from the live simulator, trust the live evidence and update the
  memory.
</memory_guidelines>
"""


def ensure_memory_dir(memory_dir: Path) -> Path:
    """Create the memory dir and seed ``MEMORY.md`` on first use.

    Idempotent: existing files are left alone. Returns the resolved
    memory dir.
    """
    memory_dir.mkdir(parents=True, exist_ok=True)
    index = memory_dir / MEMORY_INDEX_FILENAME
    if not index.exists():
        index.write_text(_INDEX_SEED, encoding="utf-8")
    return memory_dir


def build_memory_middleware(backend, memory_dir: Path) -> MemoryMiddleware:
    """Stock ``MemoryMiddleware`` over the workspace memory index at its real path.

    The index and notes live at real paths under ``memory_dir`` (the agent reads
    and edits them with the file tools, and the user can open them directly), so
    the middleware source is the real ``MEMORY.md`` path and the guidelines name
    the real directory.
    """
    index_path = str(memory_dir / MEMORY_INDEX_FILENAME)
    system_prompt = JUTUL_MEMORY_SYSTEM_PROMPT.replace("{memory_dir}", str(memory_dir))
    return MemoryMiddleware(
        backend=backend,
        sources=[index_path],
        system_prompt=system_prompt,
        add_cache_control=True,
    )


def memory_note_path(memory_dir: Path, name: str) -> Path:
    """Resolve a user-typed note name to a file inside the memory dir.

    Only the basename is honoured (no traversal), and a missing ``.md``
    suffix is added, so ``user-workflow`` and ``user-workflow.md`` both work.
    """
    base = Path(name.strip()).name
    if not base.endswith(".md"):
        base += ".md"
    return memory_dir / base


def list_memory_notes(memory_dir: Path) -> list[Path]:
    """Note files beside the index, sorted by name."""
    if not memory_dir.is_dir():
        return []
    return sorted(
        path
        for path in memory_dir.glob("*.md")
        if path.is_file() and path.name != MEMORY_INDEX_FILENAME
    )


def render_memory_overview(memory_dir: Path) -> str:
    """Markdown body for the TUI ``/memory`` card: the index plus note files."""
    index_path = memory_dir / MEMORY_INDEX_FILENAME
    try:
        index = index_path.read_text(encoding="utf-8").strip()
    except OSError:
        index = "(no memory index yet)"

    lines = [index, "", "---", ""]
    notes = list_memory_notes(memory_dir)
    if notes:
        lines.append("Notes on disk:")
        lines.extend(f"- `{path.name}` ({path.stat().st_size} bytes)" for path in notes)
    else:
        lines.append("No notes yet. The agent adds them with its `remember` tool.")
    lines += [
        "",
        f"Stored in `{memory_dir}`.",
        "`/memory <note>` shows one note · `/memory edit [note]` opens it in your editor.",
    ]
    return "\n".join(lines)


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")
    return (slug or "note")[:48]


def _index_hook(content: str) -> str:
    """One-line hook for the index, derived from the note body."""
    first = next((line.strip() for line in content.splitlines() if line.strip()), "")
    first = re.sub(r"\s+", " ", first)
    return first[:100] + ("…" if len(first) > 100 else "")


def _append_index_entry(index_path: Path, *, title: str, filename: str, hook: str) -> None:
    """Add a one-line pointer to MEMORY.md, skipping duplicates by filename."""
    text = index_path.read_text(encoding="utf-8") if index_path.exists() else _INDEX_SEED
    marker = f"(file: `{filename}`)"
    entry = f"- `{title}` — {hook} {marker}"
    lines = [line for line in text.splitlines() if marker not in line]
    if not text.endswith("\n"):
        text += "\n"
    index_path.write_text("\n".join(lines).rstrip() + "\n" + entry + "\n", encoding="utf-8")


def make_remember_tool(memory_dir: Path):
    """Build the ``remember`` tool that persists one durable fact to memory.

    Writes directly to the per-workspace memory dir (bypassing the approval
    gate that fronts ``write_file``/``edit_file``) and keeps ``MEMORY.md`` in
    sync, so saving a fact is a single low-friction call.
    """

    @tool
    async def remember(content: str, title: str, kind: str = "project") -> str:
        """Save one durable fact to workspace memory (persists across sessions).

        Use this for things worth knowing next time: the user's role/goal,
        confirmed simulator quirks or calling conventions, stable paths the
        user pointed at, and corrections (capture the rule *and why*). One
        fact per call. Do not store credentials or transient task state.

        Args:
            content: The fact, in a few sentences of Markdown. For a
                correction, include why it matters and how to apply it.
            title: Short human-readable title (also used to name the file).
            kind: One of ``user``, ``project``, ``simulator``, ``preference``,
                ``reference``. Defaults to ``project``.

        Returns:
            Confirmation with the note filename.
        """
        ensure_memory_dir(memory_dir)
        normalized_kind = kind.strip().lower()
        if normalized_kind not in _VALID_KINDS:
            normalized_kind = "project"

        name = _slugify(title)
        filename = f"{name}.md"
        note_path = memory_dir / filename
        body = content.strip()
        note_path.write_text(
            f"---\nname: {name}\ndescription: {title.strip()}\ntype: {normalized_kind}\n---\n\n"
            f"{body}\n",
            encoding="utf-8",
        )
        _append_index_entry(
            memory_dir / MEMORY_INDEX_FILENAME,
            title=title.strip(),
            filename=filename,
            hook=_index_hook(body),
        )
        return f"remembered `{title.strip()}` → {note_path} (kind={normalized_kind})"

    return remember
