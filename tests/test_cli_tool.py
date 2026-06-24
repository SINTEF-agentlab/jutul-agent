"""Tests for the ``jutul-agent tool`` subcommand."""

from __future__ import annotations

from pathlib import Path

import pytest

from jutul_agent.interfaces.cli import main
from jutul_agent.paths import user_simulators_dir, user_tools_dir
from jutul_agent.simulators import registry

# A built-in simulator guaranteed present regardless of any user-tools/state-home
# overrides, since the registry discovers it from the shipped package modules.
_BUILTIN_SIM = "jutuldarcy"
assert _BUILTIN_SIM in registry.names()


def _write_tool_file(path: Path, tool_name: str, *, body: str = 'return "ok"') -> None:
    path.write_text(
        f'''\
from langchain_core.tools import tool

from jutul_agent.session import Session

def make_{tool_name}_tool(session: Session):
    @tool
    async def {tool_name}() -> str:
        """A test tool."""
        {body}

    return {tool_name}
''',
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# tool add (scaffold)
# ---------------------------------------------------------------------------


def test_tool_add_creates_global_scaffold(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["tool", "add", "mynewtool"])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    dest = user_tools_dir() / "mynewtool.py"
    assert dest.exists()
    assert "def make_mynewtool_tool(session: Session):" in dest.read_text(encoding="utf-8")
    assert "Created new tool scaffold" in captured.out


def test_tool_add_creates_scoped_scaffold_for_known_simulator(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["tool", "add", "simtool", "--sim", _BUILTIN_SIM])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    dest = user_simulators_dir() / _BUILTIN_SIM / "tools" / "simtool.py"
    assert dest.exists()
    # Scoped tool must not also land in the global directory.
    assert not (user_tools_dir() / "simtool.py").exists()


def test_tool_add_rejects_unknown_simulator(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["tool", "add", "foo", "--sim", "not-a-real-simulator"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Unknown simulator" in captured.err
    assert not (user_tools_dir() / "foo.py").exists()


def test_tool_add_errors_when_already_exists(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tool", "add", "duptool"]) == 0
    capsys.readouterr()

    code = main(["tool", "add", "duptool"])
    captured = capsys.readouterr()

    assert code == 1
    assert "already exists" in captured.err


# ---------------------------------------------------------------------------
# tool add (register existing file)
# ---------------------------------------------------------------------------


def test_tool_add_registers_existing_file_via_symlink(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "external_tool.py"
    _write_tool_file(src, "external_tool")

    code = main(["tool", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    dest = user_tools_dir() / "external_tool.py"
    assert dest.is_symlink()
    assert dest.resolve() == src.resolve()
    assert "Registered (symlink)" in captured.out


def test_tool_add_registers_existing_file_scoped_to_simulator(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "external_sim_tool.py"
    _write_tool_file(src, "external_sim_tool")

    code = main(["tool", "add", str(src), "--sim", _BUILTIN_SIM])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    dest = user_simulators_dir() / _BUILTIN_SIM / "tools" / "external_sim_tool.py"
    assert dest.is_symlink()
    assert not (user_tools_dir() / "external_sim_tool.py").exists()


def test_tool_add_falls_back_to_copy_when_symlinks_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "copied_tool.py"
    _write_tool_file(src, "copied_tool")

    monkeypatch.setattr(
        Path, "symlink_to", lambda self, *a, **k: (_ for _ in ()).throw(OSError("no symlinks"))
    )

    code = main(["tool", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    dest = user_tools_dir() / "copied_tool.py"
    assert dest.is_file() and not dest.is_symlink()
    assert dest.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")
    assert "Registered (copy)" in captured.out
    assert "symlinks unavailable" in captured.err


def test_tool_add_errors_when_already_registered(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "again_tool.py"
    _write_tool_file(src, "again_tool")
    assert main(["tool", "add", str(src)]) == 0
    capsys.readouterr()

    # _register_path exits the process directly on a duplicate registration.
    with pytest.raises(SystemExit) as exc_info:
        main(["tool", "add", str(src)])
    captured = capsys.readouterr()

    assert exc_info.value.code == 1
    assert "Already registered" in captured.err


def test_tool_add_rejects_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "a-directory"
    src.mkdir()

    code = main(["tool", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 1
    assert "Not a file" in captured.err


def test_tool_add_wraps_bare_decorated_function_in_factory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "bare_tool.py"
    src.write_text(
        '''\
from langchain_core.tools import tool

@tool
async def bare_tool() -> str:
    """A bare tool, not yet wrapped in a factory."""
    return "ok"
''',
        encoding="utf-8",
    )

    code = main(["tool", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    dest = user_tools_dir() / "bare_tool.py"
    assert dest.is_file() and not dest.is_symlink()
    text = dest.read_text(encoding="utf-8")
    assert "def make_bare_tool_tool(session: Session):" in text
    assert text.count("@tool") == 1  # not duplicated alongside the original decorator
    compile(text, str(dest), "exec")
    assert "Wrapped" in captured.out


def test_tool_add_wraps_bare_undecorated_function_in_factory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "undecorated_tool.py"
    src.write_text("def undecorated_tool() -> str:\n    return 'ok'\n", encoding="utf-8")

    code = main(["tool", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    dest = user_tools_dir() / "undecorated_tool.py"
    text = dest.read_text(encoding="utf-8")
    assert "@tool" in text
    assert "def make_undecorated_tool_tool(session: Session):" in text
    compile(text, str(dest), "exec")


def test_tool_add_wrapping_preserves_helper_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "helper_tool.py"
    src.write_text(
        """\
import math
from langchain_core.tools import tool

def helper(x):
    return x * 2

@tool(name="helper_tool")
def helper_tool() -> str:
    return str(helper(math.pi))
""",
        encoding="utf-8",
    )

    code = main(["tool", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    dest = user_tools_dir() / "helper_tool.py"
    text = dest.read_text(encoding="utf-8")
    assert "def helper(x):" in text
    assert "import math" in text
    compile(text, str(dest), "exec")


def test_tool_add_wrapping_refuses_to_overwrite_existing_registration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "again_bare_tool.py"
    src.write_text("def again_bare_tool() -> str:\n    return 'ok'\n", encoding="utf-8")
    assert main(["tool", "add", str(src)]) == 0
    capsys.readouterr()

    code = main(["tool", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 1
    assert "Already registered" in captured.err


def test_tool_add_rejects_file_without_matching_factory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "broken_tool.py"
    src.write_text("def some_other_function():\n    return 'nope'\n", encoding="utf-8")

    code = main(["tool", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 1
    assert "does not define `make_broken_tool_tool()`" in captured.err
    assert not (user_tools_dir() / "broken_tool.py").exists()


# ---------------------------------------------------------------------------
# tool list
# ---------------------------------------------------------------------------


def test_tool_list_reports_when_empty(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["tool", "list"])
    captured = capsys.readouterr()

    assert code == 0
    assert "No user-defined tools found." in captured.out


def test_tool_list_shows_global_and_scoped_tools(capsys: pytest.CaptureFixture[str]) -> None:
    main(["tool", "add", "globaltool"])
    main(["tool", "add", "simtool2", "--sim", _BUILTIN_SIM])
    capsys.readouterr()

    code = main(["tool", "list"])
    captured = capsys.readouterr()

    assert code == 0
    assert "User tool: globaltool.py" in captured.out
    assert f"User tool for {_BUILTIN_SIM}: simtool2.py" in captured.out


# ---------------------------------------------------------------------------
# tool remove
# ---------------------------------------------------------------------------


def test_tool_remove_deletes_scaffold(capsys: pytest.CaptureFixture[str]) -> None:
    main(["tool", "add", "removable"])
    capsys.readouterr()

    code = main(["tool", "remove", "removable"])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    assert not (user_tools_dir() / "removable.py").exists()
    assert "Removed" in captured.out


def test_tool_remove_unlinks_symlink_without_deleting_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "shared_tool.py"
    _write_tool_file(src, "shared_tool")
    main(["tool", "add", str(src)])
    capsys.readouterr()

    code = main(["tool", "remove", "shared_tool"])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    assert not (user_tools_dir() / "shared_tool.py").exists()
    assert src.exists()  # original source untouched


def test_tool_remove_errors_when_missing(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["tool", "remove", "ghost"])
    captured = capsys.readouterr()

    assert code == 1
    assert "not found" in captured.err


def test_tool_remove_rejects_unknown_simulator(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["tool", "remove", "foo", "--sim", "not-a-real-simulator"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Unknown simulator" in captured.err


def test_tool_remove_respects_sim_scope(capsys: pytest.CaptureFixture[str]) -> None:
    main(["tool", "add", "scoped", "--sim", _BUILTIN_SIM])
    capsys.readouterr()

    # Removing without --sim looks in the global dir, where it doesn't exist.
    code = main(["tool", "remove", "scoped"])
    captured = capsys.readouterr()
    assert code == 1
    assert "not found" in captured.err

    code = main(["tool", "remove", "scoped", "--sim", _BUILTIN_SIM])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert not (user_simulators_dir() / _BUILTIN_SIM / "tools" / "scoped.py").exists()
