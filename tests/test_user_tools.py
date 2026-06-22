"""Tests for ``load_user_tools``: global and simulator-scoped user tool loading."""

from __future__ import annotations

from pathlib import Path

from fakes import FakeJulia, make_fake_adapter
from jutul_agent.agent.user_tools import load_user_tools
from jutul_agent.paths import user_simulators_dir, user_tools_dir
from jutul_agent.session import Session


def _write_tool_file(path: Path, tool_name: str, *, returns: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''\
from langchain_core.tools import tool

from jutul_agent.session import Session

def make_{tool_name}_tool(session: Session):
    @tool
    async def {tool_name}() -> str:
        """A test tool."""
        return {returns!r}

    return {tool_name}
''',
        encoding="utf-8",
    )


def _session(tmp_path: Path, *, sim_name: str = "fakesim", sid: str = "user-tools-test") -> Session:
    return Session.create(
        julia=FakeJulia(),
        state_root=tmp_path,
        simulator=make_fake_adapter(tmp_path, name=sim_name),
        session_id=sid,
    )


async def test_load_user_tools_returns_empty_when_no_dirs(tmp_path: Path) -> None:
    tools = load_user_tools(_session(tmp_path))
    assert tools == []


async def test_load_user_tools_loads_global_tool(tmp_path: Path) -> None:
    _write_tool_file(user_tools_dir() / "ping_pong.py", "ping_pong", returns="pong")

    tools = load_user_tools(_session(tmp_path))

    assert [t.name for t in tools] == ["ping_pong"]
    assert await tools[0].ainvoke({}) == "pong"


async def test_load_user_tools_loads_simulator_scoped_tool_for_matching_simulator(
    tmp_path: Path,
) -> None:
    _write_tool_file(
        user_simulators_dir() / "fakesim" / "tools" / "weather_check.py",
        "weather_check",
        returns="sunny",
    )

    tools = load_user_tools(_session(tmp_path, sim_name="fakesim"))

    assert [t.name for t in tools] == ["weather_check"]
    assert await tools[0].ainvoke({}) == "sunny"


async def test_load_user_tools_does_not_leak_across_simulators(tmp_path: Path) -> None:
    _write_tool_file(
        user_simulators_dir() / "fakesim" / "tools" / "weather_check.py",
        "weather_check",
        returns="sunny",
    )

    # A session for a different simulator must not see fakesim's scoped tool.
    tools = load_user_tools(_session(tmp_path, sim_name="othersim"))

    assert tools == []


async def test_load_user_tools_combines_global_and_scoped_tools(tmp_path: Path) -> None:
    _write_tool_file(user_tools_dir() / "ping_pong.py", "ping_pong", returns="pong")
    _write_tool_file(
        user_simulators_dir() / "fakesim" / "tools" / "weather_check.py",
        "weather_check",
        returns="sunny",
    )

    tools = load_user_tools(_session(tmp_path, sim_name="fakesim"))

    assert sorted(t.name for t in tools) == ["ping_pong", "weather_check"]


async def test_load_user_tools_skips_file_without_matching_factory(tmp_path: Path) -> None:
    bad = user_tools_dir() / "broken.py"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("def some_other_function():\n    return 'nope'\n", encoding="utf-8")

    tools = load_user_tools(_session(tmp_path))

    assert tools == []
