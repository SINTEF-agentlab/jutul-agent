"""Tests for the capability-composition and extension model."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from jutul_agent.agent import builder
from jutul_agent.agent.capabilities import (
    Capability,
    HttpToolSpec,
    collect_dependency_paths,
    collect_prompt_fragments,
    collect_skill_dirs,
    collect_subagents,
    collect_tools,
    discover_extensions,
    http_tool_capability,
    select_for_surface,
)
from jutul_agent.agent.prompts import assemble_session_prompt
from jutul_agent.session import Session


@tool
def _demo_tool(x: str) -> str:
    """A demo tool."""
    return x


def _cap(**kwargs) -> Capability:
    return Capability(name="ext", **kwargs)


# --- surface selection and collection -------------------------------------


def test_select_for_surface() -> None:
    everywhere = _cap()
    web_only = Capability(name="web", surfaces=("web",))
    tui_only = Capability(name="tui", surfaces=("tui",))
    selected = select_for_surface([everywhere, web_only, tui_only], "web")
    assert selected == [everywhere, web_only]


def test_collect_helpers(session: Session) -> None:
    cap = _cap(
        tools=(lambda _s: _demo_tool,),
        skill_dirs=(("/skills/demo", "Demo"),),
        dependencies=(Path("/local/Foo/Project.toml"),),
        subagents=(lambda _s: {"name": "sub"},),
        prompt_fragment="  FRAGMENT  ",
    )
    assert [t.name for t in collect_tools([cap], session)] == ["_demo_tool"]
    assert collect_skill_dirs([cap]) == [("/skills/demo", "Demo")]
    assert collect_dependency_paths([cap]) == [Path("/local/Foo/Project.toml")]
    assert collect_subagents([cap], session) == [{"name": "sub"}]
    assert collect_prompt_fragments([cap]) == ["  FRAGMENT  "]
    assert collect_prompt_fragments([_cap(prompt_fragment="   ")]) == []


# --- entry-point discovery ------------------------------------------------


def test_discover_extensions_returns_list() -> None:
    assert isinstance(discover_extensions(), list)


def test_discover_extensions_loads_capability(monkeypatch) -> None:
    import importlib.metadata as importlib_metadata

    cap = Capability(name="discovered")

    class _EntryPoint:
        name = "ext"

        def load(self):
            return cap

    monkeypatch.setattr(importlib_metadata, "entry_points", lambda group: [_EntryPoint()])
    assert discover_extensions() == [cap]


def test_discover_extensions_skips_broken_but_says_why(monkeypatch, capsys) -> None:
    """A capability that fails to load is otherwise indistinguishable from one
    that is not installed, and the cause is usually a fixable setup step."""
    import importlib.metadata as importlib_metadata

    class _Broken:
        name = "brokenext"

        def load(self):
            raise RuntimeError("SEISMIC_DATA_DIR is not set")

    class _NotACapability:
        name = "wrongext"

        def load(self):
            return lambda: "not a capability"

    monkeypatch.setattr(
        importlib_metadata, "entry_points", lambda group: [_Broken(), _NotACapability()]
    )
    assert discover_extensions() == []
    warned = capsys.readouterr().err
    assert "brokenext" in warned and "SEISMIC_DATA_DIR is not set" in warned
    assert "wrongext" in warned and "not a Capability" in warned


# --- declarative HTTP tools -----------------------------------------------


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def post(self, url: str, json: dict) -> _FakeResponse:
        self.calls.append((url, json))
        return _FakeResponse("result-text")


def test_http_tool_capability_builds_and_calls() -> None:
    client = _FakeClient()
    cap = http_tool_capability(
        "host-app",
        [
            HttpToolSpec(
                name="run_sim",
                description="Run the host app simulation.",
                endpoint="http://host/run",
                parameters={"p": {"type": "number", "description": "a parameter"}},
            )
        ],
        client=client,
    )
    built = cap.tools[0](None)  # factory ignores the session
    assert built.name == "run_sim"
    assert "host app" in built.description

    result = asyncio.run(built.ainvoke({"p": 2}))
    assert result == "result-text"
    assert client.calls == [("http://host/run", {"p": 2.0})]


def _http_tool(client: Any) -> Any:
    cap = http_tool_capability(
        "host-app",
        [HttpToolSpec(name="run_sim", description="Run it.", endpoint="http://host/run")],
        client=client,
    )
    return cap.tools[0](None)


def test_http_tool_returns_a_status_error_as_text() -> None:
    # A failing host endpoint must come back as a tool-result string the model can
    # act on, not raise out and abort the whole turn.
    import httpx

    class _Resp:
        def raise_for_status(self) -> None:
            request = httpx.Request("POST", "http://host/run")
            response = httpx.Response(500, text="boom", request=request)
            raise httpx.HTTPStatusError("server error", request=request, response=response)

    class _Client:
        async def post(self, url: str, json: dict) -> _Resp:
            return _Resp()

    result = asyncio.run(_http_tool(_Client()).ainvoke({}))
    assert "HTTP 500" in result and "boom" in result


def test_http_tool_returns_a_transport_error_as_text() -> None:
    import httpx

    class _Client:
        async def post(self, url: str, json: dict) -> Any:
            raise httpx.ConnectError("connection refused")

    result = asyncio.run(_http_tool(_Client()).ainvoke({}))
    assert "could not be reached" in result


# --- prompt composition ----------------------------------------------------


def test_prompt_surface_note(fake_adapter) -> None:
    web = assemble_session_prompt(fake_adapter, surface="web")
    assert "web application" in web
    tui = assemble_session_prompt(fake_adapter, surface="tui")
    assert "web application" not in tui


def test_prompt_extra_fragments(fake_adapter) -> None:
    prompt = assemble_session_prompt(fake_adapter, extra_fragments=["MARKER FRAGMENT", "  "])
    assert "MARKER FRAGMENT" in prompt


# --- build_agent composition ----------------------------------------------


def _capture_create_deep_agent(monkeypatch) -> dict:
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(builder, "create_deep_agent", fake)
    return captured


def test_build_agent_composes_matching_surface(session: Session, monkeypatch) -> None:
    captured = _capture_create_deep_agent(monkeypatch)
    cap = Capability(
        name="ext",
        tools=(lambda _s: _demo_tool,),
        prompt_fragment="EXTENSION FRAGMENT",
        surfaces=("web",),
    )
    builder.build_agent(session, surface="web", extensions=[cap])
    assert "_demo_tool" in [t.name for t in captured["tools"]]
    assert "EXTENSION FRAGMENT" in captured["system_prompt"]


def test_build_agent_skips_other_surface(session: Session, monkeypatch) -> None:
    captured = _capture_create_deep_agent(monkeypatch)
    cap = Capability(
        name="ext",
        tools=(lambda _s: _demo_tool,),
        prompt_fragment="EXTENSION FRAGMENT",
        surfaces=("web",),
    )
    builder.build_agent(session, surface="tui", extensions=[cap])
    assert "_demo_tool" not in [t.name for t in captured["tools"]]


def test_build_agent_defaults_surface_to_the_session(session: Session, monkeypatch) -> None:
    """No explicit ``surface=`` falls back to what the session was created for."""
    captured = _capture_create_deep_agent(monkeypatch)
    session.surface = "web"
    cap = Capability(
        name="ext",
        tools=(lambda _s: _demo_tool,),
        surfaces=("web",),
    )
    builder.build_agent(session, extensions=[cap])
    assert "_demo_tool" in [t.name for t in captured["tools"]]
    assert "EXTENSION FRAGMENT" not in captured["system_prompt"]


def test_build_agent_default_has_base_tools(session: Session, monkeypatch, tmp_path: Path) -> None:
    captured = _capture_create_deep_agent(monkeypatch)
    builder.build_agent(session)
    names = [t.name for t in captured["tools"]]
    assert "run_julia" in names and "plot_julia" in names
