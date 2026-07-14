"""Tests for the example demo app's Python wiring (no Julia or browser needed)."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from fakes import FakeJulia, make_fake_adapter
from jutul_agent.session import Session

_DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "demo-app" / "demo.py"


def _load_demo():
    spec = importlib.util.spec_from_file_location("demo_app", _DEMO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


demo = _load_demo()


def _session(tmp_path: Path) -> Session:
    return Session.create(
        julia=FakeJulia(), simulator=make_fake_adapter(tmp_path), state_root=tmp_path
    )


def test_capability_is_web_with_two_tools() -> None:
    cap = demo.demo_capability()
    assert cap.surfaces == ("web",)
    assert len(cap.tools) == 2
    assert cap.prompt_fragment


def test_set_param_emits_ui_event(tmp_path: Path) -> None:
    session = _session(tmp_path)
    set_param = demo._make_set_param_tool(session)
    out = asyncio.run(set_param.ainvoke({"p": 5}))
    assert "5" in out
    ui_events = [e for e in session.trace.iter_events() if e.kind == "ui"]
    assert ui_events[-1].payload == {"action": "set_param", "payload": {"p": 5.0}}


def test_plot_response_records_html_artifact(tmp_path: Path) -> None:
    session = _session(tmp_path)
    plot_response = demo._make_plot_tool(session)
    out = asyncio.run(plot_response.ainvoke({"p": 3}))
    assert "3" in out
    artifacts = [e for e in session.trace.iter_events() if e.kind == "artifact"]
    assert artifacts[-1].payload["mime"] == "text/html"
    assert artifacts[-1].payload["path"].endswith(".html")
    # The tool drove the Julia export.
    assert any("Bonito.export_static" in code for code in session.julia.calls)


def test_ensure_env_refreshes_capability_dependencies(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    captured: dict[str, object] = {}

    def _fake_prepare(adapter, *, workspace, julia_project, sim_name, dependencies):
        captured["workspace"] = workspace
        captured["julia_project"] = julia_project
        captured["sim_name"] = sim_name
        captured["dependencies"] = dependencies

    monkeypatch.setattr(demo, "WORKSPACE", workspace)
    monkeypatch.setattr(demo, "prepare_workspace_env", _fake_prepare)

    demo.ensure_env()

    assert captured["workspace"] == workspace
    assert captured["sim_name"] == "demo"
    assert captured["dependencies"] == dict(demo.demo_capability().dependencies)
