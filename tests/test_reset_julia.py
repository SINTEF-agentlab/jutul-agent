"""Tests for the reset_julia tool and the stale-load hint on run_julia."""

from __future__ import annotations

from pathlib import Path

from jutul_agent.agent.tools import make_reset_julia_tool, make_run_julia_tool
from jutul_agent.julia.session import EvalResult
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter
from jutul_agent.session import Session


def _session(julia: FakeJulia, tmp_path: Path, sid: str = "reset-test") -> Session:
    return Session.create(
        julia=julia,
        state_root=tmp_path,
        simulator=make_fake_adapter(tmp_path),
        session_id=sid,
    )


async def _call(tool, name: str, **args) -> str:
    msg = await tool.ainvoke({"type": "tool_call", "name": name, "id": "c1", "args": args})
    return str(getattr(msg, "content", msg))


async def test_reset_julia_uses_cooperative_reset_when_healthy(tmp_path: Path) -> None:
    julia = FakeJulia()
    tool = make_reset_julia_tool(_session(julia, tmp_path))
    out = await _call(tool, "reset_julia")
    assert julia.reset_count == 1
    assert julia.restart_count == 0  # healthy session: no full restart needed
    assert "restarted" in out.lower()


async def test_reset_julia_falls_back_to_restart_when_session_dead(tmp_path: Path) -> None:
    # A killed/crashed process makes the cooperative reset raise; the tool must
    # still recover by force-restarting the subprocess.
    julia = FakeJulia()

    async def _dead_reset() -> EvalResult:
        raise RuntimeError("Connection closed")

    julia.reset = _dead_reset  # type: ignore[method-assign]
    tool = make_reset_julia_tool(_session(julia, tmp_path))
    out = await _call(tool, "reset_julia")
    assert julia.restart_count == 1
    assert "restarted" in out.lower()


async def test_run_julia_reports_recoverable_error_on_dead_session(tmp_path: Path) -> None:
    def _boom(code: str) -> EvalResult:
        raise RuntimeError("Connection closed")

    julia = FakeJulia(eval_handler=_boom)
    tool = make_run_julia_tool(_session(julia, tmp_path))
    out = await _call(tool, "run_julia", code="1 + 1")
    assert "reset_julia" in out
    assert "unavailable" in out.lower()


async def test_run_julia_appends_reset_hint_on_stale_load(tmp_path: Path) -> None:
    warning = (
        "  1 dependency precompiled but a different version is currently loaded. "
        "Restart julia to access the new version."
    )
    julia = FakeJulia(eval_handler=lambda code: EvalResult(output=warning))
    tool = make_run_julia_tool(_session(julia, tmp_path))
    out = await _call(tool, "run_julia", code='Pkg.add("X")')
    assert "reset_julia" in out
    assert warning in out  # original output preserved


async def test_run_julia_no_hint_when_no_stale_load(tmp_path: Path) -> None:
    julia = FakeJulia(eval_handler=lambda code: EvalResult(output="42"))
    tool = make_run_julia_tool(_session(julia, tmp_path))
    out = await _call(tool, "run_julia", code="6 * 7")
    assert "reset_julia" not in out
    assert "Pkg.update" not in out
    assert out.strip() == "42"


async def test_run_julia_appends_pkg_update_hint_on_precompile_failure(tmp_path: Path) -> None:
    err = "LoadError: Failed to precompile GeoStats [dcc97b0b] to ..."
    julia = FakeJulia(eval_handler=lambda code: EvalResult(output="", error=err))
    tool = make_run_julia_tool(_session(julia, tmp_path))
    out = await _call(tool, "run_julia", code='Pkg.add("GeoStats")')
    assert "Pkg.update()" in out
    assert err in out  # original error preserved


async def test_run_julia_appends_reset_hint_on_worker_module_desync(tmp_path: Path) -> None:
    err = (
        "On worker 1:\n"
        'KeyError: key Base.PkgId(Base.UUID("92933f4c-..."), "ProgressMeter") not found'
    )
    julia = FakeJulia(eval_handler=lambda code: EvalResult(output="", error=err))
    tool = make_run_julia_tool(_session(julia, tmp_path))
    out = await _call(tool, "run_julia", code="using GeoStats")
    assert "reset_julia" in out
    assert err in out  # original error preserved
