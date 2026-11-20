"""A session keeps working when its simulator's warm package cannot be precompiled.

The warm package only makes the first solve and plot fast. Where Julia cannot build
it, every ``using`` of it spends about a minute failing, so: a session loads without
it, nothing tries it twice, and the user is told instead of reading "done".
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from makker.agent.plot_julia import make_plot_julia_tool
from makker.interfaces.cli import doctor, main
from makker.julia import requirements
from makker.julia.session import EvalResult
from makker.lab.fakes import make_fake_adapter
from makker.session import Session
from makker.simulators import env_setup
from makker.simulators.warmup import fallback_load_statement, load_statement, start_warmup
from makker.workspace import (
    WARM_FAILURE_MARKER,
    mark_dependency_source,
    mark_env_precompiled,
    record_warm_failure,
    warm_failure_is_current,
    workspace_julia_env,
)

WARM = "MakkerFakeSim"
PRECOMPILE_ERROR = f'Failed to precompile {WARM} [4b91d94b] to "/depot/compiled/jl_x".'


@pytest.fixture(autouse=True)
def julia(monkeypatch: pytest.MonkeyPatch):
    """Pretend a Julia is installed; call the fixture to pretend it was updated."""

    def install(version: str) -> None:
        numbers = tuple(int(part) for part in version.split("."))
        check = requirements.JuliaCheck(found=True, version=numbers)
        monkeypatch.setattr(requirements, "check_julia", lambda *a, **k: check)

    install("1.12.6")
    return install


# ---- the session side ------------------------------------------------------


class _Kernel:
    """Julia in which the warm package fails to load and everything else works."""

    def __init__(self, *, helpers_error: str | None = None) -> None:
        self.calls: list[str] = []
        self.helpers_error = helpers_error
        self.release = asyncio.Event()
        self.release.set()

    async def eval(self, code: str, on_chunk: object = None) -> EvalResult:
        self.calls.append(code)
        if WARM in code:
            await self.release.wait()
            return EvalResult(output="", error=PRECOMPILE_ERROR)
        if self.helpers_error and code.endswith("using Makker"):
            return EvalResult(output="", error=self.helpers_error)
        if "MakkerPlots.capture" in code:
            from PIL import Image

            line = next(line for line in code.splitlines() if "path = raw" in line)
            Image.new("RGB", (8, 8)).save(line[line.index('raw"') + 4 : line.rindex('"')])
        return EvalResult(output="")

    @property
    def warm_attempts(self) -> int:
        return sum(WARM in call for call in self.calls)


def _session(tmp_path: Path, kernel: _Kernel) -> Session:
    adapter = make_fake_adapter(tmp_path, warm_package=WARM)
    return Session.create(julia=kernel, state_root=tmp_path, simulator=adapter)


async def _plot(session: Session, slot: str = "a") -> str:
    call = {"type": "tool_call", "name": "plot_julia", "id": slot}
    msg = await make_plot_julia_tool(session).ainvoke(
        {**call, "args": {"code": "lines(1:3)", "slot": slot}}
    )
    return str(msg.content)


def test_fallback_loads_the_simulator_then_the_shared_package() -> None:
    # The order the warm package brings things up in itself; capabilities last.
    assert fallback_load_statement("FakePkg", ("Cap",)) == (
        "try; @eval import FakePkg; catch; end\nusing Makker, Cap"
    )
    assert fallback_load_statement("") == "using Makker"


async def test_a_plot_is_made_without_the_warm_package_and_it_is_tried_once(
    tmp_path: Path,
) -> None:
    kernel = _Kernel()
    session = _session(tmp_path, kernel)
    told: list[str] = []
    session.warm.on_unavailable = told.append

    replies = [await _plot(session, slot) for slot in ("a", "b", "c")]

    # What the model reads is an ordinary result, with nothing in it to repair.
    assert all(reply.startswith("saved plot to") for reply in replies)
    assert kernel.calls[:2] == [load_statement(WARM), fallback_load_statement("FakePkg")]
    assert kernel.warm_attempts == 1
    assert told == [PRECOMPILE_ERROR]


async def test_the_warm_up_falls_back_and_a_plot_waits_for_its_verdict(tmp_path: Path) -> None:
    # The plot arrives while the warm-up's attempt is still failing. It must not
    # read "available" and queue a second attempt behind it.
    kernel = _Kernel()
    kernel.release.clear()
    session = _session(tmp_path, kernel)
    warmup = start_warmup(kernel, session.warm, ["Cap"])
    await asyncio.sleep(0)
    plot = asyncio.create_task(_plot(session))
    await asyncio.sleep(0)
    kernel.release.set()

    assert (await plot).startswith("saved plot to")
    await warmup
    assert kernel.calls[:2] == [
        load_statement(WARM, ["Cap"]),
        fallback_load_statement("FakePkg", ["Cap"]),
    ]
    assert kernel.warm_attempts == 1


async def test_a_session_that_knows_the_package_fails_never_tries_it(tmp_path: Path) -> None:
    kernel = _Kernel()
    session = _session(tmp_path, kernel)
    session.warm.available = False

    await start_warmup(kernel, session.warm)
    assert (await _plot(session)).startswith("saved plot to")
    assert kernel.warm_attempts == 0


async def test_helpers_that_cannot_load_at_all_are_still_an_error(tmp_path: Path) -> None:
    kernel = _Kernel(helpers_error="ArgumentError: Package Makker not found")
    reply = await _plot(_session(tmp_path, kernel))
    assert reply.startswith("ERROR: failed to load Makker plot helpers")
    assert "Package Makker not found" in reply


# ---- the env's record of the failure ---------------------------------------


@pytest.fixture
def env(tmp_path: Path) -> Path:
    """A managed workspace env that declares the shared and the warm package."""

    env = workspace_julia_env(tmp_path / "ws")
    (env / WARM / "src").mkdir(parents=True)
    (env / WARM / "src" / f"{WARM}.jl").write_text("module M end\n", encoding="utf-8")
    (env / "Project.toml").write_text(
        f'[deps]\nFoo = "u1"\nMakker = "u2"\n{WARM} = "u3"\n', encoding="utf-8"
    )
    (env / "Manifest.toml").write_text("# resolved\n", encoding="utf-8")
    return env


def test_a_recorded_failure_holds_until_something_it_depends_on_changes(env: Path, julia) -> None:
    assert not warm_failure_is_current(env)
    record_warm_failure(env)
    assert warm_failure_is_current(env)

    julia("1.12.7")
    assert not warm_failure_is_current(env)  # Julia was updated
    julia("1.12.6")
    assert warm_failure_is_current(env)

    (env / "Manifest.toml").write_text("# resolved differently\n", encoding="utf-8")
    assert not warm_failure_is_current(env)  # the dependencies moved

    record_warm_failure(env)
    (env / WARM / "src" / f"{WARM}.jl").write_text("module Changed end\n", encoding="utf-8")
    assert not warm_failure_is_current(env)  # the package source changed


def test_precompile_asks_about_the_package_then_leaves_a_failed_one_out(env: Path, julia) -> None:
    asked = env_setup._precompile_cmds(env)
    assert asked[0] == env_setup._PRECOMPILE
    assert f'Base.identify_package("{WARM}")' in asked[1] and "isprecompiled" in asked[1]

    # On record as failing: every other dependency is named, and it is not.
    record_warm_failure(env)
    [skipping] = env_setup._precompile_cmds(env)
    assert 'Pkg.precompile(["Foo", "Makker"])' in skipping

    julia("1.12.7")  # the record no longer describes the env, so it is tried again
    assert env_setup._precompile_cmds(env) == asked


def test_julias_report_records_a_failure_or_clears_one(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = env / env_setup.WARM_REPORT_FILENAME

    report.write_text("failed", encoding="utf-8")
    env_setup._settle_warm_package(env)
    assert warm_failure_is_current(env) and not report.exists()
    told = capsys.readouterr().err
    assert "fast start-up could not be prepared on Julia 1.12.6" in told
    assert "Makker works as usual" in told and "Updating Julia may fix it" in told

    env_setup._settle_warm_package(env)  # no report: nothing is concluded
    assert warm_failure_is_current(env)

    report.write_text("ready", encoding="utf-8")
    env_setup._settle_warm_package(env)
    assert not (env / WARM_FAILURE_MARKER).exists()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(("installed", "retried"), [("1.12.6", False), ("1.12.7", True)])
def test_launch_retries_a_failed_package_only_after_a_change(
    env: Path, monkeypatch: pytest.MonkeyPatch, julia, installed: str, retried: bool
) -> None:
    record_warm_failure(env)
    mark_env_precompiled(env)
    mark_dependency_source(env, None)
    precompiled: list[Path] = []
    monkeypatch.setattr(env_setup, "precompile_env", precompiled.append)

    julia(installed)
    env_setup._ensure_env_warmed(env.parent.parent, env, "fakesim")

    assert precompiled == ([env] if retried else [])


def test_a_session_records_only_a_failed_build(env: Path) -> None:
    env_setup.note_warm_load_failure(env, "InterruptException:")  # may work next time
    assert not warm_failure_is_current(env)
    env_setup.note_warm_load_failure(env, PRECOMPILE_ERROR)
    assert warm_failure_is_current(env)


# ---- what the commands say -------------------------------------------------


@pytest.mark.parametrize("report", ["failed", "ready"])
def test_init_only_calls_the_precompile_done_when_the_warm_package_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], report: str
) -> None:
    class _Result:
        returncode = 0
        stdout = "julia version 1.12.6\n"
        stderr = ""

    def _fake_run(argv, **_kwargs):
        if env_setup.WARM_REPORT_FILENAME in argv[-1]:
            project = next(arg for arg in argv if arg.startswith("--project=")).partition("=")[2]
            Path(project, env_setup.WARM_REPORT_FILENAME).write_text(report, encoding="utf-8")
        return _Result()

    monkeypatch.setattr(env_setup.shutil, "which", lambda _: "/usr/bin/julia")
    monkeypatch.setattr(env_setup.subprocess, "run", _fake_run)
    ws = tmp_path / "ws"
    ws.mkdir()
    code = main(["init", "--sim", "jutuldarcy", "--workspace", str(ws), "--state-home", str(ws)])
    out, err = capsys.readouterr()

    assert code == 0, err  # the workspace is usable either way
    failed = report == "failed"
    assert ("precompile:    done (" in out) is not failed
    assert ("done, except the fast start-up" in out) is failed
    assert ("fast start-up could not be prepared" in err) is failed


def test_doctor_shows_a_warm_package_on_record_as_not_building(env: Path) -> None:
    lines: list[tuple[str, ...]] = []

    class _Report:
        def line(self, *parts: str) -> None:
            lines.append(parts)

    doctor._check_fast_startup(_Report(), env, "1.12.6")
    assert lines == []

    record_warm_failure(env)
    doctor._check_fast_startup(_Report(), env, "1.12.6")
    [(status, label, detail, fix)] = lines
    assert (status, label) == (doctor.WARN, "Fast start-up")  # never a failure
    assert "could not be prepared on Julia 1.12.6" in detail and "Makker works" in detail
    assert "Updating Julia may fix it" in fix
