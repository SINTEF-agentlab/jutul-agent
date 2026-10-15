"""``start_core``: the kernel half of the session bootstrap, usable without an agent.

``SessionHost.start`` builds the agent on top of a core (its own tests cover
that path); these pin what a front end that brings its own agent relies on: a
core stands up the session with no agent built, defers the warm-up until asked,
and tears down in the safe order.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from jutul_agent.agent.capabilities import Capability
from jutul_agent.lab.fakes import FakeJulia, make_fake_adapter


@pytest.fixture
def fake_kernel_world(monkeypatch) -> dict[str, Any]:
    """Patch out Julia, env preparation and package resolution; record the calls."""
    import jutul_agent.juliakernel as juliakernel_mod
    from jutul_agent.agent import builder as builder_mod
    from jutul_agent.julia import requirements as requirements_mod
    from jutul_agent.simulators import env_setup as env_setup_mod

    world: dict[str, Any] = {"kernel": FakeJulia(), "prepared": [], "configs": []}
    monkeypatch.setattr(requirements_mod, "require_julia", lambda *a, **kw: None)

    def _kernel(config):
        world["configs"].append(config)
        return world["kernel"]

    monkeypatch.setattr(juliakernel_mod, "JuliaKernel", _kernel)
    monkeypatch.setattr(builder_mod, "resolve_package_sources", lambda project: ["pkg-src"])

    def _no_agent(*_a, **_kw):
        raise AssertionError("start_core must not build an agent")

    monkeypatch.setattr(builder_mod, "build_agent", _no_agent)

    def _prepare(simulator, *, workspace, julia_project, sim_name, dependencies):
        world["prepared"].append(dependencies)

    monkeypatch.setattr(env_setup_mod, "prepare_workspace_env", _prepare)
    return world


async def test_start_core_opens_a_session_without_an_agent(
    fake_kernel_world: dict[str, Any], tmp_path: Path
) -> None:
    from jutul_agent.session_core import start_core

    workspace = tmp_path / "ws"
    workspace.mkdir()
    dep = tmp_path / "Cap" / "Project.toml"
    cap = Capability(name="cap", dependencies=(dep,), warm_packages=("CapPkg",))

    core = await start_core(
        simulator=make_fake_adapter(tmp_path),
        surface="mcp",
        workspace=workspace,
        state_root=tmp_path,
        extensions=[cap],
        discover=False,
        virtual_display=False,
    )
    try:
        assert core.session.surface == "mcp"
        assert core.workspace == workspace
        assert core.extensions == [cap]
        assert core.package_sources == ["pkg-src"]
        # The capability's Julia dependencies went into the env it starts against.
        assert fake_kernel_world["prepared"] == [[dep]]
        # The tools' Julia is the yielding wrapper around the raw kernel.
        assert core.kernel is fake_kernel_world["kernel"]
        assert core.session.julia is not core.kernel
        # The warm-up waits for the caller.
        assert core.warmup_task is None
    finally:
        await core.aclose()


async def test_the_kernel_starts_without_credentials(
    fake_kernel_world: dict[str, Any], tmp_path: Path
) -> None:
    """Julia runs the model's code, so the kernel withholds every credential."""
    from jutul_agent.credentials import is_secret_env_var
    from jutul_agent.session_core import start_core

    workspace = tmp_path / "ws"
    workspace.mkdir()
    core = await start_core(
        simulator=make_fake_adapter(tmp_path),
        surface="tui",
        workspace=workspace,
        state_root=tmp_path,
        discover=False,
        virtual_display=False,
    )
    try:
        [config] = fake_kernel_world["configs"]
        assert config.env_filter is is_secret_env_var
    finally:
        await core.aclose()


async def test_start_warmup_is_idempotent_and_aclose_cancels_it(
    fake_kernel_world: dict[str, Any], tmp_path: Path, monkeypatch
) -> None:
    from jutul_agent.session_core import start_core
    from jutul_agent.simulators import warmup as warmup_mod
    from jutul_agent.trace import TraceLog
    from jutul_agent.trace.schema import SESSION_END

    calls: list[tuple[Any, ...]] = []

    def _start_warmup(julia, warm_package, packages=(), code=(), abandonable=None):
        calls.append((julia, warm_package, list(packages)))
        return asyncio.create_task(asyncio.sleep(3600))

    monkeypatch.setattr(warmup_mod, "start_warmup", _start_warmup)
    cap = Capability(name="cap", warm_packages=("CapPkg",))
    core = await start_core(
        simulator=make_fake_adapter(tmp_path),
        surface="mcp",
        workspace=tmp_path,
        state_root=tmp_path,
        extensions=[cap],
        discover=False,
        virtual_display=False,
        prepare_env=False,
    )
    core.start_warmup()
    core.start_warmup()
    task = core.warmup_task
    assert task is not None
    # Once, against the raw kernel, with the capability's package included.
    assert calls == [(core.kernel, core.session.simulator.warm_package, ["CapPkg"])]

    await core.aclose()
    assert task.cancelled()
    with TraceLog.open_readonly(core.session.state_dir / "trace.sqlite") as log:
        assert [e.kind for e in log.iter_events()][-1] == SESSION_END


def _record_warmups(monkeypatch, kernel: Any) -> list[tuple[asyncio.Task[None], asyncio.Event]]:
    """Patch the warm-up to record each one started, against ``kernel``, as a task
    that runs until cancelled, with the event that would mark it abandonable."""
    from jutul_agent.simulators import warmup as warmup_mod

    started: list[tuple[asyncio.Task[None], asyncio.Event]] = []

    def _start_warmup(julia, warm_package, packages=(), code=(), abandonable=None):
        assert julia is kernel
        started.append((asyncio.create_task(asyncio.sleep(3600)), abandonable))
        return started[-1][0]

    monkeypatch.setattr(warmup_mod, "start_warmup", _start_warmup)
    return started


async def _fake_core(tmp_path: Path) -> Any:
    from jutul_agent.session_core import start_core

    return await start_core(
        simulator=make_fake_adapter(tmp_path),
        surface="mcp",
        workspace=tmp_path,
        state_root=tmp_path,
        discover=False,
        virtual_display=False,
        prepare_env=False,
    )


async def test_a_restart_warms_the_fresh_process_again(
    fake_kernel_world: dict[str, Any], tmp_path: Path, monkeypatch
) -> None:
    """A restart takes all the warm-up did with the old process, so it runs again
    on the new one; what was left of the last is stopped, and the tools' wrapper
    yields to the new one as it did to the first."""
    kernel = fake_kernel_world["kernel"]
    started = _record_warmups(monkeypatch, kernel)
    core = await _fake_core(tmp_path)
    try:
        await kernel.reset()
        assert started == [], "nothing to redo before a warm-up has run"
        core.start_warmup()
        await core.session.julia.reset()  # as `reset_julia` does, through the wrapper
        await kernel.restart()  # as a cancel's recovery does, inside the kernel
        assert len(started) == 3
        (first, _), (second, _), (current, abandonable) = started
        assert core.warmup_task is current
        await asyncio.sleep(0)
        assert first.cancelled() and second.cancelled()
        assert not current.done()
        # Each restart's warm-up is the setup the kernel holds other calls for.
        assert len(kernel.setups) == 2
        await asyncio.sleep(0.01)
        assert kernel.setups[0].done(), "a stopped warm-up holds nothing up"
        assert not kernel.setups[1].done(), "calls wait while the fresh process loads"
        abandonable.set()
        await asyncio.sleep(0.01)
        assert kernel.setups[1].done()
        core.session.julia.drop_warmup()  # real work arriving once it is abandonable
        await asyncio.sleep(0)
        assert current.cancelled()
    finally:
        await core.aclose()


async def test_teardown_stops_the_current_warmup_and_the_rewarming(
    fake_kernel_world: dict[str, Any], tmp_path: Path, monkeypatch
) -> None:
    kernel = fake_kernel_world["kernel"]
    started = _record_warmups(monkeypatch, kernel)
    core = await _fake_core(tmp_path)
    core.start_warmup()
    await kernel.reset()
    current = core.warmup_task
    assert current is started[-1][0]
    await core.aclose()
    assert current.cancelled()
    await kernel.reset()
    assert len(started) == 2, "a restart after teardown must not warm again"
