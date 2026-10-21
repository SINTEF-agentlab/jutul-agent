"""Updating jutul-agent and the folder's Julia environment from the web UI."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from jutul_agent.interfaces.server import updates
from jutul_agent.interfaces.server.access import is_loopback_host, request_is_local
from jutul_agent.interfaces.server.app import create_app
from jutul_agent.interfaces.server.manager import SessionManager
from jutul_agent.interfaces.server.updates import Updater, UpdateRefused
from jutul_agent.paths import state_home
from jutul_agent.update_check import InstallInfo

LOCAL = {"host": "127.0.0.1:8742", "origin": "http://127.0.0.1:8742"}


@pytest.mark.parametrize(
    ("headers", "local"),
    [
        (LOCAL, True),
        ({"host": "localhost:8742", "origin": "http://localhost:8742"}, True),
        ({"host": "[::1]:8742", "origin": "http://[::1]:8742"}, True),
        # Another site open in the same browser.
        ({"host": "127.0.0.1:8742", "origin": "https://example.com"}, False),
        # A site that points its own domain at this computer.
        ({"host": "evil.example:8742", "origin": "http://evil.example:8742"}, False),
        # No Origin: not a browser page's request.
        ({"host": "127.0.0.1:8742"}, False),
        # Served to the network, then reached by its address.
        ({"host": "192.168.1.5:8742", "origin": "http://192.168.1.5:8742"}, False),
    ],
)
def test_only_this_computers_own_page_may_update(headers, local) -> None:
    assert request_is_local(headers) is local


@pytest.mark.parametrize(
    ("host", "loopback"),
    [("127.0.0.1", True), ("localhost", True), ("::1", True), ("0.0.0.0", False), ("", False)],
)
def test_loopback_bind_addresses(host, loopback) -> None:
    assert is_loopback_host(host) is loopback


@pytest.fixture
def uv_tool(tmp_path, monkeypatch):
    """Run as a uv tool install with uv on PATH, on a POSIX system."""
    prefix = tmp_path / "tool-env"
    prefix.mkdir()
    (prefix / "uv-receipt.toml").write_text("[tool]\n", encoding="utf-8")
    monkeypatch.setattr(updates.sys, "prefix", str(prefix))
    monkeypatch.setattr(updates, "_on_windows", lambda: False)
    monkeypatch.setattr(updates, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(updates.shutil, "which", lambda _name: "/usr/bin/uv")
    return prefix


def _updater(tmp_path, *, local=True, restartable=True) -> Updater:
    updater = Updater(tmp_path, local=local)
    if restartable:
        updater.bind_server(lambda: None)
    return updater


@pytest.mark.parametrize("windows", [False, True])
def test_a_uv_tool_install_can_update_itself(tmp_path, uv_tool, monkeypatch, windows) -> None:
    monkeypatch.setattr(updates, "_on_windows", lambda: windows)
    updater = _updater(tmp_path)
    assert updater.install_blocker() is None
    # Windows can't replace a running program, so it updates in a separate window.
    assert updater.status()["separate_window"] is windows


@pytest.mark.parametrize(
    ("change", "blocker"),
    [
        ("remote", "not-local"),
        ("editable", "dev-checkout"),
        ("pip", "not-uv-tool"),
        ("no-uv", "no-uv"),
        ("embedded", "no-restart"),
    ],
)
def test_installs_that_cannot_update_themselves(
    tmp_path, uv_tool, monkeypatch, change, blocker
) -> None:
    if change == "editable":
        monkeypatch.setattr(updates, "install_info", lambda: InstallInfo("editable"))
    elif change == "pip":
        (uv_tool / "uv-receipt.toml").unlink()
    elif change == "no-uv":
        monkeypatch.setattr(updates.shutil, "which", lambda _name: None)
    updater = _updater(tmp_path, local=change != "remote", restartable=change != "embedded")

    assert updater.install_blocker() == blocker
    status = updater.status()
    assert status["can_install"] is False
    assert status["install_note"]  # says what to do instead


class _Manager:
    """Enough of a SessionManager for the updater."""

    def __init__(self, busy: bool = False) -> None:
        self.busy = busy
        self.closed = False

    def any_busy(self) -> bool:
        return self.busy

    async def aclose(self) -> None:
        self.closed = True


def _join_update_thread() -> None:
    # The state flips to "restarting" before the server is stopped, so wait for
    # the whole job rather than for the state.
    for thread in threading.enumerate():
        if thread.name == "jutul-agent-update":
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.fixture
def run_update(monkeypatch):
    """Start an update whose commands exit with ``codes``, and wait for it.

    ``versions`` is the installed jutul-agent version before and after.
    """

    def run(updater, manager, *, install, sysimage=None, codes=(), versions=("1.0", "1.0")):
        ran: list[list[str]] = []
        exits = iter(codes)

        def fake_step(job, argv, *, shown=None):
            ran.append(argv)
            job.add(f"$ {shown or ' '.join(argv)}")
            return next(exits)

        monkeypatch.setattr(updater, "_step", fake_step)
        updater._version_at_start = versions[0]
        monkeypatch.setattr(updates, "_installed_version", lambda: versions[1])
        asyncio.run(updater.start(manager, install=install, sysimage=sysimage))
        _join_update_thread()
        return ran

    return run


@pytest.fixture(autouse=True)
def _no_restart_delay(monkeypatch):
    monkeypatch.setattr(updates, "_RESTART_DELAY_SECONDS", 0)


def test_environment_only_update(tmp_path, run_update) -> None:
    updater, manager = _updater(tmp_path), _Manager()
    ran = run_update(updater, manager, install=False, sysimage="off", codes=[0])

    assert manager.closed  # sessions are closed before anything changes
    env_step = [sys.executable, "-m", "jutul_agent", "upgrade", "--env-only"]
    assert ran == [[*env_step, "--workspace", str(tmp_path), "--sysimage", "off"]]
    assert updater._job.state == "done"
    assert not updater.restart_requested


def test_new_version_then_environment_then_restart(tmp_path, uv_tool, run_update) -> None:
    stopped = []
    updater = _updater(tmp_path)
    updater.bind_server(lambda: stopped.append(True))
    ran = run_update(updater, _Manager(), install=True, codes=[0, 0], versions=("1.0", "1.1"))

    assert ran[0] == ["uv", "tool", "upgrade", "jutul-agent"]
    # The interpreter now imports the new version, so its templates decide.
    assert ran[1][:5] == [sys.executable, "-m", "jutul_agent", "upgrade", "--env-only"]
    assert updater._job.state == "restarting"
    assert updater.restart_requested and stopped == [True]
    assert not updater.handed_off


def test_no_restart_when_already_up_to_date(tmp_path, uv_tool, run_update) -> None:
    updater = _updater(tmp_path)
    run_update(updater, _Manager(), install=True, codes=[0, 0])
    assert updater._job.state == "done" and not updater.restart_requested


def test_a_failed_upgrade_changes_nothing_else(tmp_path, uv_tool, run_update) -> None:
    updater = _updater(tmp_path)
    ran = run_update(updater, _Manager(), install=True, codes=[2])
    assert len(ran) == 1  # the environment was not touched
    assert updater._job.state == "failed" and "could not be upgraded" in updater._job.error
    assert not updater.restart_requested


def test_restarts_into_the_new_version_even_if_the_environment_failed(
    tmp_path, uv_tool, run_update
) -> None:
    # This process runs old code over new files, so it must not stay up; the
    # restarted server reports the environment instead.
    updater = _updater(tmp_path)
    run_update(updater, _Manager(), install=True, codes=[0, 1], versions=("1.0", "1.1"))
    assert updater._job.state == "restarting" and updater.restart_requested
    assert "Julia environment could not be updated" in updater._job.error


def test_the_restarted_server_reports_how_the_update_went(tmp_path, uv_tool, run_update) -> None:
    # The page reloads into the new server, so the failure has to be reported there.
    run_update(_updater(tmp_path), _Manager(), install=True, codes=[0, 1], versions=("1.0", "1.1"))
    restarted = _updater(tmp_path)
    restarted.report_last_update()
    job = restarted.status()["job"]
    assert job["state"] == "failed" and "Julia environment" in job["error"]
    assert job["lines"] == ["$ uv tool upgrade jutul-agent", "$ jutul-agent upgrade --env-only"]
    assert not restarted.running  # sessions may start
    # Reported once: a later restart of the server says nothing.
    later = _updater(tmp_path)
    later.report_last_update()
    assert later.status()["job"] is None


@pytest.fixture
def helper(monkeypatch):
    """Capture the plans handed to the update helper instead of starting it."""
    plans: list[dict] = []
    monkeypatch.setattr(updates, "start_helper", plans.append)
    return plans


def _start_and_wait(updater, manager, **kwargs) -> None:
    asyncio.run(updater.start(manager, **kwargs))
    _join_update_thread()


def test_windows_hands_the_update_to_a_helper_and_exits(
    tmp_path, uv_tool, monkeypatch, helper
) -> None:
    monkeypatch.setattr(updates, "_on_windows", lambda: True)
    launcher = tmp_path / "bin" / "jutul-agent.exe"
    launcher.parent.mkdir()
    launcher.write_bytes(b"")
    # As the launcher's own script leaves it: without ".exe".
    monkeypatch.setattr(sys, "argv", [str(launcher.with_suffix("")), "--port", "9000"])
    stopped, manager = [], _Manager()
    updater = _updater(tmp_path)
    updater.bind_server(lambda: stopped.append(True))
    monkeypatch.setattr(updater, "_step", lambda *a, **k: pytest.fail("the helper runs it"))

    _start_and_wait(updater, manager, install=True, sysimage="off")

    [plan] = helper
    # The server, and the launcher it runs under, which uv replaces.
    assert plan["wait_for"] == [os.getpid(), os.getppid()]
    assert plan["free"] == [str(launcher)]
    upgrade, env = (step["argv"] for step in plan["steps"])
    assert upgrade == ["/usr/bin/uv", "tool", "upgrade", "jutul-agent"]
    assert env[0] == str(launcher)
    assert env[1:] == ["upgrade", "--env-only", "--workspace", str(tmp_path), "--sysimage", "off"]
    assert plan["relaunch"] == {
        "argv": [str(launcher), "--port", "9000"],
        "cwd": os.getcwd(),
        "env": {updates.RESTARTED_ENV_VAR: "1"},
    }
    assert Path(plan["log"]).parent == updates._update_dir()
    # Sessions closed, then the server stops without restarting itself.
    assert manager.closed and stopped == [True]
    assert updater._job.state == "restarting" and updater.status()["separate_window"]
    assert updater.handed_off and not updater.restart_requested


def test_the_separate_window_can_be_forced_for_testing(
    tmp_path, uv_tool, monkeypatch, helper
) -> None:
    monkeypatch.setenv(updates.SEPARATE_WINDOW_ENV_VAR, "1")
    monkeypatch.setattr(sys, "argv", ["/somewhere/jutul_agent/__main__.py", "web"])
    _start_and_wait(_updater(tmp_path), _Manager(), install=True, sysimage=None)

    [plan] = helper
    # Not started from a launcher: nothing else to wait for, and `python -m` runs it.
    assert plan["wait_for"] == [os.getpid()] and plan["free"] == []
    assert plan["relaunch"]["argv"] == [sys.executable, "-m", "jutul_agent", "web"]


def test_a_helper_that_cannot_start_leaves_the_server_running(
    tmp_path, uv_tool, monkeypatch
) -> None:
    def broken(_plan):
        raise OSError("no console")

    monkeypatch.setattr(updates, "_on_windows", lambda: True)
    monkeypatch.setattr(updates, "start_helper", broken)
    stopped = []
    updater = _updater(tmp_path)
    updater.bind_server(lambda: stopped.append(True))
    _start_and_wait(updater, _Manager(), install=True, sysimage=None)
    assert updater._job.state == "failed" and "no console" in updater._job.error
    assert not stopped and not updater.handed_off


def test_windows_updates_only_the_environment_in_place(
    tmp_path, uv_tool, monkeypatch, run_update, helper
) -> None:
    monkeypatch.setattr(updates, "_on_windows", lambda: True)
    updater = _updater(tmp_path)
    ran = run_update(updater, _Manager(), install=False, codes=[0])
    assert ran[0][:3] == [sys.executable, "-m", "jutul_agent"]
    assert helper == [] and updater._job.state == "done"


@pytest.mark.parametrize("reason", ["busy", "running", "remote", "blocked"])
def test_refusals(tmp_path, uv_tool, monkeypatch, reason) -> None:
    updater = _updater(tmp_path, local=reason != "remote")
    manager = _Manager(busy=reason == "busy")
    if reason == "running":
        updater._job = updates.UpdateJob(install=False)
    if reason == "blocked":
        monkeypatch.setattr(updates, "install_info", lambda: InstallInfo("editable"))
    with pytest.raises(UpdateRefused):
        asyncio.run(updater.start(manager, install=reason != "busy", sysimage=None))
    assert not manager.closed


def test_an_unexpected_error_ends_the_update(tmp_path, monkeypatch) -> None:
    # A job left "running" would keep every session from starting.
    def broken_step(job, argv, *, shown=None):
        raise RuntimeError("boom")

    updater = _updater(tmp_path)
    monkeypatch.setattr(updater, "_step", broken_step)
    asyncio.run(updater.start(_Manager(), install=False, sysimage=None))
    _join_update_thread()
    assert updater._job.state == "failed" and "boom" in updater._job.error


def test_a_starting_session_counts_as_busy() -> None:
    # Its kernel would load the environment while the update changes it.
    seen = []

    async def factory(**_kwargs):
        seen.append(manager.any_busy())
        raise RuntimeError("stop here")

    manager = SessionManager(host_factory=factory)
    with pytest.raises(RuntimeError):
        asyncio.run(manager.create(sim="jutuldarcy"))
    assert seen == [True]
    assert not manager.any_busy()


def test_a_step_streams_its_output(tmp_path) -> None:
    job = updates.UpdateJob(install=False)
    code = Updater(tmp_path, local=True)._step(
        job, [sys.executable, "-c", "print('one'); print('two')"], shown="demo"
    )
    assert code == 0
    assert job.lines == ["$ demo", "one", "two"]


def test_the_page_reads_new_lines_only(monkeypatch) -> None:
    monkeypatch.setattr(updates, "_MAX_LINES", 3)
    job = updates.UpdateJob(install=False)
    for i in range(5):
        job.add(str(i))
    assert job.snapshot(0)["lines"] == ["2", "3", "4"]  # the oldest were dropped
    assert job.snapshot(4) == {**job.snapshot(4), "lines": ["4"], "next": 5}


# ---------------------------------------------------------------------------
# Endpoints


def _app(updater: Updater, manager: SessionManager | None = None):
    async def unused_factory(**_kwargs):
        raise AssertionError("no session should start")

    manager = manager or SessionManager(host_factory=unused_factory)
    return create_app(manager, default_sim="jutuldarcy", updater=updater)


def test_update_endpoint_refuses_other_sites(tmp_path) -> None:
    updater = _updater(tmp_path)
    with TestClient(_app(updater)) as client:
        response = client.post("/update", json={}, headers={"origin": "https://example.com"})
    assert response.status_code == 403
    assert updater._job is None


def test_update_endpoint_starts_an_update(tmp_path, monkeypatch) -> None:
    started = []

    async def fake_start(manager, *, install, sysimage):
        started.append((install, sysimage))

    updater = _updater(tmp_path)
    monkeypatch.setattr(updater, "start", fake_start)
    with TestClient(_app(updater), base_url="http://127.0.0.1:8742") as client:
        response = client.post(
            "/update",
            json={"install": True, "sysimage": "rebuild"},
            headers={"origin": "http://127.0.0.1:8742"},
        )
        assert response.status_code == 202
        assert started == [(True, "rebuild")]
        # Only the two system image answers are accepted.
        bad = client.post(
            "/update", json={"sysimage": "rm -rf"}, headers={"origin": "http://127.0.0.1:8742"}
        )
        assert bad.status_code == 422


def test_sessions_wait_for_an_update(tmp_path) -> None:
    updater = _updater(tmp_path)
    updater._job = updates.UpdateJob(install=False)  # running
    with TestClient(_app(updater)) as client:
        assert client.post("/sessions", json={}).status_code == 409
        resumed = client.post("/sessions/2026-01-01-0000-abcd/resume", json={})
        assert resumed.status_code == 409 and "updating" in resumed.json()["detail"]


def test_an_outdated_environment_points_at_the_sidebar(tmp_path) -> None:
    from jutul_agent.simulators.env_setup import EnvOutdatedError

    async def outdated_factory(**_kwargs):
        raise EnvOutdatedError("old", [("JutulDarcy", "0.3.12", "0.4")])

    manager = SessionManager(host_factory=outdated_factory)
    with TestClient(_app(_updater(tmp_path), manager)) as client:
        response = client.post("/sessions", json={"model": "ollama:qwen3"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "has JutulDarcy 0.3.12" in detail and "Update environment" in detail


def test_version_tells_the_page_about_restarts_and_the_environment(tmp_path, monkeypatch) -> None:
    updater = _updater(tmp_path)
    monkeypatch.setattr(updater, "env_update_needed", lambda: True)
    with TestClient(_app(updater)) as client:
        info = client.get("/version").json()
        status = client.get("/update").json()
    assert info["boot"] == updates.BOOT_ID and info["env_update"] is True
    assert status["local"] is True and status["job"] is None


# ---------------------------------------------------------------------------
# The server process


def _web_args(**overrides):
    from jutul_agent.interfaces.cli import web

    # Keep the test's state folder: launching sets it from the arguments.
    args = web.build_parser().parse_args(["--sim", "jutuldarcy", "--state-home", str(state_home())])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def web_run(tmp_path, monkeypatch):
    """``web.run`` up to serving, with a uvicorn that returns at once."""
    import uvicorn

    from jutul_agent.interfaces.cli import web

    monkeypatch.chdir(tmp_path)
    calls = SimpleNamespace(
        checks=[], restarted=False, updater=None, ask_restart=False, hand_off=False, ctrl_c=False
    )

    class FakeServer:
        def __init__(self, config) -> None:
            self.started = True
            self.should_exit = False
            calls.updater = config.app.state.updater
            calls.app = config.app

        def run(self) -> None:
            calls.updater.restart_requested = calls.ask_restart
            calls.updater.handed_off = calls.hand_off
            if calls.ctrl_c:  # as uvicorn re-raises it once it has shut down
                raise KeyboardInterrupt

    def check(name):
        return lambda *a, **k: not calls.checks.append(name)

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr(web, "_env_versions_ok", check("env"))
    monkeypatch.setattr(web, "_sysimage_ready", check("image"))
    monkeypatch.setattr(updates, "restart_server", lambda: setattr(calls, "restarted", True))
    monkeypatch.setattr("jutul_agent.update_check.notify_at_launch", lambda: None)
    return web, calls


def test_the_server_restarts_after_an_update(web_run) -> None:
    web, calls = web_run
    calls.ask_restart = True
    assert web.run(_web_args()) == 0
    assert calls.restarted
    assert calls.updater.local  # served on 127.0.0.1 by default


def test_the_server_exits_once_a_helper_has_the_update(web_run, capsys) -> None:
    web, calls = web_run
    calls.hand_off = True
    assert web.run(_web_args()) == 0
    assert not calls.restarted  # the helper starts the new server
    assert "separate window" in capsys.readouterr().err


def test_ctrl_c_stops_the_server_cleanly(web_run) -> None:
    web, calls = web_run
    calls.ctrl_c = calls.ask_restart = True
    try:
        code = web.run(_web_args())
    except KeyboardInterrupt:  # escaping, it would stop pytest itself
        pytest.fail("Ctrl+C escaped web.run as a traceback")
    # A clean exit, so a window an update opened closes too; and stopping means
    # stopping, even with a restart pending.
    assert code == 0
    assert not calls.restarted


def test_a_restarted_server_starts_even_if_something_needs_fixing(web_run, monkeypatch) -> None:
    web, calls = web_run
    monkeypatch.setenv(updates.RESTARTED_ENV_VAR, "1")
    monkeypatch.setattr(web, "_sysimage_ready", lambda *a, **k: calls.checks.append("image"))
    updates._record_outcome(["$ uv tool upgrade jutul-agent"], None)
    args = _web_args()
    assert web.run(args) == 0
    # The env check (which would refuse) is skipped; a stale image is dropped.
    assert calls.checks == ["image"] and args.sysimage is False
    assert updates.RESTARTED_ENV_VAR not in os.environ
    # The page that asked for the update hears how it went.
    assert calls.updater.status()["job"]["state"] == "done"


def _guard(app):
    from jutul_agent.interfaces.server.access import OriginGuard

    return next(m.kwargs for m in app.user_middleware if m.cls is OriginGuard)


def test_a_local_server_checks_hosts_too(web_run) -> None:
    web, calls = web_run
    assert web.run(_web_args(allowed_origin=["http://jutul.test:8742"])) == 0
    assert _guard(calls.app) == {
        "local_only": True,
        "allowed_origins": ["http://jutul.test:8742"],
    }


def test_behind_a_proxy_origins_are_checked_but_not_hosts(web_run) -> None:
    # The proxy's public name arrives as Host, so only the origin can be checked.
    web, calls = web_run
    assert web.run(_web_args(base_path="/restricted")) == 0
    assert _guard(calls.app)["local_only"] is False
    assert not calls.updater.local  # nor is jutul-agent updated from a proxied page


def test_a_server_on_the_network_never_updates(web_run) -> None:
    web, calls = web_run
    assert web.run(_web_args(host="0.0.0.0")) == 0
    assert not calls.updater.local and not calls.restarted


def test_upgrade_choice_for_a_system_image(tmp_path, monkeypatch) -> None:
    from jutul_agent.interfaces.cli import upgrade
    from jutul_agent.workspace import WorkspaceConfig, load_workspace_config, write_workspace_config

    write_workspace_config(
        WorkspaceConfig(simulator="jutuldarcy", sysimage=True), workspace=tmp_path
    )
    monkeypatch.setattr("builtins.input", lambda _p: pytest.fail("must not ask"))
    assert upgrade.decide_sysimage(tmp_path, interactive=True, choice="rebuild") is True
    assert load_workspace_config(tmp_path).sysimage is True
    assert upgrade.decide_sysimage(tmp_path, interactive=True, choice="off") is False
    assert load_workspace_config(tmp_path).sysimage is False
