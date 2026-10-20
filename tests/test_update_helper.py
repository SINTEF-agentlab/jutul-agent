"""The helper that finishes an update in a separate window, as Windows needs.

It runs for real here, on every system CI covers: small Python scripts stand in for
uv, the environment update and the restarted server.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from jutul_agent.interfaces.server import update_helper, updates
from jutul_agent.interfaces.server.updates import RESTARTED_ENV_VAR, Updater

# Stands in for each command: notes how it ran, says so, and exits with the code given.
FAKE = textwrap.dedent(
    """
    import json, os, sys
    record, name, code = sys.argv[1], sys.argv[2], int(sys.argv[3])
    entry = {
        "name": name,
        "args": sys.argv[4:],
        "cwd": os.getcwd(),
        "restarted": os.environ.get("JUTUL_AGENT_RESTARTED_AFTER_UPDATE"),
    }
    with open(record, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\\n")
    print(f"{name} ran")
    sys.exit(code)
    """
)

# Starts the helper as the server does, says when it is about to exit, and exits.
SERVER = textwrap.dedent(
    """
    import json, os, sys, time
    from jutul_agent.interfaces.server.updates import start_helper
    plan, record = json.loads(sys.argv[1]), sys.argv[2]
    plan["wait_for"] = [os.getpid()]
    start_helper(plan)
    time.sleep(3)  # well past the helper's start, so it has to wait
    with open(record, "a", encoding="utf-8") as f:
        f.write(json.dumps({"name": "server exits"}) + "\\n")
    """
)


@pytest.fixture
def plan(tmp_path):
    """A plan whose commands are the fake, recording to ``tmp_path / "record"``."""
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE, encoding="utf-8")
    record = tmp_path / "record"

    def command(name: str, code: int = 0) -> list[str]:
        return [sys.executable, str(fake), str(record), name, str(code)]

    workspace, started_in = tmp_path / "workspace", tmp_path / "started-in"
    workspace.mkdir()
    started_in.mkdir()
    folder = updates._update_dir()
    folder.mkdir(parents=True)
    return {
        "wait_for": [],
        "free": [str(fake)],
        "timeout": 60,
        "cwd": str(workspace),
        "steps": [
            {
                "argv": [*command("uv"), "tool", "upgrade", "jutul-agent"],
                "shown": "uv tool upgrade jutul-agent",
                "error": "jutul-agent could not be upgraded.",
            },
            {
                "argv": [*command("env"), "upgrade", "--env-only"],
                "shown": "jutul-agent upgrade --env-only",
                "error": "The Julia environment could not be updated.",
            },
        ],
        "relaunch": {
            "argv": [*command("server"), "--port", "8742"],
            "cwd": str(started_in),
            "env": {RESTARTED_ENV_VAR: "1"},
        },
        "log": str(folder / "update.log"),
        "outcome": str(folder / "outcome.json"),
    }


def _entries(record: Path, *, until: str, timeout: float = 60) -> list[dict]:
    """The record's entries, once one named ``until`` is among them."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if record.exists():
            entries = [json.loads(line) for line in record.read_text("utf-8").splitlines()]
            if any(entry["name"] == until for entry in entries):
                return entries
        time.sleep(0.1)
    raise AssertionError(f"no {until!r} within {timeout}s: {record.read_text('utf-8')}")


def _log(plan) -> list[str]:
    return Path(plan["log"]).read_text("utf-8").splitlines()


def test_the_helper_updates_after_the_server_exits_then_restarts_it(tmp_path, plan) -> None:
    record = tmp_path / "record"
    server = subprocess.run(
        [sys.executable, "-c", SERVER, json.dumps(plan), str(record)], timeout=60, check=False
    )
    assert server.returncode == 0

    entries = _entries(record, until="server")
    # Nothing ran until the server had gone, then each step in turn.
    assert [entry["name"] for entry in entries] == ["server exits", "uv", "env", "server"]
    _, upgrade, env, restarted = entries
    assert upgrade["args"] == ["tool", "upgrade", "jutul-agent"]
    assert os.path.samefile(env["cwd"], plan["cwd"]) and env["restarted"] is None
    # Started again as it was, where it was, knowing it follows an update.
    assert restarted["args"] == ["--port", "8742"]
    assert os.path.samefile(restarted["cwd"], plan["relaunch"]["cwd"])
    assert restarted["restarted"] == "1"

    assert json.loads(Path(plan["outcome"]).read_text("utf-8")) == {"state": "done", "error": None}
    log = _log(plan)
    shown = [
        "$ uv tool upgrade jutul-agent",
        "uv ran",
        "$ jutul-agent upgrade --env-only",
        "env ran",
        "Starting jutul-agent again...",
    ]
    assert [line for line in log if line in shown] == shown


def test_a_failed_upgrade_still_restarts_and_is_reported(tmp_path, plan) -> None:
    plan["steps"][0]["argv"][4] = "1"  # uv fails
    server = update_helper.finish_update(plan)
    assert server is not None and server.wait(timeout=60) == 0

    entries = _entries(tmp_path / "record", until="server")
    # The environment is left alone, and the old version starts again.
    assert [entry["name"] for entry in entries] == ["uv", "server"]
    assert "jutul-agent could not be upgraded." in _log(plan)

    # The restarted server reads what the helper left.
    restarted = Updater(tmp_path, local=True)
    restarted.report_last_update()
    job = restarted.status()["job"]
    assert job["state"] == "failed" and job["error"] == "jutul-agent could not be upgraded."
    assert "uv ran" in job["lines"]


def test_a_relaunch_that_cannot_start_is_said(plan) -> None:
    plan["relaunch"]["argv"] = [str(Path(plan["cwd"]) / "missing-program")]
    assert update_helper.finish_update(plan) is None
    assert any(line.startswith("Could not start jutul-agent again") for line in _log(plan))


def test_waiting_for_processes() -> None:
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert update_helper.wait_for_exit([gone.pid], time.monotonic() + 5)
        assert not update_helper.wait_for_exit([alive.pid], time.monotonic() + 0.5)
    finally:
        alive.kill()
        alive.wait()


def test_a_file_nothing_runs_is_free(tmp_path) -> None:
    unused = tmp_path / "jutul-agent.exe"
    unused.write_bytes(b"MZ")
    assert update_helper.wait_until_free(str(unused), time.monotonic() + 5)
    assert update_helper.wait_until_free(str(tmp_path / "missing.exe"), time.monotonic() + 5)
    assert unused.read_bytes() == b"MZ"  # untouched


@pytest.mark.skipif(os.name != "nt", reason="only Windows locks a running program")
def test_a_running_program_is_not_free() -> None:
    # The interpreter running this test is a running program; uv couldn't replace it.
    assert not update_helper.wait_until_free(sys.executable, time.monotonic() + 0.5)
