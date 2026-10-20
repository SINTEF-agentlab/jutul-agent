"""CLI smoke tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from jutul_agent.interfaces.cli import main


def test_help_exits_zero_subprocess() -> None:
    """Sanity check that the installed entry-point script wires up correctly."""

    result = subprocess.run(
        [sys.executable, "-m", "jutul_agent", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "jutul-agent" in result.stdout.lower()
    assert "--sim" in result.stdout
    assert "init" in result.stdout
    assert "transcript" in result.stdout


def test_version_exits_zero(capsys) -> None:
    from jutul_agent import __version__

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    # Version is derived from git tags by hatch-vcs, so assert the wiring (the
    # resolved __version__ is printed) rather than a fixed string.
    assert f"jutul-agent {__version__}" in out


def test_default_invocation_without_sim_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An interface invoked without ``--sim`` must fail fast when no simulator is
    configured.

    Pass an isolated ``--workspace``: the repo checkout has
    ``.jutul-agent/config.toml`` with ``simulator = "battmo"``, and ``main``
    resets workspace overrides via ``_apply_workspace_flags`` (``None`` → cwd).
    Without isolation the CLI would launch Julia and hang or crash
    under pytest capture (stderr has no fileno).
    """
    ws = tmp_path / "workspace"
    ws.mkdir()
    state = tmp_path / "state"
    state.mkdir()

    code = main(["tui", "--workspace", str(ws), "--state-home", str(state)])
    captured = capsys.readouterr()
    assert code == 2
    assert "--sim is required" in captured.err


@pytest.fixture
def web_runs(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record each launch of the web UI instead of starting a server."""
    from jutul_agent.interfaces.cli import web

    launched: list = []
    monkeypatch.setattr(web, "run", lambda args: launched.append(args) or 0)
    return launched


@pytest.mark.parametrize(
    ("argv", "port", "sim"),
    [
        ([], 8742, None),
        (["web"], 8742, None),
        # The web UI's options work without the command too.
        (["--port", "8800", "--sim", "battmo"], 8800, "battmo"),
        (["web", "--port", "8800", "--sim", "battmo"], 8800, "battmo"),
    ],
)
def test_no_command_starts_the_web_ui(web_runs: list, argv: list[str], port, sim) -> None:
    assert main(argv) == 0
    assert [(args.port, args.sim) for args in web_runs] == [(port, sim)]


def test_help_says_no_command_starts_the_web_ui(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "starts the web UI" in out and "`jutul-agent web` is the same" in out
    assert "jutul-agent tui" in out and 'run "<prompt>"' in out


@pytest.mark.parametrize(
    ("argv", "usage"),
    [(["--bogus"], "usage: jutul-agent [-h]"), (["web", "--bogus"], "usage: jutul-agent web")],
)
def test_an_unknown_option_fails_before_starting_a_server(
    web_runs: list, capsys: pytest.CaptureFixture[str], argv: list[str], usage: str
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert usage in err and "unrecognized arguments: --bogus" in err
    assert web_runs == []


def test_unknown_command_errors(web_runs: list, capsys: pytest.CaptureFixture[str]) -> None:
    # Neither an unknown word nor a bare prompt starts the web UI.
    assert main(["frobnicate"]) == 2
    assert "unknown command" in capsys.readouterr().err
    assert main(["plot the voltage curve"]) == 2
    assert web_runs == []


def test_a_mistyped_command_names_the_likely_one(
    web_runs: list, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["tiu"]) == 2
    assert "Did you mean `jutul-agent tui`?" in capsys.readouterr().err
    assert web_runs == []


@pytest.mark.parametrize(
    ("argv", "module", "entry"),
    [
        (["tui"], "run", "dispatch"),
        (["run", "compute mean([1,2,3])"], "run", "dispatch"),
        (["init", "--sim", "battmo"], "init", "run"),
        (["doctor"], "doctor", "run"),
        (["sessions"], "sessions", "run"),
        (["upgrade", "--check"], "upgrade", "run"),
    ],
)
def test_commands_dispatch_as_before(
    web_runs: list,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    module: str,
    entry: str,
) -> None:
    import importlib

    target = importlib.import_module(f"jutul_agent.interfaces.cli.{module}")
    called: list = []
    monkeypatch.setattr(target, entry, lambda args: called.append(args) or 0)

    workspace = ["--workspace", str(tmp_path), "--state-home", str(tmp_path / "state")]
    assert main([*argv, *workspace]) == 0
    assert len(called) == 1
    assert web_runs == []


def test_tui_rejects_a_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    # `tui` is interactive; a prompt belongs to `run`. Guarded before any kernel start.
    assert main(["tui", "run a sim"]) == 2
    assert "takes no prompt" in capsys.readouterr().err


def test_run_requires_a_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run"]) == 2
    assert "needs a prompt" in capsys.readouterr().err
