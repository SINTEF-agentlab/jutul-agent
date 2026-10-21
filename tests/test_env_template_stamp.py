"""Tests for env-template fingerprinting and staleness detection."""

from __future__ import annotations

from pathlib import Path

from jutul_agent.simulators import env_setup
from jutul_agent.simulators.base import SimulatorAdapter
from jutul_agent.workspace import (
    TEMPLATE_STAMP_FILENAME,
    bootstrap_julia_env,
    ensure_env_template_stamp,
    env_template_drifted,
    read_env_template_stamp,
    template_fingerprint,
    workspace_julia_env,
    write_env_template_stamp,
)

# A template Project.toml that declares a warm package, so the env reads as a
# managed jutul-agent env (env_declares_warm_packages → True).
_TEMPLATE_PROJECT = '[deps]\nFoo = "uuid"\nJutulAgent = "ja-uuid"\n'


def _adapter(module_dir: Path) -> SimulatorAdapter:
    return SimulatorAdapter(
        name="test",
        display_name="Test",
        module_dir=module_dir,
        package_imports=("Foo",),
        primary_package="Foo",
        domain_hints="",
    )


def _make_template(tmp_path: Path, body: str = _TEMPLATE_PROJECT) -> Path:
    module_dir = tmp_path / "sim"
    template = module_dir / "julia_env"
    template.mkdir(parents=True)
    (template / "Project.toml").write_text(body, encoding="utf-8")
    return module_dir


def test_fingerprint_changes_with_project_toml(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    template = module_dir / "julia_env"
    first = template_fingerprint(template)
    (template / "Project.toml").write_text(_TEMPLATE_PROJECT + 'Bar = "b"\n', encoding="utf-8")
    assert template_fingerprint(template) != first


def test_fingerprint_empty_when_missing(tmp_path: Path) -> None:
    assert template_fingerprint(tmp_path / "nope") == ""


def test_stamp_roundtrip(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    template = module_dir / "julia_env"
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    write_env_template_stamp(env_dir, template)

    stamp = read_env_template_stamp(env_dir)
    assert stamp is not None
    assert stamp["fingerprint"] == template_fingerprint(template)
    assert "version" in stamp


def test_drift_false_without_stamp(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    # No stamp written → unknown, never nag.
    assert env_template_drifted(env_dir, module_dir / "julia_env") is False


def test_drift_detected_after_template_change(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    template = module_dir / "julia_env"
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    write_env_template_stamp(env_dir, template)
    assert env_template_drifted(env_dir, template) is False

    (template / "Project.toml").write_text(_TEMPLATE_PROJECT + 'Baz = "z"\n', encoding="utf-8")
    assert env_template_drifted(env_dir, template) is True


def test_ensure_stamp_only_writes_when_missing(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    template = module_dir / "julia_env"
    env_dir = tmp_path / "env"
    env_dir.mkdir()

    ensure_env_template_stamp(env_dir, template)
    first = read_env_template_stamp(env_dir)
    assert first is not None

    # Change the template, then ensure again: an existing stamp is left intact
    # (so a healthy env keeps its baseline rather than silently re-baselining).
    (template / "Project.toml").write_text(_TEMPLATE_PROJECT + 'Q = "q"\n', encoding="utf-8")
    ensure_env_template_stamp(env_dir, template)
    assert read_env_template_stamp(env_dir) == first


def test_bootstrap_writes_stamp(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    bootstrap_julia_env(module_dir / "julia_env", workspace=workspace)
    env_dir = workspace_julia_env(workspace)
    assert (env_dir / TEMPLATE_STAMP_FILENAME).exists()
    assert not env_template_drifted(env_dir, module_dir / "julia_env")


def test_reconcile_warns_on_drift(tmp_path: Path, capsys) -> None:
    module_dir = _make_template(tmp_path)
    template = module_dir / "julia_env"
    workspace = tmp_path / "ws"
    workspace.mkdir()
    bootstrap_julia_env(template, workspace=workspace)
    env_dir = workspace_julia_env(workspace)

    # Drift the template after the env was stamped.
    (template / "Project.toml").write_text(_TEMPLATE_PROJECT + 'New = "n"\n', encoding="utf-8")

    env_setup._reconcile_env_template(_adapter(module_dir), workspace, env_dir, "test")
    err = capsys.readouterr().err
    assert "older" in err and "init --force" in err


def test_reconcile_silent_when_current(tmp_path: Path, capsys) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    bootstrap_julia_env(module_dir / "julia_env", workspace=workspace)
    env_dir = workspace_julia_env(workspace)

    env_setup._reconcile_env_template(_adapter(module_dir), workspace, env_dir, "test")
    assert capsys.readouterr().err == ""


def _managed_env(tmp_path: Path) -> tuple[Path, Path]:
    """A workspace + its managed julia-env declaring a warm package."""

    ws = tmp_path / "ws"
    env_dir = ws / ".jutul-agent" / "julia-env"
    env_dir.mkdir(parents=True)
    (env_dir / "Project.toml").write_text('[deps]\nJutulAgent = "x"\n', encoding="utf-8")
    return ws, env_dir


def test_doctor_flags_drifted_env(tmp_path: Path) -> None:
    import json

    from jutul_agent.interfaces.cli import doctor
    from jutul_agent.simulators import registry

    ws, env_dir = _managed_env(tmp_path)
    (env_dir / TEMPLATE_STAMP_FILENAME).write_text(
        json.dumps({"fingerprint": "bogus", "version": "0"}), encoding="utf-8"
    )

    report = doctor._Report()
    doctor._check_env_template_current(report, ws, env_dir, registry.names()[0])
    assert report.worst == doctor.WARN


def test_doctor_passes_current_env(tmp_path: Path) -> None:
    from jutul_agent.interfaces.cli import doctor
    from jutul_agent.simulators import registry

    ws, env_dir = _managed_env(tmp_path)
    sim = registry.names()[0]
    write_env_template_stamp(env_dir, registry.get(sim).julia_env_template_path)

    report = doctor._Report()
    doctor._check_env_template_current(report, ws, env_dir, sim)
    assert report.worst == doctor.PASS


def test_doctor_fails_an_outdated_simulator(tmp_path: Path, capsys) -> None:
    """The report users paste names the versions and the command that fixes them."""
    from jutul_agent.interfaces.cli import doctor
    from jutul_agent.simulators import registry

    ws, env_dir = _managed_env(tmp_path)
    write_env_template_stamp(env_dir, registry.get("jutuldarcy").julia_env_template_path)
    (env_dir / "Manifest.toml").write_text(
        '[[deps.JutulDarcy]]\nversion = "0.3.12"\n', encoding="utf-8"
    )

    report = doctor._Report()
    doctor._check_env_template_current(report, ws, env_dir, "jutuldarcy")
    assert report.worst == doctor.FAIL
    out = capsys.readouterr().out
    assert "JutulDarcy 0.3.12" in out and "jutul-agent upgrade" in out


def test_doctor_warns_about_a_newer_simulator(tmp_path: Path, capsys) -> None:
    """A newer simulator runs; the report says so without calling it a failure."""
    from jutul_agent.interfaces.cli import doctor
    from jutul_agent.simulators import registry

    ws, env_dir = _managed_env(tmp_path)
    write_env_template_stamp(env_dir, registry.get("jutuldarcy").julia_env_template_path)
    (env_dir / "Manifest.toml").write_text(
        '[[deps.JutulDarcy]]\nversion = "9.0.0"\n', encoding="utf-8"
    )

    report = doctor._Report()
    doctor._check_env_template_current(report, ws, env_dir, "jutuldarcy")
    assert report.worst == doctor.WARN
    out = capsys.readouterr().out
    assert "JutulDarcy 9.0.0, newer than this jutul-agent was made for" in out
