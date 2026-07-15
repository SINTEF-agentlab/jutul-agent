"""Tests for the workspace Julia environment bootstrap."""

from __future__ import annotations

from pathlib import Path

import pytest

from jutul_agent.simulators import env_setup
from jutul_agent.simulators.base import SimulatorAdapter
from jutul_agent.workspace import workspace_julia_env


def _adapter(module_dir: Path) -> SimulatorAdapter:
    return SimulatorAdapter(
        name="test",
        display_name="Test",
        module_dir=module_dir,
        package_imports=("Foo",),
        primary_package="Foo",
        domain_hints="",
    )


def _make_template(tmp_path: Path) -> Path:
    """Lay out a fake simulator module dir with a julia_env/ template."""

    module_dir = tmp_path / "sim"
    template = module_dir / "julia_env"
    template.mkdir(parents=True)
    (template / "Project.toml").write_text('[deps]\nFoo = "uuid"\n', encoding="utf-8")
    (template / "Manifest.toml").write_text("# manifest\n", encoding="utf-8")
    return module_dir


def test_bootstrap_copies_template_into_workspace(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    project = env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace)

    assert project == workspace_julia_env(workspace)
    assert (project / "Project.toml").read_text(encoding="utf-8").startswith("[deps]")
    # The template's Manifest.toml is intentionally NOT carried over; the workspace
    # resolves its own at instantiate (a stale template manifest would omit newly
    # added deps like the per-sim warm package).
    assert not (project / "Manifest.toml").exists()


def test_bootstrap_uses_root_project_when_present(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "Project.toml").write_text("[deps]\n", encoding="utf-8")

    project = env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace)

    assert project == workspace
    assert not workspace_julia_env(workspace).exists()


def test_bootstrap_force_recopies_template(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    template = module_dir / "julia_env"
    workspace = tmp_path / "ws"
    workspace.mkdir()

    env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace)
    target = workspace_julia_env(workspace)
    target.joinpath("stale-marker").write_text("old", encoding="utf-8")
    (template / "fresh-marker").write_text("new", encoding="utf-8")

    env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace, force=True)

    assert target.joinpath("fresh-marker").exists()
    assert not target.joinpath("stale-marker").exists()


def test_bootstrap_is_idempotent(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace)
    target = workspace_julia_env(workspace)
    target.joinpath("touched").write_text("x", encoding="utf-8")

    env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace)
    assert target.joinpath("touched").exists()  # not overwritten


def test_manifest_has_package_format_2(tmp_path: Path) -> None:
    proj = tmp_path / "env"
    proj.mkdir()
    (proj / "Manifest.toml").write_text(
        'julia_version = "1.12.0"\n'
        'manifest_format = "2.0"\n\n'
        "[deps]\n"
        "[[deps.BattMo]]\n"
        'uuid = "6f0c0536-3c2c-4762-a987-c605a8a6f898"\n'
        'version = "1.0.0"\n',
        encoding="utf-8",
    )
    assert env_setup.manifest_has_package(proj, "BattMo") is True
    assert env_setup.manifest_has_package(proj, "JutulDarcy") is False


def test_manifest_has_package_format_1(tmp_path: Path) -> None:
    proj = tmp_path / "env"
    proj.mkdir()
    (proj / "Manifest.toml").write_text(
        '[[Jutul]]\nuuid = "x"\nversion = "1.0"\n',
        encoding="utf-8",
    )
    assert env_setup.manifest_has_package(proj, "Jutul") is True
    assert env_setup.manifest_has_package(proj, "BattMo") is False


def test_manifest_has_package_missing_manifest(tmp_path: Path) -> None:
    proj = tmp_path / "env"
    proj.mkdir()
    assert env_setup.manifest_has_package(proj, "BattMo") is False


def test_is_workspace_env_ready_reflects_project_toml(tmp_path: Path) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()

    assert env_setup.is_workspace_env_ready(workspace) is False
    env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace)
    assert env_setup.is_workspace_env_ready(workspace) is True


def test_bootstrap_with_source_path_runs_pkg_develop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = tmp_path / "FooSource"
    source.mkdir()

    captured: dict[str, list[str]] = {}

    class _Result:
        returncode = 0

    def _fake_run(argv, check=False):
        captured["argv"] = argv
        return _Result()

    monkeypatch.setattr(env_setup.shutil, "which", lambda _: "/usr/bin/julia")
    monkeypatch.setattr(env_setup.subprocess, "run", _fake_run)

    env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace, source_path=source)

    argv = captured["argv"]
    assert "julia" in argv  # argv[0] may be `xvfb-run` on headless Linux
    assert any(arg.startswith("--project=") for arg in argv)
    code = argv[-1]
    assert "using Pkg" in code
    assert f'Pkg.develop(path=raw"{source}")' in code


def test_precompile_runs_instantiate_and_precompile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    captured: list[str] = []

    class _Result:
        returncode = 0

    def _fake_run(argv, check=False):
        captured.append(argv[-1])
        return _Result()

    monkeypatch.setattr(env_setup.shutil, "which", lambda _: "/usr/bin/julia")
    monkeypatch.setattr(env_setup.subprocess, "run", _fake_run)

    env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace, precompile=True)

    # Resolve/download then precompile; the plotting bake is the env's
    # JutulAgent @compile_workload (run by Pkg.precompile), not a separate
    # GLMakie eval here.
    assert any("Pkg.instantiate()" in cmd for cmd in captured)
    assert any("Pkg.precompile()" in cmd for cmd in captured)
    # The post-precompile boot probe still runs.
    assert any("print(1 + 1)" in cmd for cmd in captured)


def test_bootstrap_raises_when_julia_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = tmp_path / "FooSource"
    source.mkdir()

    monkeypatch.setattr(env_setup.shutil, "which", lambda _: None)
    with pytest.raises(env_setup.EnvSetupError, match="julia"):
        env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace, source_path=source)


def test_bootstrap_raises_on_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    source = tmp_path / "FooSource"
    source.mkdir()

    class _Result:
        returncode = 17

    monkeypatch.setattr(env_setup.shutil, "which", lambda _: "/usr/bin/julia")
    monkeypatch.setattr(env_setup.subprocess, "run", lambda *a, **kw: _Result())
    with pytest.raises(env_setup.EnvSetupError, match="code 17"):
        env_setup.bootstrap_workspace(_adapter(module_dir), workspace=workspace, source_path=source)


def test_bootstrap_skips_dev_when_workspace_is_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "Project.toml").write_text('name = "Foo"\n[deps]\n', encoding="utf-8")

    called = False

    def _should_not_be_called(*a, **kw):
        nonlocal called
        called = True

    monkeypatch.setattr(env_setup.subprocess, "run", _should_not_be_called)
    monkeypatch.setattr(env_setup.shutil, "which", lambda _: "/usr/bin/julia")

    env_setup.bootstrap_workspace(
        _adapter(module_dir), workspace=workspace, source_path=tmp_path / "elsewhere"
    )
    assert called is False


def test_prepare_workspace_env_bootstraps_missing_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    installs: list[Path] = []
    monkeypatch.setattr(
        env_setup, "resolve_and_instantiate", lambda project, **kw: installs.append(project)
    )

    project = workspace_julia_env(workspace)
    env_setup.prepare_workspace_env(
        _adapter(module_dir), workspace=workspace, julia_project=project
    )

    assert (project / "Project.toml").exists()
    # The simulator was declared but never resolved, so it gets installed.
    assert installs == [project]


def test_prepare_workspace_env_rebuilds_foreign_managed_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A managed env built for another simulator is replaced from the template."""
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    env = workspace_julia_env(workspace)
    env.mkdir(parents=True)
    (env / "Project.toml").write_text('[deps]\nJutulDarcy = "uuid"\n', encoding="utf-8")
    monkeypatch.setattr(env_setup, "_run_pkg", lambda *a, **k: None)

    env_setup.prepare_workspace_env(_adapter(module_dir), workspace=workspace, julia_project=env)

    text = (env / "Project.toml").read_text(encoding="utf-8")
    assert "Foo" in text
    assert "JutulDarcy" not in text


def test_prepare_workspace_env_leaves_user_root_project_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    root = workspace / "Project.toml"
    root.write_text('[deps]\nFoo = "uuid"\n', encoding="utf-8")
    (workspace / "Manifest.toml").write_text("[deps.Foo]\n", encoding="utf-8")

    env_setup.prepare_workspace_env(
        _adapter(module_dir), workspace=workspace, julia_project=workspace
    )

    assert root.read_text(encoding="utf-8") == '[deps]\nFoo = "uuid"\n'
    assert not workspace_julia_env(workspace).exists()
