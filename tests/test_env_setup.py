"""Tests for the workspace Julia environment bootstrap."""

from __future__ import annotations

from pathlib import Path

import pytest

from jutul_agent.simulators import env_setup
from jutul_agent.simulators.base import SimulatorAdapter
from jutul_agent.workspace import WARM_SOURCE_MARKER, workspace_julia_env


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
        stdout = ""
        stderr = ""

    def _fake_run(argv, **_kwargs):
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
    # The rebuild bootstraps with `precompile=True`, whose boot probe launches a
    # real Julia. Stubbed, or this test waits on a subprocess it never asked for
    # (it hung the Windows lane, where reading that pipe does not come back).
    monkeypatch.setattr(env_setup, "verify_julia_runs", lambda _project: None)

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
    # A [sources] table of its own: the warm-source refresh acts on any project
    # that has one, so a user-owned project must be kept out of that path.
    original = '[deps]\nFoo = "uuid"\n\n[sources]\nFoo = {path = "Foo"}\n'
    root.write_text(original, encoding="utf-8")
    (workspace / "Manifest.toml").write_text("[deps.Foo]\n", encoding="utf-8")

    installs: list[Path] = []
    monkeypatch.setattr(
        env_setup, "resolve_and_instantiate", lambda project, **kw: installs.append(project)
    )

    env_setup.prepare_workspace_env(
        _adapter(module_dir), workspace=workspace, julia_project=workspace
    )

    assert root.read_text(encoding="utf-8") == original
    assert not workspace_julia_env(workspace).exists()
    assert installs == []
    assert not (workspace / WARM_SOURCE_MARKER).exists()


def test_prepare_workspace_env_syncs_capability_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A managed env gets a capability's local package merged in and resolved."""
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    env = workspace_julia_env(workspace)
    env.mkdir(parents=True)
    (env / "Project.toml").write_text('[deps]\nFoo = "uuid"\n', encoding="utf-8")

    local_pkg = tmp_path / "FooCap"
    local_pkg.mkdir()
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\nuuid = "11111111-1111-1111-1111-111111111111"\n',
        encoding="utf-8",
    )

    installs: list[Path] = []
    monkeypatch.setattr(
        env_setup, "resolve_and_instantiate", lambda project, **kw: installs.append(project)
    )

    env_setup.prepare_workspace_env(
        _adapter(module_dir),
        workspace=workspace,
        julia_project=env,
        dependencies=[local_pkg / "Project.toml"],
    )

    text = (env / "Project.toml").read_text(encoding="utf-8")
    assert 'FooCap = "11111111-1111-1111-1111-111111111111"' in text
    assert installs  # resolve_and_instantiate ran after the capability dep was added


def test_prepare_workspace_env_keeps_a_capability_package_on_a_later_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dep is already declared and the warm sources are stale (post-upgrade).

    The refresh then walks every [sources] entry, including the capability's
    absolute path, so it must leave that package where it is.
    """
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    env = workspace_julia_env(workspace)
    env.mkdir(parents=True)

    local_pkg = tmp_path / "FooCap"
    (local_pkg / "src").mkdir(parents=True)
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\nuuid = "11111111-1111-1111-1111-111111111111"\n',
        encoding="utf-8",
    )
    (local_pkg / "src" / "FooCap.jl").write_text("module FooCap end\n", encoding="utf-8")

    (env / "Project.toml").write_text(
        "[deps]\n"
        'Foo = "uuid"\n'
        'FooCap = "11111111-1111-1111-1111-111111111111"\n'
        "\n[sources]\n"
        f'FooCap = {{path = "{local_pkg.as_posix()}"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(env_setup, "resolve_and_instantiate", lambda project, **kw: None)

    env_setup.prepare_workspace_env(
        _adapter(module_dir),
        workspace=workspace,
        julia_project=env,
        dependencies=[local_pkg / "Project.toml"],
    )

    assert (local_pkg / "src" / "FooCap.jl").exists()


def test_prepare_workspace_env_syncs_capability_dependencies_into_user_root_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capability deps are merged even into a user-owned root Project.toml."""
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    root = workspace / "Project.toml"
    root.write_text('[deps]\nFoo = "uuid"\n', encoding="utf-8")
    (workspace / "Manifest.toml").write_text("[deps.Foo]\n", encoding="utf-8")

    local_pkg = tmp_path / "FooCap"
    local_pkg.mkdir()
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\nuuid = "11111111-1111-1111-1111-111111111111"\n',
        encoding="utf-8",
    )

    installs: list[Path] = []
    monkeypatch.setattr(
        env_setup, "resolve_and_instantiate", lambda project, **kw: installs.append(project)
    )

    env_setup.prepare_workspace_env(
        _adapter(module_dir),
        workspace=workspace,
        julia_project=workspace,
        dependencies=[local_pkg / "Project.toml"],
    )

    text = root.read_text(encoding="utf-8")
    assert 'FooCap = "11111111-1111-1111-1111-111111111111"' in text
    assert not workspace_julia_env(workspace).exists()
    assert installs == [workspace]
    # The capability's own package is still there; syncing must not move or copy it.
    assert (local_pkg / "Project.toml").exists()


def test_prepare_workspace_env_warns_when_dependency_sync_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    root = workspace / "Project.toml"
    root.write_text('[deps]\nFoo = "uuid"\n', encoding="utf-8")
    (workspace / "Manifest.toml").write_text("[deps.Foo]\n", encoding="utf-8")

    def _boom(*a, **kw):
        raise RuntimeError("bad Project.toml")

    monkeypatch.setattr(env_setup, "sync_julia_project_with_dependencies", _boom)

    # Must not raise: a broken dependency sync is best-effort, not fatal.
    env_setup.prepare_workspace_env(
        _adapter(module_dir),
        workspace=workspace,
        julia_project=workspace,
        dependencies=[tmp_path / "FooCap" / "Project.toml"],
    )

    assert "warning: capability dependency sync failed" in capsys.readouterr().err


def test_prepare_workspace_env_warns_when_dependency_resolve_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module_dir = _make_template(tmp_path)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    root = workspace / "Project.toml"
    root.write_text('[deps]\nFoo = "uuid"\n', encoding="utf-8")
    (workspace / "Manifest.toml").write_text("[deps.Foo]\n", encoding="utf-8")

    local_pkg = tmp_path / "FooCap"
    local_pkg.mkdir()
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\nuuid = "11111111-1111-1111-1111-111111111111"\n',
        encoding="utf-8",
    )

    def _raise(*a, **kw):
        raise env_setup.EnvSetupError("boom")

    monkeypatch.setattr(env_setup, "resolve_and_instantiate", _raise)

    # Must not raise: a failed resolve after adding capability deps is best-effort.
    env_setup.prepare_workspace_env(
        _adapter(module_dir),
        workspace=workspace,
        julia_project=workspace,
        dependencies=[local_pkg / "Project.toml"],
    )

    err = capsys.readouterr().err
    assert "could not resolve and instantiate the capability dependencies" in err
    # The dep was still written even though the follow-up resolve failed.
    assert 'FooCap = "11111111-1111-1111-1111-111111111111"' in root.read_text(encoding="utf-8")
