"""Tests for workspace config and bootstrap."""

from __future__ import annotations

from pathlib import Path

import pytest

from jutul_agent.workspace import (
    SimulatorConfig,
    WorkspaceConfig,
    auto_detect_simulator,
    bootstrap_julia_env,
    env_declares_warm_packages,
    env_precompile_is_current,
    load_workspace_config,
    mark_env_precompiled,
    merge_simulator_config,
    resolve_julia_project,
    sync_julia_env_with_template,
    sync_julia_project_with_dependencies,
    workspace_is_simulator_source,
    workspace_julia_env,
    write_workspace_config,
)


def test_load_and_write_workspace_config_round_trip(tmp_path: Path) -> None:
    config = WorkspaceConfig(
        simulator="jutuldarcy",
        simulators={"jutuldarcy": SimulatorConfig(source_path=tmp_path / "src")},
    )
    write_workspace_config(config, workspace=tmp_path)
    loaded = load_workspace_config(tmp_path)
    assert loaded.simulator == "jutuldarcy"
    assert loaded.simulator_config("jutuldarcy").source_path == (tmp_path / "src").resolve()


def test_workspace_config_persists_model(tmp_path: Path) -> None:
    write_workspace_config(
        WorkspaceConfig(simulator="battmo", model="anthropic:claude-sonnet-4-6"),
        workspace=tmp_path,
    )
    text = (tmp_path / ".jutul-agent" / "config.toml").read_text(encoding="utf-8")
    # Top-level `model` key must precede the [workspace] table to be valid TOML.
    assert text.index("model =") < text.index("[workspace]")

    loaded = load_workspace_config(tmp_path)
    assert loaded.model == "anthropic:claude-sonnet-4-6"
    assert loaded.simulator == "battmo"


def test_workspace_config_without_model_omits_key(tmp_path: Path) -> None:
    write_workspace_config(WorkspaceConfig(simulator="battmo"), workspace=tmp_path)
    text = (tmp_path / ".jutul-agent" / "config.toml").read_text(encoding="utf-8")
    assert "model" not in text
    assert load_workspace_config(tmp_path).model is None


def test_auto_detect_from_deps(tmp_path: Path) -> None:
    (tmp_path / "Project.toml").write_text(
        '[deps]\nJutulDarcy = "uuid"\n',
        encoding="utf-8",
    )
    known = {"JutulDarcy": "jutuldarcy", "BattMo": "battmo"}
    assert auto_detect_simulator(known, tmp_path) == "jutuldarcy"


def test_auto_detect_from_project_name(tmp_path: Path) -> None:
    (tmp_path / "Project.toml").write_text(
        'name = "JutulDarcy"\n[deps]\n',
        encoding="utf-8",
    )
    known = {"JutulDarcy": "jutuldarcy"}
    assert auto_detect_simulator(known, tmp_path) == "jutuldarcy"


def test_workspace_is_simulator_source(tmp_path: Path) -> None:
    (tmp_path / "Project.toml").write_text('name = "Foo"\n', encoding="utf-8")
    assert workspace_is_simulator_source("Foo", tmp_path) is True
    assert workspace_is_simulator_source("Bar", tmp_path) is False


def test_bootstrap_julia_env_uses_root_project(tmp_path: Path) -> None:
    (tmp_path / "Project.toml").write_text("[deps]\n", encoding="utf-8")
    template = tmp_path / "doesnt-matter"
    project = bootstrap_julia_env(template, workspace=tmp_path)
    assert project == tmp_path.resolve()


def test_resolve_julia_project_prefers_julia_env_without_root_project(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    env = ws / ".jutul-agent" / "julia-env"
    env.mkdir(parents=True)
    (env / "Project.toml").write_text('[deps]\nJutul = "uuid"\n', encoding="utf-8")

    assert resolve_julia_project(ws) == env


def test_bootstrap_julia_env_copies_template(tmp_path: Path) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "Project.toml").write_text("[deps]\n", encoding="utf-8")

    ws = tmp_path / "ws"
    ws.mkdir()
    project = bootstrap_julia_env(template, workspace=ws)
    assert project.name == "julia-env"
    assert (project / "Project.toml").exists()
    # The shared JutulAgent package is synced in from julia_runtime/ alongside the
    # template (the env's relative [sources] entry resolves only after this copy).
    assert (project / "JutulAgent" / "Project.toml").exists()


def test_bootstrap_julia_env_copies_template_sources(tmp_path: Path) -> None:
    template = tmp_path / "template"
    (template / "DemoSim" / "src").mkdir(parents=True)
    (template / "DemoSim" / "Project.toml").write_text('name = "DemoSim"\n', encoding="utf-8")
    (template / "Project.toml").write_text(
        "[deps]\n"
        'DemoSim = "399ba059-2ef4-46df-b0ff-fda998e6d1cf"\n'
        "\n[sources]\n"
        'DemoSim = {path = "../DemoSim"}\n',
        encoding="utf-8",
    )

    ws = tmp_path / "ws"
    ws.mkdir()
    project = bootstrap_julia_env(template, workspace=ws)

    assert (project / "DemoSim" / "Project.toml").exists()


def test_merge_simulator_config_updates_one_entry() -> None:
    base = WorkspaceConfig(simulators={"jutuldarcy": SimulatorConfig()})
    updated = merge_simulator_config(base, "jutuldarcy", source_path=Path("/tmp/src"))
    assert updated.simulator_config("jutuldarcy").source_path == Path("/tmp/src")
    assert base.simulator_config("jutuldarcy").source_path is None


@pytest.fixture
def _template_with_extra_deps(tmp_path: Path) -> Path:
    """Sim env template with three deps; workspace will be seeded with one."""
    template = tmp_path / "template"
    template.mkdir()
    (template / "Project.toml").write_text(
        "[deps]\n"
        'Jutul = "c6b0b931-bd15-49f6-a31f-cf7d80eb5e81"\n'
        'CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"\n'
        'Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"\n',
        encoding="utf-8",
    )
    return template


def test_sync_adds_missing_deps_from_template(
    tmp_path: Path, _template_with_extra_deps: Path
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    env = workspace_julia_env(ws)
    env.mkdir(parents=True)
    (env / "Project.toml").write_text(
        '[deps]\nJutul = "c6b0b931-bd15-49f6-a31f-cf7d80eb5e81"\n',
        encoding="utf-8",
    )

    added = sync_julia_env_with_template(_template_with_extra_deps, workspace=ws)
    assert sorted(added) == ["CSV", "Interpolations"]

    text = (env / "Project.toml").read_text(encoding="utf-8")
    assert 'CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"' in text
    assert 'Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"' in text


def test_sync_adds_extra_deps_from_capabilities(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    env = workspace_julia_env(ws)
    env.mkdir(parents=True)
    (env / "Project.toml").write_text(
        '[deps]\nJutul = "c6b0b931-bd15-49f6-a31f-cf7d80eb5e81"\n',
        encoding="utf-8",
    )

    local_pkg = tmp_path / "FooCap"
    local_pkg.mkdir()
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\nuuid = "11111111-1111-1111-1111-111111111111"\n',
        encoding="utf-8",
    )

    added = sync_julia_project_with_dependencies(env, [local_pkg / "Project.toml"])
    assert added == ["FooCap"]

    text = (env / "Project.toml").read_text(encoding="utf-8")
    assert 'FooCap = "11111111-1111-1111-1111-111111111111"' in text
    assert f'FooCap = {{path = "{local_pkg.resolve().as_posix()}"}}' in text


def test_sync_adds_dependencies_to_root_project(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    proj = ws / "Project.toml"
    proj.write_text('[deps]\nJutul = "uuid"\n', encoding="utf-8")

    local_pkg = tmp_path / "FooCap"
    local_pkg.mkdir()
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\nuuid = "11111111-1111-1111-1111-111111111111"\n',
        encoding="utf-8",
    )

    added = sync_julia_project_with_dependencies(
        ws,
        [local_pkg / "Project.toml"],
    )

    assert added == ["FooCap"]
    text = proj.read_text(encoding="utf-8")
    assert 'FooCap = "11111111-1111-1111-1111-111111111111"' in text
    assert f'FooCap = {{path = "{local_pkg.resolve().as_posix()}"}}' in text


def test_sync_is_noop_when_already_in_sync(tmp_path: Path, _template_with_extra_deps: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    env = workspace_julia_env(ws)
    env.mkdir(parents=True)
    (env / "Project.toml").write_text(
        "[deps]\n"
        'Jutul = "c6b0b931-bd15-49f6-a31f-cf7d80eb5e81"\n'
        'CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"\n'
        'Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"\n',
        encoding="utf-8",
    )

    assert sync_julia_env_with_template(_template_with_extra_deps, workspace=ws) == []


def test_sync_skipped_when_workspace_owns_project(
    tmp_path: Path, _template_with_extra_deps: Path
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "Project.toml").write_text("[deps]\n", encoding="utf-8")
    assert sync_julia_env_with_template(_template_with_extra_deps, workspace=ws) == []


@pytest.fixture
def _template_with_path_source(tmp_path: Path) -> Path:
    """Template whose extra dep is a relative `[sources]` path (a warm-up package)."""
    template = tmp_path / "template"
    (template / "FooWarm" / "src").mkdir(parents=True)
    (template / "FooWarm" / "Project.toml").write_text('name = "FooWarm"\n', encoding="utf-8")
    (template / "Project.toml").write_text(
        "[deps]\n"
        'Jutul = "c6b0b931-bd15-49f6-a31f-cf7d80eb5e81"\n'
        'FooWarm = "11111111-1111-1111-1111-111111111111"\n'
        "\n[sources]\n"
        'FooWarm = {path = "FooWarm"}\n',
        encoding="utf-8",
    )
    return template


def test_sync_brings_path_sourced_dep_with_its_source_entry_and_dir(
    tmp_path: Path, _template_with_path_source: Path
) -> None:
    # Regression: a path-sourced dep added without its `[sources]` entry and
    # package dir makes `Pkg.resolve` fail with "expected package ... registered".
    ws = tmp_path / "ws"
    ws.mkdir()
    env = workspace_julia_env(ws)
    env.mkdir(parents=True)
    (env / "Project.toml").write_text(
        '[deps]\nJutul = "c6b0b931-bd15-49f6-a31f-cf7d80eb5e81"\n',
        encoding="utf-8",
    )

    assert sync_julia_env_with_template(_template_with_path_source, workspace=ws) == ["FooWarm"]

    text = (env / "Project.toml").read_text(encoding="utf-8")
    assert 'FooWarm = "11111111-1111-1111-1111-111111111111"' in text  # the dep
    assert "[sources]" in text
    assert 'FooWarm = {path = "FooWarm"}' in text  # the source entry
    assert (env / "FooWarm" / "Project.toml").exists()  # the package dir copied in


def test_env_declares_warm_packages_detects_the_jutulagent_prefix(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.mkdir()
    proj = env / "Project.toml"

    proj.write_text('[deps]\nJutul = "c6b0b931-bd15-49f6-a31f-cf7d80eb5e81"\n', encoding="utf-8")
    assert not env_declares_warm_packages(env)

    proj.write_text(
        '[deps]\nJutulAgentJutulDarcy = "69df87d8-8b4b-4157-81d2-8b93ff139141"\n',
        encoding="utf-8",
    )
    assert env_declares_warm_packages(env)


def test_env_precompile_marker_tracks_the_manifest(tmp_path: Path) -> None:
    import os

    from jutul_agent.workspace import PRECOMPILE_MARKER

    env = tmp_path / "env"
    env.mkdir()
    manifest = env / "Manifest.toml"

    assert not env_precompile_is_current(env)  # no marker, no manifest

    manifest.write_text("", encoding="utf-8")
    mark_env_precompiled(env)
    os.utime(manifest, (100, 100))
    os.utime(env / PRECOMPILE_MARKER, (200, 200))
    assert env_precompile_is_current(env)  # baked after the last manifest write

    os.utime(env / PRECOMPILE_MARKER, (50, 50))
    assert not env_precompile_is_current(env)  # manifest changed since the bake
