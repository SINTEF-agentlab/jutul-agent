"""Tests for workspace config and bootstrap."""

from __future__ import annotations

import tomllib
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
    """A [sources] path pointing outside the template is copied in alongside the env.

    The package lives next to the template rather than inside it, so the plain
    template copy cannot account for it landing next to the env.
    """
    template = tmp_path / "template"
    template.mkdir()
    (tmp_path / "DemoSim" / "src").mkdir(parents=True)
    (tmp_path / "DemoSim" / "Project.toml").write_text('name = "DemoSim"\n', encoding="utf-8")
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

    # `../DemoSim` resolves relative to the env, so the copy sits beside it.
    assert (project.parent / "DemoSim" / "Project.toml").exists()


def test_bootstrap_julia_env_leaves_absolute_source_paths_in_place(tmp_path: Path) -> None:
    """An absolute [sources] path names a package that already exists on disk.

    ``env_dir / absolute`` collapses onto that package, so copying it would wipe
    and rewrite the package the entry points at.
    """
    external = tmp_path / "FooCap"
    (external / "src").mkdir(parents=True)
    (external / "Project.toml").write_text('name = "FooCap"\n', encoding="utf-8")
    (external / "src" / "FooCap.jl").write_text("module FooCap end\n", encoding="utf-8")

    template = tmp_path / "template"
    template.mkdir()
    (template / "Project.toml").write_text(
        "[deps]\n"
        'FooCap = "11111111-1111-1111-1111-111111111111"\n'
        "\n[sources]\n"
        f'FooCap = {{path = "{external.as_posix()}"}}\n',
        encoding="utf-8",
    )

    ws = tmp_path / "ws"
    ws.mkdir()
    bootstrap_julia_env(template, workspace=ws)

    assert (external / "src" / "FooCap.jl").exists()


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


def test_sync_adds_the_dependencys_own_deps_as_plain_deps(tmp_path: Path) -> None:
    """A capability's tools `include` its sources into Main.

    The `using` lines in those files resolve against this project, so the
    dependency's own deps are declared here; only the dependency itself gets a
    [sources] path.
    """
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
        'name = "FooCap"\n'
        'uuid = "11111111-1111-1111-1111-111111111111"\n'
        "[deps]\n"
        'Bar = "22222222-2222-2222-2222-222222222222"\n',
        encoding="utf-8",
    )

    added = sync_julia_project_with_dependencies(env, [local_pkg / "Project.toml"])
    assert added == ["Bar", "FooCap"]

    text = (env / "Project.toml").read_text(encoding="utf-8")
    assert 'FooCap = "11111111-1111-1111-1111-111111111111"' in text
    assert 'Bar = "22222222-2222-2222-2222-222222222222"' in text
    # Bar rides in as a normal dep, to be resolved from the registry.
    assert "Bar" not in text.split("[sources]")[-1]


def test_sync_keeps_an_existing_pin_for_a_dependencys_own_dep(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    env = workspace_julia_env(ws)
    env.mkdir(parents=True)
    (env / "Project.toml").write_text(
        '[deps]\nBar = "already-there"\n',
        encoding="utf-8",
    )

    local_pkg = tmp_path / "FooCap"
    local_pkg.mkdir()
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\n'
        'uuid = "11111111-1111-1111-1111-111111111111"\n'
        "[deps]\n"
        'Bar = "22222222-2222-2222-2222-222222222222"\n',
        encoding="utf-8",
    )

    assert sync_julia_project_with_dependencies(env, [local_pkg / "Project.toml"]) == ["FooCap"]

    text = (env / "Project.toml").read_text(encoding="utf-8")
    assert 'Bar = "already-there"' in text
    assert "22222222" not in text


def test_sync_adds_a_dependencys_own_deps_on_a_later_run(tmp_path: Path) -> None:
    """The package itself is already declared, its deps are not.

    This is the state after an upgrade that added a dep to the capability's
    package, so the second sync still has work to do.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    env = workspace_julia_env(ws)
    env.mkdir(parents=True)

    local_pkg = tmp_path / "FooCap"
    local_pkg.mkdir()
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\n'
        'uuid = "11111111-1111-1111-1111-111111111111"\n'
        "[deps]\n"
        'Bar = "22222222-2222-2222-2222-222222222222"\n',
        encoding="utf-8",
    )
    (env / "Project.toml").write_text(
        "[deps]\n"
        'FooCap = "11111111-1111-1111-1111-111111111111"\n'
        "\n[sources]\n"
        f'FooCap = {{path = "{local_pkg.as_posix()}"}}\n',
        encoding="utf-8",
    )

    assert sync_julia_project_with_dependencies(env, [local_pkg / "Project.toml"]) == ["Bar"]
    assert 'Bar = "22222222-2222-2222-2222-222222222222"' in (env / "Project.toml").read_text(
        encoding="utf-8"
    )


def test_sync_skips_a_dependencys_own_dep_that_it_path_sources(tmp_path: Path) -> None:
    """An unregistered sub-dependency cannot be declared without its own source.

    Pkg would fail with "expected package ... to be registered", so it is left
    out and reported instead.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    env = workspace_julia_env(ws)
    env.mkdir(parents=True)
    (env / "Project.toml").write_text('[deps]\nJutul = "uuid"\n', encoding="utf-8")

    local_pkg = tmp_path / "FooCap"
    local_pkg.mkdir()
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\n'
        'uuid = "11111111-1111-1111-1111-111111111111"\n'
        "[deps]\n"
        'SubLocal = "22222222-2222-2222-2222-222222222222"\n'
        "\n[sources]\n"
        'SubLocal = {path = "../SubLocal"}\n',
        encoding="utf-8",
    )

    warnings: list[str] = []
    added = sync_julia_project_with_dependencies(
        env, [local_pkg / "Project.toml"], warn=warnings.append
    )

    assert added == ["FooCap"]
    assert "SubLocal" not in (env / "Project.toml").read_text(encoding="utf-8")
    assert any("SubLocal" in msg for msg in warnings)


def test_sync_skips_a_dependency_whose_uuid_clashes_with_a_declared_dep(tmp_path: Path) -> None:
    """A name already resolved from elsewhere is left alone.

    Adding a [sources] path under it would make Pkg fail on the UUID mismatch.
    """
    ws = tmp_path / "ws"
    ws.mkdir()
    env = workspace_julia_env(ws)
    env.mkdir(parents=True)
    original = '[deps]\nFooCap = "99999999-9999-9999-9999-999999999999"\n'
    (env / "Project.toml").write_text(original, encoding="utf-8")

    local_pkg = tmp_path / "FooCap"
    local_pkg.mkdir()
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\nuuid = "11111111-1111-1111-1111-111111111111"\n',
        encoding="utf-8",
    )

    warnings: list[str] = []
    added = sync_julia_project_with_dependencies(
        env, [local_pkg / "Project.toml"], warn=warnings.append
    )

    assert added == []
    assert (env / "Project.toml").read_text(encoding="utf-8") == original
    assert any("clashes" in msg for msg in warnings)


def test_sync_warns_when_a_dependency_is_not_a_julia_package(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    env = workspace_julia_env(ws)
    env.mkdir(parents=True)
    (env / "Project.toml").write_text('[deps]\nJutul = "uuid"\n', encoding="utf-8")

    warnings: list[str] = []
    added = sync_julia_project_with_dependencies(env, [tmp_path / "nope"], warn=warnings.append)

    assert added == []
    assert any("not a Julia package" in msg for msg in warnings)


def test_sync_merges_local_preferences_from_dependency(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    env = workspace_julia_env(ws)
    env.mkdir(parents=True)
    (env / "Project.toml").write_text(
        '[deps]\nJutul = "c6b0b931-bd15-49f6-a31f-cf7d80eb5e81"\n',
        encoding="utf-8",
    )
    (env / "LocalPreferences.toml").write_text(
        '[Other]\na = 1\n\n[FooCap]\nexisting = "kept"\n',
        encoding="utf-8",
    )

    local_pkg = tmp_path / "FooCap"
    local_pkg.mkdir()
    (local_pkg / "Project.toml").write_text(
        'name = "FooCap"\nuuid = "11111111-1111-1111-1111-111111111111"\n',
        encoding="utf-8",
    )
    (local_pkg / "LocalPreferences.toml").write_text(
        '[FooCap]\nexisting = "clobbered"\nnew = "added"\n\n[Bar]\nb = 2\n',
        encoding="utf-8",
    )

    added = sync_julia_project_with_dependencies(env, [local_pkg / "Project.toml"])
    assert added == ["FooCap"]

    pref = tomllib.loads((env / "LocalPreferences.toml").read_text(encoding="utf-8"))
    assert pref["Other"] == {"a": 1}
    # A key the target already had for FooCap is not clobbered by the dependency's copy...
    assert pref["FooCap"]["existing"] == "kept"
    # ...but a new key the dependency adds is merged in.
    assert pref["FooCap"]["new"] == "added"
    # A whole new table from the dependency's LocalPreferences.toml is merged in too.
    assert pref["Bar"] == {"b": 2}


def test_sync_creates_local_preferences_when_target_has_none(tmp_path: Path) -> None:
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
    (local_pkg / "LocalPreferences.toml").write_text(
        "[FooCap]\nprecompile = true\n", encoding="utf-8"
    )

    sync_julia_project_with_dependencies(ws, [local_pkg / "Project.toml"])

    pref = tomllib.loads((ws / "LocalPreferences.toml").read_text(encoding="utf-8"))
    assert pref["FooCap"]["precompile"] is True


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


# --- capability package sources gate the launch precompile -------------------


def test_dependency_source_change_is_seen_even_though_the_manifest_is_not(tmp_path) -> None:
    """A capability's package is path-sourced and lives outside the env, so editing it
    leaves the Manifest untouched. Without this check launch skips the bake and the
    session's first `using` pays it, silently, inside a tool call."""
    from jutul_agent.workspace import (
        dependency_source_is_current,
        env_precompile_is_current,
        mark_dependency_source,
        mark_env_precompiled,
    )

    env = tmp_path / "env"
    env.mkdir()
    (env / "Manifest.toml").write_text("# manifest", encoding="utf-8")
    pkg = tmp_path / "SomeCapabilityPkg"
    (pkg / "src").mkdir(parents=True)
    (pkg / "Project.toml").write_text('name = "SomeCapabilityPkg"', encoding="utf-8")
    src = pkg / "src" / "SomeCapabilityPkg.jl"
    src.write_text("module SomeCapabilityPkg end", encoding="utf-8")
    deps = [pkg / "Project.toml"]

    mark_env_precompiled(env)
    mark_dependency_source(env, deps)
    assert env_precompile_is_current(env)
    assert dependency_source_is_current(env, deps)

    # Edit the capability's Julia source. The manifest is untouched, so the older
    # check still reports "current" — this is exactly the case it cannot see.
    src.write_text("module SomeCapabilityPkg\n# workload\nend", encoding="utf-8")
    assert env_precompile_is_current(env)
    assert not dependency_source_is_current(env, deps)

    mark_dependency_source(env, deps)
    assert dependency_source_is_current(env, deps)


def test_dependency_source_is_not_current_for_an_env_that_never_recorded_one(tmp_path) -> None:
    from jutul_agent.workspace import dependency_source_is_current

    env = tmp_path / "env"
    env.mkdir()
    assert not dependency_source_is_current(env, [tmp_path / "Nope" / "Project.toml"])
    # No capability packages at all is a stable, satisfiable state, not a forced bake.
    assert not dependency_source_is_current(env, [])
