"""The in-env JutulAgent runtime auto-refreshes when the install's source changes.

Warm packages are dev-pathed copies laid down at bootstrap, so without this an
existing managed env keeps a stale JutulAgent after a jutul-agent update.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jutul_agent import workspace
from jutul_agent.simulators import env_setup
from jutul_agent.simulators.base import SimulatorAdapter
from jutul_agent.workspace import (
    PRECOMPILE_MARKER,
    WARM_SOURCE_MARKER,
    mark_warm_source,
    recopy_warm_sources,
    warm_source_fingerprint,
    warm_source_is_current,
)


@pytest.fixture
def shared_pkg(tmp_path: Path, monkeypatch) -> Path:
    """A fake shared JutulAgent source the fingerprint reads from."""
    pkg = tmp_path / "shared" / "JutulAgent"
    (pkg / "src").mkdir(parents=True)
    (pkg / "Project.toml").write_text('name = "JutulAgent"\n', encoding="utf-8")
    (pkg / "src" / "ensemble.jl").write_text("# v1\n", encoding="utf-8")
    monkeypatch.setattr(workspace, "shared_julia_package_path", lambda: pkg)
    return pkg


@pytest.fixture
def template(tmp_path: Path) -> Path:
    """A template env with a per-simulator warm package and the [sources] table."""
    tpl = tmp_path / "template"
    (tpl / "JutulAgentSim" / "src").mkdir(parents=True)
    (tpl / "JutulAgentSim" / "Project.toml").write_text(
        'name = "JutulAgentSim"\n', encoding="utf-8"
    )
    (tpl / "JutulAgentSim" / "src" / "warm.jl").write_text("# warm v1\n", encoding="utf-8")
    (tpl / "Project.toml").write_text(
        "[deps]\n\n[sources]\n"
        'JutulAgent = {path = "JutulAgent"}\n'
        'JutulAgentSim = {path = "JutulAgentSim"}\n',
        encoding="utf-8",
    )
    return tpl


def _make_env(tmp_path: Path, template: Path, shared_pkg: Path) -> Path:
    """An env bootstrapped from the template (warm sources copied + fingerprinted)."""
    env = tmp_path / "env"
    env.mkdir()
    (env / "Project.toml").write_text(
        template.joinpath("Project.toml").read_text(), encoding="utf-8"
    )
    workspace._copy_source_packages(template, env, {"JutulAgentSim": {"path": "JutulAgentSim"}})
    workspace.sync_shared_julia_package(env)
    workspace.mark_warm_source(env, template)
    (env / PRECOMPILE_MARKER).touch()
    return env


def test_fresh_env_is_current(tmp_path, template, shared_pkg):
    env = _make_env(tmp_path, template, shared_pkg)
    assert warm_source_is_current(env, template) is True


def test_fingerprint_tracks_shared_and_template_sources(tmp_path, template, shared_pkg):
    before = warm_source_fingerprint(template)
    (shared_pkg / "src" / "ensemble.jl").write_text("# v2 changed\n", encoding="utf-8")
    assert warm_source_fingerprint(template) != before
    after_shared = warm_source_fingerprint(template)
    (template / "JutulAgentSim" / "src" / "warm.jl").write_text("# warm v2\n", encoding="utf-8")
    assert warm_source_fingerprint(template) != after_shared


def test_recopy_then_mark_is_the_refresh_cycle(tmp_path, template, shared_pkg):
    env = _make_env(tmp_path, template, shared_pkg)
    # The install's shared source changes (a jutul-agent update).
    (shared_pkg / "src" / "ensemble.jl").write_text("# v2 changed\n", encoding="utf-8")
    assert warm_source_is_current(env, template) is False

    assert recopy_warm_sources(env, template) is True
    # New source copied into the env and the precompile marker dropped (re-bake).
    assert (env / "JutulAgent" / "src" / "ensemble.jl").read_text() == "# v2 changed\n"
    assert not (env / PRECOMPILE_MARKER).exists()
    # recopy does NOT mark current — the caller marks only after a successful resolve,
    # so a failed resolve leaves it stale and the next launch retries.
    assert warm_source_is_current(env, template) is False

    mark_warm_source(env, template)
    assert warm_source_is_current(env, template) is True


def test_never_fingerprinted_env_is_stale(tmp_path, template, shared_pkg):
    env = _make_env(tmp_path, template, shared_pkg)
    (env / WARM_SOURCE_MARKER).unlink()  # an env from before this feature existed
    assert warm_source_is_current(env, template) is False


def test_recopy_skips_user_owned_env(tmp_path, template, shared_pkg):
    env = _make_env(tmp_path, template, shared_pkg)
    (env / "Project.toml").write_text("[deps]\n", encoding="utf-8")  # no [sources]
    assert recopy_warm_sources(env, template) is False


def test_failed_refresh_restores_previous_sources_and_markers(
    tmp_path: Path, template: Path, shared_pkg: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _make_env(tmp_path, template, shared_pkg)
    (env / "Manifest.toml").write_text("old manifest", encoding="utf-8")
    old_marker = (env / WARM_SOURCE_MARKER).read_text(encoding="utf-8")
    (shared_pkg / "src" / "ensemble.jl").write_text("# v2 changed\n", encoding="utf-8")

    def failed_resolve(project: Path, **_kwargs) -> None:
        (project / "Manifest.toml").write_text("partial manifest", encoding="utf-8")
        raise env_setup.EnvSetupError("resolve failed")

    monkeypatch.setattr(env_setup, "resolve_and_instantiate", failed_resolve)
    adapter = SimulatorAdapter(
        name="test",
        display_name="Test",
        module_dir=template.parent,
        package_imports=("JutulAgentSim",),
        primary_package="JutulAgentSim",
        domain_hints="",
    )
    monkeypatch.setattr(
        SimulatorAdapter, "julia_env_template_path", property(lambda _self: template)
    )

    env_setup._refresh_warm_sources(adapter, tmp_path, env, "test")

    assert (env / "JutulAgent" / "src" / "ensemble.jl").read_text() == "# v1\n"
    assert (env / "Manifest.toml").read_text() == "old manifest"
    assert (env / WARM_SOURCE_MARKER).read_text(encoding="utf-8") == old_marker
    assert (env / PRECOMPILE_MARKER).exists()
