"""Tests for runtime prompt helpers, Deep Agents assets, and simulator registry."""

from __future__ import annotations

from pathlib import Path

from deepagents.backends import CompositeBackend, LocalShellBackend

from jutul_agent.agent.builder import build_backend, skill_sources
from jutul_agent.agent.prompts import assemble_session_prompt
from jutul_agent.lab.fakes import make_fake_adapter
from jutul_agent.paths import SHARED_SKILLS_DIR
from jutul_agent.simulators import registry
from jutul_agent.simulators.base import SimulatorAdapter


def test_registry_lists_known_simulators() -> None:
    assert "jutuldarcy" in registry.names()
    assert "battmo" in registry.names()
    assert "fimbul" in registry.names()
    assert "mocca" in registry.names()


def test_assemble_session_prompt_includes_runtime_context(tmp_path: Path) -> None:
    adapter = SimulatorAdapter(
        name="testsim",
        display_name="TestSim",
        module_dir=tmp_path,
        package_imports=("Jutul", "TestSim"),
        primary_package="TestSim",
        domain_hints="use the runtime and inspect installed packages",
    )

    prompt = assemble_session_prompt(adapter)

    assert "TestSim" in prompt
    assert "Jutul, TestSim" in prompt
    assert "inspect installed packages" in prompt
    assert "run_julia" in prompt
    assert "write_file" in prompt
    assert "workspace" in prompt
    # Installed source is reachable by its pkgdir path (the general mechanism); with
    # no resolved primary source, the prompt does not yet hand over a concrete path.
    assert "pkgdir(<Package>)" in prompt
    assert "already on disk at" not in prompt
    assert "/packages/" not in prompt
    # Retry guidance lives in HarnessProfile suffix, not the static prompt.
    assert "retry before responding" not in prompt


def test_assemble_session_prompt_hands_over_resolved_source_path(tmp_path: Path) -> None:
    # When the simulator package's source path is resolved at session start, the
    # prompt states it directly so the agent reads it without `using <Sim>;
    # pkgdir(<Sim>)` (which would load the package just to find a known path).
    adapter = SimulatorAdapter(
        name="testsim",
        display_name="TestSim",
        module_dir=tmp_path,
        package_imports=("Jutul", "TestSim"),
        primary_package="TestSim",
        domain_hints="",
    )
    src = str(tmp_path / "depot" / "TestSim")

    prompt = assemble_session_prompt(adapter, primary_source=src)

    assert src in prompt
    assert "already on disk at" in prompt
    assert "Do not run `using TestSim`" in prompt  # don't pay the load to find it
    assert "pkgdir(<Package>)" in prompt  # other packages still use pkgdir


def test_build_backend_is_a_single_real_path_backend(tmp_path: Path) -> None:
    adapter = make_fake_adapter(tmp_path)

    backend = build_backend(workspace=tmp_path)

    # One real-path backend over the workspace.
    assert isinstance(backend, CompositeBackend)
    assert isinstance(backend.default, LocalShellBackend)
    assert backend.default.virtual_mode is False
    assert backend.routes == {}
    # Skills are sourced from their real directories (the seam for user/project
    # skills is appending more real dirs here, last wins).
    assert skill_sources(adapter) == [
        (str(SHARED_SKILLS_DIR), "Built-in"),
        (str(adapter.skills_dir), adapter.display_name),
    ]


def test_repo_deepagents_skill_assets_exist() -> None:
    assert SHARED_SKILLS_DIR.joinpath("julia-and-repl", "SKILL.md").exists()
    assert SHARED_SKILLS_DIR.joinpath("workspace-and-source", "SKILL.md").exists()
    assert SHARED_SKILLS_DIR.joinpath("plotting-basics", "SKILL.md").exists()
    expected = [
        ("jutuldarcy", "jutuldarcy-overview"),
        ("jutuldarcy", "jutuldarcy-wells"),
        ("battmo", "battmo-overview"),
        ("battmo", "battmo-cycling"),
        ("fimbul", "fimbul-overview"),
        ("mocca", "mocca-overview"),
    ]
    for sim, skill in expected:
        assert registry.get(sim).skills_dir.joinpath(skill, "SKILL.md").exists()


def test_real_adapters_produce_nonempty_session_prompt() -> None:
    for name in registry.names():
        adapter = registry.get(name)
        prompt = assemble_session_prompt(adapter)
        assert adapter.display_name in prompt
        assert adapter.name in prompt


def test_every_simulator_has_env_template_and_overview_skill() -> None:
    for name in registry.names():
        adapter = registry.get(name)
        template = adapter.julia_env_template_path / "Project.toml"
        assert template.exists(), f"missing env template for {name}: {template}"
        overview = adapter.skills_dir / f"{name}-overview" / "SKILL.md"
        assert overview.exists(), f"missing overview skill for {name}: {overview}"


def test_every_skill_frontmatter_parses_with_name_and_description() -> None:
    """A malformed frontmatter makes deepagents silently skip the skill.

    An unquoted colon in ``description:`` is enough; the skill then never
    reaches the agent, with only a session-time warning to show for it.
    """
    import yaml

    skill_files = list(SHARED_SKILLS_DIR.glob("*/SKILL.md"))
    for name in registry.names():
        skill_files.extend(registry.get(name).skills_dir.glob("*/SKILL.md"))
    assert skill_files

    for skill in skill_files:
        text = skill.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{skill}: missing frontmatter"
        frontmatter = text.split("---\n", 2)[1]
        meta = yaml.safe_load(frontmatter)
        assert isinstance(meta, dict), f"{skill}: frontmatter is not a mapping"
        assert meta.get("name"), f"{skill}: frontmatter lacks name"
        assert meta.get("description"), f"{skill}: frontmatter lacks description"
