"""Tests for the ``jutul-agent skill`` and ``jutul-agent simulator`` subcommands."""

from __future__ import annotations

from pathlib import Path

import pytest

from jutul_agent.interfaces.cli import main
from jutul_agent.paths import user_simulators_dir, user_skills_dir
from jutul_agent.simulators import registry

# A built-in simulator guaranteed present regardless of any user-skills/state-home
# overrides, since the registry discovers it from the shipped package modules.
_BUILTIN_SIM = "jutuldarcy"
assert _BUILTIN_SIM in registry.names()


# ---------------------------------------------------------------------------
# skill add
# ---------------------------------------------------------------------------


def test_skill_add_creates_global_scaffold(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["skill", "add", "mynewskill"])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    skill_md = user_skills_dir() / "mynewskill" / "SKILL.md"
    assert skill_md.exists()
    assert "name: mynewskill" in skill_md.read_text(encoding="utf-8")
    assert "Created skill scaffold" in captured.out


def test_skill_add_creates_scoped_scaffold_for_known_simulator(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["skill", "add", "wellrate-tips", "--sim", _BUILTIN_SIM])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    skill_md = user_simulators_dir() / _BUILTIN_SIM / "skills" / "wellrate-tips" / "SKILL.md"
    assert skill_md.exists()
    # Scoped skill must not also land in the global directory.
    assert not (user_skills_dir() / "wellrate-tips").exists()


def test_skill_add_rejects_unknown_simulator(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["skill", "add", "foo", "--sim", "not-a-real-simulator"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Unknown simulator" in captured.err
    assert not (user_skills_dir() / "foo").exists()


def test_skill_add_rejects_path_like_name_that_does_not_exist(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["skill", "add", "no/such/directory"])
    captured = capsys.readouterr()

    assert code == 1
    assert "looks like a path" in captured.err


def test_skill_add_errors_when_already_exists(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["skill", "add", "dupskill"]) == 0
    capsys.readouterr()

    code = main(["skill", "add", "dupskill"])
    captured = capsys.readouterr()

    assert code == 1
    assert "already exists" in captured.err


def test_skill_add_registers_existing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "external-skill"
    src.mkdir()
    (src / "SKILL.md").write_text(
        "---\nname: external-skill\ndescription: test\n---\nbody\n", encoding="utf-8"
    )

    code = main(["skill", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    dest = user_skills_dir() / "external-skill"
    assert (dest / "SKILL.md").read_text(encoding="utf-8") == (src / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert "Registered" in captured.out


def test_skill_add_rejects_directory_without_skill_md(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "not-a-skill"
    src.mkdir()

    code = main(["skill", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 1
    assert "does not contain a SKILL.md" in captured.err


# ---------------------------------------------------------------------------
# skill list
# ---------------------------------------------------------------------------


def test_skill_list_reports_when_empty(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["skill", "list"])
    captured = capsys.readouterr()

    assert code == 0
    assert "No user skills found." in captured.out


def test_skill_list_shows_global_and_scoped_skills(capsys: pytest.CaptureFixture[str]) -> None:
    main(["skill", "add", "globalskill"])
    main(["skill", "add", "simskill", "--sim", _BUILTIN_SIM])
    capsys.readouterr()

    code = main(["skill", "list"])
    captured = capsys.readouterr()

    assert code == 0
    assert "[global]  globalskill" in captured.out
    assert f"[{_BUILTIN_SIM}]  simskill" in captured.out


# ---------------------------------------------------------------------------
# skill remove
# ---------------------------------------------------------------------------


def test_skill_remove_deletes_scaffold(capsys: pytest.CaptureFixture[str]) -> None:
    main(["skill", "add", "removable"])
    capsys.readouterr()

    code = main(["skill", "remove", "removable"])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    assert not (user_skills_dir() / "removable").exists()
    assert "Removed" in captured.out


def test_skill_remove_unlinks_symlink_without_deleting_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "shared-skill"
    src.mkdir()
    (src / "SKILL.md").write_text(
        "---\nname: shared-skill\ndescription: x\n---\n", encoding="utf-8"
    )
    main(["skill", "add", str(src)])
    capsys.readouterr()

    code = main(["skill", "remove", "shared-skill"])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    assert not (user_skills_dir() / "shared-skill").exists()
    assert (src / "SKILL.md").exists()  # original source untouched


def test_skill_remove_errors_when_missing(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["skill", "remove", "ghost"])
    captured = capsys.readouterr()

    assert code == 1
    assert "not found" in captured.err


def test_skill_remove_rejects_unknown_simulator(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["skill", "remove", "foo", "--sim", "not-a-real-simulator"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Unknown simulator" in captured.err


def test_skill_remove_respects_sim_scope(capsys: pytest.CaptureFixture[str]) -> None:
    main(["skill", "add", "scoped", "--sim", _BUILTIN_SIM])
    capsys.readouterr()

    # Removing without --sim looks in the global dir, where it doesn't exist.
    code = main(["skill", "remove", "scoped"])
    captured = capsys.readouterr()
    assert code == 1
    assert "not found" in captured.err

    code = main(["skill", "remove", "scoped", "--sim", _BUILTIN_SIM])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert not (user_simulators_dir() / _BUILTIN_SIM / "skills" / "scoped").exists()


# ---------------------------------------------------------------------------
# simulator add
# ---------------------------------------------------------------------------


def test_simulator_add_creates_scaffold_with_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["simulator", "add", "mysim"])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    sim_dir = user_simulators_dir() / "mysim"
    adapter_text = (sim_dir / "adapter.py").read_text(encoding="utf-8")
    assert 'name="mysim"' in adapter_text
    assert 'display_name="Mysim"' in adapter_text
    assert (sim_dir / "skills" / "mysim-overview" / "SKILL.md").exists()
    assert "Created simulator scaffold" in captured.out


def test_simulator_add_creates_scaffold_with_custom_display_name_and_packages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "simulator",
            "add",
            "mysim2",
            "--display-name",
            "My Sim",
            "--packages",
            "PkgA,PkgB",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0, captured.err
    adapter_text = (user_simulators_dir() / "mysim2" / "adapter.py").read_text(encoding="utf-8")
    assert 'display_name="My Sim"' in adapter_text
    assert '"PkgA", "PkgB"' in adapter_text
    assert 'primary_package="PkgA"' in adapter_text


def test_simulator_add_errors_when_directory_exists(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["simulator", "add", "dupsim"]) == 0
    capsys.readouterr()

    code = main(["simulator", "add", "dupsim"])
    captured = capsys.readouterr()

    assert code == 1
    assert "already exists" in captured.err


def test_simulator_add_rejects_path_like_name_that_does_not_exist(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["simulator", "add", "no/such/directory"])
    captured = capsys.readouterr()

    assert code == 1
    assert "looks like a path" in captured.err


def test_simulator_add_registers_existing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "external-sim"
    src.mkdir()
    (src / "adapter.py").write_text("# adapter stub\n", encoding="utf-8")

    code = main(["simulator", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    dest = user_simulators_dir() / "external-sim"
    assert (dest / "adapter.py").read_text(encoding="utf-8") == "# adapter stub\n"
    assert "Registered" in captured.out


def test_simulator_add_rejects_directory_without_adapter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "not-a-sim"
    src.mkdir()

    code = main(["simulator", "add", str(src)])
    captured = capsys.readouterr()

    assert code == 1
    assert "does not contain an adapter.py" in captured.err


# ---------------------------------------------------------------------------
# simulator list
# ---------------------------------------------------------------------------


def test_simulator_list_shows_builtins_with_no_user_simulators(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["simulator", "list"])
    captured = capsys.readouterr()

    assert code == 0
    assert "Built-in simulators:" in captured.out
    assert _BUILTIN_SIM in captured.out
    assert "User simulators: (none)" in captured.out


def test_simulator_list_shows_user_simulators(capsys: pytest.CaptureFixture[str]) -> None:
    main(["simulator", "add", "listedsim"])
    capsys.readouterr()

    code = main(["simulator", "list"])
    captured = capsys.readouterr()

    assert code == 0
    assert "User simulators:" in captured.out
    assert "listedsim" in captured.out


# ---------------------------------------------------------------------------
# simulator remove
# ---------------------------------------------------------------------------


def test_simulator_remove_user_simulator(capsys: pytest.CaptureFixture[str]) -> None:
    main(["simulator", "add", "removablesim"])
    capsys.readouterr()

    code = main(["simulator", "remove", "removablesim"])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    assert not (user_simulators_dir() / "removablesim").exists()
    assert "Removed" in captured.out


def test_simulator_remove_protects_builtin_without_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["simulator", "remove", _BUILTIN_SIM])
    captured = capsys.readouterr()

    assert code == 1
    assert "cannot be removed" in captured.err
    assert not (user_simulators_dir() / _BUILTIN_SIM).exists()


def test_simulator_remove_allows_removing_builtin_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    override = user_simulators_dir() / _BUILTIN_SIM
    override.mkdir(parents=True)
    (override / "adapter.py").write_text("# override stub\n", encoding="utf-8")

    code = main(["simulator", "remove", _BUILTIN_SIM])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    assert not override.exists()


def test_simulator_remove_errors_when_missing(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["simulator", "remove", "ghost-sim"])
    captured = capsys.readouterr()

    assert code == 1
    assert "not found" in captured.err
