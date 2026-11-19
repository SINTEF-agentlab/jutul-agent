"""Carrying a jutul-agent install over to Makker (temporary; see ``makker.legacy``)."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from makker import legacy, paths
from makker.legacy import adopt_previous_install as real_adopt_previous_install
from makker.session import existing_output_dir

posix_only = pytest.mark.skipif(os.name == "nt", reason="needs a symlink, which Windows may refuse")

# A workspace env as jutul-agent 0.2.1 left it, with one package the user added.
OLD_PROJECT = """\
[deps]
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
JutulAgent = "79b12aa6-73c9-4fa8-9c54-e97745f42687"
JutulAgentJutulDarcy = "69df87d8-8b4b-4157-81d2-8b93ff139141"
JutulDarcy = "82210473-ab04-4dce-b31b-11573c4f8e0a"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
JutulDarcy = "0.3"

[sources]
JutulAgent = {path = "JutulAgent"}
JutulAgentJutulDarcy = {path = "JutulAgentJutulDarcy"}
"""


def _old_data_folder(root: Path) -> Path:
    old = root / "jutul-agent"
    (old / "workspaces" / "abc" / "memory").mkdir(parents=True)
    (old / ".env").write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    (old / "workspaces" / "abc" / "memory" / "MEMORY.md").write_text("notes\n", encoding="utf-8")
    (old / "update-check.json").write_text('{"checked_at": 1, "latest": "0.3.0"}', encoding="utf-8")
    return old


# --- the data folder --------------------------------------------------------


def test_data_is_found_under_the_old_name_until_it_has_been_moved(tmp_path: Path) -> None:
    new = tmp_path / "makker"
    assert legacy.resolve_state_home(new) == new  # a fresh install

    old = _old_data_folder(tmp_path)
    assert legacy.resolve_state_home(new) == old

    new.mkdir()
    assert legacy.resolve_state_home(new) == new


@posix_only
def test_moving_the_data_folder_keeps_the_old_path_working(tmp_path: Path, capsys) -> None:
    old = _old_data_folder(tmp_path)
    new = tmp_path / "makker"

    legacy.migrate_state_home(new)

    assert (new / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=sk-test\n"
    assert old.is_symlink()
    # A saved chat holds the old absolute path to its memory notes.
    assert (old / "workspaces" / "abc" / "memory" / "MEMORY.md").read_text() == "notes\n"
    assert not (new / "update-check.json").exists()
    assert legacy.resolve_state_home(new) == new
    assert "Moved your data" in capsys.readouterr().err

    legacy.migrate_state_home(new)  # nothing left to do
    assert capsys.readouterr().err == ""


def test_a_data_folder_that_cannot_be_moved_is_used_where_it_is(
    tmp_path: Path, monkeypatch
) -> None:
    old = _old_data_folder(tmp_path)
    new = tmp_path / "makker"

    def in_use(self, target):
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "rename", in_use)
    legacy.migrate_state_home(new)

    assert not new.exists()  # creating it would hide the old folder for good
    assert legacy.resolve_state_home(new) == old


def test_two_real_data_folders_are_left_alone_and_mentioned_once(tmp_path: Path, capsys) -> None:
    old = _old_data_folder(tmp_path)
    new = tmp_path / "makker"
    new.mkdir()

    legacy.migrate_state_home(new)
    assert "is not used" in capsys.readouterr().err
    legacy.migrate_state_home(new)
    assert capsys.readouterr().err == ""
    assert (old / ".env").exists()


def test_the_default_data_folder_falls_back_to_the_old_one(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    paths.set_state_home(None)
    assert paths.state_home() == tmp_path.resolve() / "makker"
    _old_data_folder(tmp_path.resolve())
    assert paths.state_home() == tmp_path.resolve() / "jutul-agent"


# --- a workspace ------------------------------------------------------------


def _old_workspace(workspace: Path) -> Path:
    env = workspace / ".jutul-agent" / "julia-env"
    for package in ("JutulAgent", "JutulAgentJutulDarcy"):
        (env / package / "src").mkdir(parents=True)
        (env / package / "Project.toml").write_text(f'name = "{package}"\n', encoding="utf-8")
    (env / "Project.toml").write_text(OLD_PROJECT, encoding="utf-8")
    (env / "Manifest.toml").write_text("# resolved\n", encoding="utf-8")
    (env / ".jutul-agent-precompiled").write_text("", encoding="utf-8")
    (workspace / ".jutul-agent" / "config.toml").write_text(
        'simulator = "jutuldarcy"\n', encoding="utf-8"
    )
    return env


def test_a_workspace_folder_is_renamed_and_ignores_itself(tmp_path: Path) -> None:
    _old_workspace(tmp_path)

    legacy.adopt_workspace(tmp_path)

    assert not (tmp_path / ".jutul-agent").exists()
    assert (tmp_path / ".makker" / "config.toml").read_text() == 'simulator = "jutuldarcy"\n'
    assert (tmp_path / ".makker" / ".gitignore").read_text(encoding="utf-8") == "*\n"
    legacy.adopt_workspace(tmp_path)  # again: nothing to do


def test_a_workspace_that_already_has_the_new_folder_is_left_alone(tmp_path: Path) -> None:
    _old_workspace(tmp_path)
    (tmp_path / ".makker").mkdir()

    legacy.adopt_workspace(tmp_path)

    assert (tmp_path / ".jutul-agent" / "config.toml").exists()


def test_a_workspace_folder_in_use_stops_with_what_to_do(tmp_path: Path, monkeypatch) -> None:
    _old_workspace(tmp_path)

    def in_use(self, target):
        raise PermissionError("in use")

    monkeypatch.setattr(Path, "rename", in_use)
    with pytest.raises(SystemExit) as stop:
        legacy.adopt_workspace(tmp_path)
    assert "Close any other Makker or jutul-agent program" in str(stop.value)


def test_the_old_julia_packages_leave_the_env_and_the_rest_stays(tmp_path: Path) -> None:
    env = _old_workspace(tmp_path)

    assert legacy.convert_env(env) is True

    project = tomllib.loads((env / "Project.toml").read_text(encoding="utf-8"))
    assert sorted(project["deps"]) == ["GLMakie", "JutulDarcy", "Statistics"]
    assert project["compat"] == {"JutulDarcy": "0.3"}
    assert project.get("sources", {}) == {}
    assert not (env / "JutulAgent").exists()
    assert not (env / "JutulAgentJutulDarcy").exists()
    assert not (env / ".jutul-agent-precompiled").exists()
    assert (env / "Manifest.toml").exists()  # the resolved versions are kept

    assert legacy.convert_env(env) is False  # and it is done once


def test_an_old_session_is_found_in_the_output_folder_it_wrote(tmp_path: Path) -> None:
    session_id = "2026-10-01-1200-ab12"
    old = tmp_path / "jutul-agent-output" / "sessions" / f"{session_id}-battery-study"
    (old / "artifacts").mkdir(parents=True)
    (old / "artifacts" / "plot.png").write_bytes(b"png")

    assert existing_output_dir(session_id) == old

    # A session written since has its folder under the new name, and that one wins.
    new = tmp_path / "makker-output" / "sessions" / session_id
    (new / "artifacts").mkdir(parents=True)
    (new / "artifacts" / "plot.png").write_bytes(b"png")
    assert existing_output_dir(session_id) == new


# --- settings and the old install --------------------------------------------


def test_only_settings_without_a_new_spelling_are_pointed_out() -> None:
    environ = {
        "JUTUL_AGENT_MODEL": "openai:gpt-5.4",
        "JUTUL_AGENT_NO_OPEN": "1",
        "MAKKER_NO_OPEN": "1",  # what the last jutul-agent release sets alongside
        "PATH": "/usr/bin",
    }
    assert legacy.old_settings(environ) == ["JUTUL_AGENT_MODEL"]


def _tools(root: Path, old_version: str) -> Path:
    """A uv tools folder holding Makker and a jutul-agent of ``old_version``."""
    makker = root / "tools" / "makker"
    makker.mkdir(parents=True)
    (makker / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{ name = "makker" }]\n', encoding="utf-8"
    )
    old = root / "tools" / "jutul-agent"
    site = old / "lib" / "python3.14" / "site-packages" / f"jutul_agent-{old_version}.dist-info"
    site.mkdir(parents=True)
    (old / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{ name = "jutul-agent" }]\n', encoding="utf-8"
    )
    return makker


@pytest.fixture
def uninstalls(monkeypatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(legacy.subprocess, "run", fake_run)
    monkeypatch.setattr(legacy.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv(legacy.VIA_OLD_COMMAND, raising=False)
    return calls


def test_the_last_jutul_agent_release_is_removed_once_makker_runs_as_itself(
    tmp_path: Path, monkeypatch, uninstalls, capsys
) -> None:
    monkeypatch.setattr(sys, "prefix", str(_tools(tmp_path, "0.3.0")))

    legacy.remove_old_install()

    assert uninstalls == [["/usr/bin/uv", "tool", "uninstall", "jutul-agent"]]
    assert "The command is `makker` from now on." in capsys.readouterr().err


def test_an_earlier_jutul_agent_is_a_working_program_and_is_kept(
    tmp_path: Path, monkeypatch, uninstalls
) -> None:
    monkeypatch.setattr(sys, "prefix", str(_tools(tmp_path, "0.2.1")))
    legacy.remove_old_install()
    assert uninstalls == []


def test_nothing_is_removed_while_started_through_the_old_command(
    tmp_path: Path, monkeypatch, uninstalls
) -> None:
    monkeypatch.setattr(sys, "prefix", str(_tools(tmp_path, "0.3.0")))
    monkeypatch.setenv(legacy.VIA_OLD_COMMAND, "1")
    legacy.remove_old_install()
    assert uninstalls == []


def test_nothing_is_removed_from_inside_the_old_install_itself(
    tmp_path: Path, monkeypatch, uninstalls
) -> None:
    # Makker running in place inside the jutul-agent tool, because the step onto
    # its own tool could not be taken.
    makker_tool = _tools(tmp_path, "0.3.0")
    monkeypatch.setattr(sys, "prefix", str(makker_tool.parent / "jutul-agent"))
    legacy.remove_old_install()
    assert uninstalls == []


# --- all of it ---------------------------------------------------------------


@posix_only
def test_a_named_data_folder_is_the_users_and_is_not_moved(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "not-a-tool"))
    old = _old_data_folder(tmp_path)
    _old_workspace(tmp_path / "ws")

    real_adopt_previous_install(tmp_path / "ws", default_state_home=None)
    assert not old.is_symlink()
    assert (tmp_path / "ws" / ".makker").is_dir()

    real_adopt_previous_install(tmp_path / "ws", default_state_home=tmp_path / "makker")
    assert old.is_symlink()


def test_a_converted_env_takes_on_the_renamed_packages_like_any_template_addition(
    tmp_path: Path,
) -> None:
    from makker.simulators import registry
    from makker.workspace import sync_julia_env_with_template

    _old_workspace(tmp_path)
    legacy.adopt_workspace(tmp_path)
    env = tmp_path / ".makker" / "julia-env"
    legacy.convert_env(env)

    template = registry.get("jutuldarcy").julia_env_template_path
    added = sync_julia_env_with_template(template, workspace=tmp_path)

    assert {"Makker", "MakkerJutulDarcy"} <= set(added)
    project = tomllib.loads((env / "Project.toml").read_text(encoding="utf-8"))
    assert not [name for name in project["deps"] if name.startswith("JutulAgent")]
    assert project["sources"]["Makker"] == {"path": "Makker"}
    assert (env / "Makker" / "Project.toml").is_file()
    assert (env / "MakkerJutulDarcy" / "Project.toml").is_file()
    assert "Statistics" in project["deps"]  # what the user added is still there
