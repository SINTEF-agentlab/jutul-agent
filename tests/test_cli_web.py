from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


def test_sysimage_preflight_prepares_env_before_final_decision(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A bundled-source refresh must refuse before uvicorn serves a dead UI."""
    from jutul_agent import sysimage
    from jutul_agent.agent import capabilities
    from jutul_agent.interfaces.cli.web import _sysimage_ready
    from jutul_agent.simulators import env_setup, registry
    from jutul_agent.workspace import WorkspaceConfig, workspace_julia_env

    project = workspace_julia_env(tmp_path)
    project.mkdir(parents=True)
    image = tmp_path / "agent.dylib"
    decisions = iter(
        [
            sysimage.Decision(status=sysimage.CURRENT, path=image),
            sysimage.Decision(
                status=sysimage.DIVERGENT,
                reason="  edited since the image was built:\n    JutulAgentJutulDarcy",
            ),
        ]
    )
    prepared: list[tuple[object, Path, Path, str | None, list[Path]]] = []
    adapter = SimpleNamespace(name="jutuldarcy")

    monkeypatch.setattr(sysimage, "decide", lambda *args, **kwargs: next(decisions))
    monkeypatch.setattr(registry, "get", lambda name: adapter)
    monkeypatch.setattr(capabilities, "discover_extensions", lambda: [])
    monkeypatch.setattr(
        env_setup,
        "prepare_workspace_env",
        lambda selected, **kw: prepared.append(
            (
                selected,
                kw["workspace"],
                kw["julia_project"],
                kw["sim_name"],
                kw["dependencies"],
            )
        ),
    )

    args = Namespace(julia_project=None, sysimage=None)
    ready = _sysimage_ready(
        args,
        tmp_path,
        WorkspaceConfig(sysimage=True),
        sim="jutuldarcy",
    )

    assert ready is False
    assert prepared == [(adapter, tmp_path, project, "jutuldarcy", [])]
    stderr = capsys.readouterr().err
    assert "JutulAgentJutulDarcy" in stderr
    assert "jutul-agent sysimage build" in stderr


def test_sysimage_preflight_leaves_explicit_julia_project_alone(
    tmp_path: Path, monkeypatch
) -> None:
    from jutul_agent import sysimage
    from jutul_agent.interfaces.cli.web import _sysimage_ready
    from jutul_agent.simulators import env_setup
    from jutul_agent.workspace import WorkspaceConfig

    project = tmp_path / "shared-env"
    image = tmp_path / "agent.dylib"
    monkeypatch.setattr(
        sysimage,
        "decide",
        lambda *args, **kwargs: sysimage.Decision(status=sysimage.CURRENT, path=image),
    )

    def fail_if_prepared(*args, **kwargs):
        raise AssertionError("an explicit Julia project must not be prepared")

    monkeypatch.setattr(env_setup, "prepare_workspace_env", fail_if_prepared)
    assert _sysimage_ready(
        Namespace(julia_project=project, sysimage=None),
        tmp_path,
        WorkspaceConfig(sysimage=True),
        sim="jutuldarcy",
    )
