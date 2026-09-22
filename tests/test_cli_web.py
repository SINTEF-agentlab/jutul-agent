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
    from jutul_agent.workspace import WorkspaceConfig

    project = tmp_path / "julia-env"
    project.mkdir()
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

    args = Namespace(julia_project=project, sysimage=None)
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
