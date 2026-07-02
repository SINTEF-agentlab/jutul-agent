"""The simulator registry auto-discovers adapters from the package folders."""

from __future__ import annotations

import pytest

from jutul_agent.simulators import registry
from jutul_agent.simulators.base import SimulatorAdapter


def test_known_simulators_are_discovered() -> None:
    names = registry.names()
    for expected in ("jutuldarcy", "battmo", "fimbul", "mocca"):
        assert expected in names


def test_get_returns_the_adapter() -> None:
    adapter = registry.get("jutuldarcy")
    assert isinstance(adapter, SimulatorAdapter)
    assert adapter.name == "jutuldarcy"


def test_names_are_sorted() -> None:
    assert registry.names() == sorted(registry.names())


def test_unknown_simulator_raises() -> None:
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


def _adapter(tmp_path, name: str) -> SimulatorAdapter:
    from fakes import make_fake_adapter

    adapter = make_fake_adapter(tmp_path, name=name, display_name=name.title())
    # Discovery validates adapters now; give the fake a real env template.
    env = adapter.julia_env_template_path
    env.mkdir(parents=True, exist_ok=True)
    (env / "Project.toml").write_text("[deps]\n", encoding="utf-8")
    return adapter


def test_installed_adapter_is_discovered(tmp_path, monkeypatch) -> None:
    """An installed package can add a simulator via the entry-point group."""
    adapter = _adapter(tmp_path, "mysim")

    class _EP:
        name = "mysim"

        def load(self):
            return adapter

    monkeypatch.setattr(registry, "_registry", registry._registry.__wrapped__)
    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        lambda group: [_EP()] if group == registry.SIMULATOR_ENTRY_POINT_GROUP else [],
    )
    assert "mysim" in registry.names()
    assert registry.get("mysim") is adapter


def test_installed_adapter_overrides_bundled(tmp_path, monkeypatch) -> None:
    """An installed adapter wins over a bundled one of the same name (own JutulDarcy)."""
    custom = _adapter(tmp_path, "jutuldarcy")

    class _EP:
        name = "jutuldarcy"

        def load(self):
            return custom

    monkeypatch.setattr(registry, "_registry", registry._registry.__wrapped__)
    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        lambda group: [_EP()] if group == registry.SIMULATOR_ENTRY_POINT_GROUP else [],
    )
    assert registry.get("jutuldarcy") is custom


def test_broken_installed_entry_point_is_skipped(tmp_path, monkeypatch) -> None:
    class _Broken:
        name = "brokensim"

        def load(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(registry, "_registry", registry._registry.__wrapped__)
    monkeypatch.setattr(
        registry.importlib_metadata,
        "entry_points",
        lambda group: [_Broken()] if group == registry.SIMULATOR_ENTRY_POINT_GROUP else [],
    )
    # The bundled simulators are still there; the broken one is ignored.
    assert "jutuldarcy" in registry.names()


@pytest.mark.parametrize("name", registry.names())
def test_env_template_declares_the_web_plotting_backend(name: str) -> None:
    """Every simulator env resolves WGLMakie and Bonito alongside the simulator.

    They have to be deps of the same environment, not of a second one stacked on
    top: Julia resolves each environment independently, so a standalone resolve
    picks the newest Makie and HTTP rather than the ones the simulator's stack
    pins, and whichever environment comes first in the load path then wins for
    every package both provide. The packages that lost the tie are left running
    against versions they were never compiled for.
    """

    import tomllib

    template = registry.get(name).julia_env_template_path / "Project.toml"
    deps = tomllib.loads(template.read_text(encoding="utf-8")).get("deps", {})
    assert {"WGLMakie", "Bonito"} <= set(deps), f"{name} env template is missing web backends"
