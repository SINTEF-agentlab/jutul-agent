"""Simulator registry: name → adapter."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import pkgutil
from pathlib import Path

import jutul_agent.simulators as simulators_pkg
from jutul_agent.simulators.base import SimulatorAdapter

_log = logging.getLogger(__name__)


def _discover_external(base: Path) -> dict[str, SimulatorAdapter]:
    """Load SimulatorAdapter instances from ``adapter.py`` files under ``base``.

    Each direct subdirectory of ``base`` that contains an ``adapter.py`` is
    imported in isolation.  Any ``SimulatorAdapter`` instance found in the
    resulting module is collected.  Import errors are logged as warnings and
    skipped so a broken user adapter never prevents the tool from starting.
    """
    registry: dict[str, SimulatorAdapter] = {}
    if not base.is_dir():
        return registry
    for adapter_path in sorted(base.glob("*/adapter.py")):
        sim_name = adapter_path.parent.name
        spec = importlib.util.spec_from_file_location(f"_user_sim_{sim_name}", adapter_path)
        if spec is None or spec.loader is None:
            continue
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            _log.warning("Failed to load user simulator from %s: %s", adapter_path, exc)
            continue
        for value in vars(module).values():
            if isinstance(value, SimulatorAdapter):
                registry[value.name] = value
    return registry


def _discover() -> dict[str, SimulatorAdapter]:
    from jutul_agent.paths import user_simulators_dir

    # User adapters are collected first; built-in adapters override on name conflict
    # so the shipped simulators cannot be accidentally shadowed.
    registry = _discover_external(user_simulators_dir())

    for module_info in pkgutil.iter_modules(simulators_pkg.__path__):
        if module_info.name == "base" or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{simulators_pkg.__name__}.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, SimulatorAdapter):
                if value.name in registry:
                    _log.warning(
                        "User simulator %r shadowed by built-in; rename it to avoid conflict.",
                        value.name,
                    )
                registry[value.name] = value
    return registry


_REGISTRY: dict[str, SimulatorAdapter] = _discover()


def names() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str) -> SimulatorAdapter:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown simulator {name!r}; known: {', '.join(names())}") from exc
