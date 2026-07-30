"""Simulator registry: discovers adapters from packages, in-tree and installed.

A simulator's adapter is found two ways, so neither this file nor any other core
file needs editing to add one:

- **Bundled**: each subpackage under ``jutul_agent.simulators`` with an
  ``adapter.py`` defining a module-level :class:`SimulatorAdapter`. Adding a
  bundled simulator is just adding its folder.
- **Installed**: any installed package that publishes a ``SimulatorAdapter``
  under the ``jutul_agent.simulators`` entry-point group. This is how a separate
  project adds its own simulator without forking jutul-agent (see
  docs/extending-for-your-application.md). An installed adapter overrides a
  bundled one of the same name.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import pkgutil
import sys
from functools import cache

from jutul_agent.simulators.base import SimulatorAdapter

# Subpackages under ``simulators`` that are not simulators themselves.
_NOT_SIMULATORS = frozenset({"shared_skills"})
# Entry-point group an installed package publishes a SimulatorAdapter under.
SIMULATOR_ENTRY_POINT_GROUP = "jutul_agent.simulators"


def _adapter_in(module: object) -> SimulatorAdapter | None:
    return next(
        (value for value in vars(module).values() if isinstance(value, SimulatorAdapter)),
        None,
    )


def _bundled_adapters() -> dict[str, SimulatorAdapter]:
    """Adapters from the in-tree simulator subpackages.

    A broken subpackage is warned about and skipped, never fatal: the registry
    feeds the CLI's argument parser, so one bad adapter folder must not take
    down every command including ``doctor``.
    """
    import jutul_agent.simulators as simulators_pkg

    found: dict[str, SimulatorAdapter] = {}
    for info in pkgutil.iter_modules(simulators_pkg.__path__):
        if not info.ispkg or info.name.startswith("_") or info.name in _NOT_SIMULATORS:
            continue
        qualified = f"{simulators_pkg.__name__}.{info.name}"
        try:
            module = importlib.import_module(qualified)
            # Prefer the adapter the subpackage exports; fall back to its adapter.py
            # so a folder with only an adapter.py is enough.
            adapter = _adapter_in(module)
            if adapter is None:
                try:
                    adapter = _adapter_in(importlib.import_module(f"{qualified}.adapter"))
                except ModuleNotFoundError:
                    adapter = None
        except Exception as exc:
            _warn_skipped(qualified, exc)
            continue
        if adapter is not None:
            problem = _adapter_problem(adapter)
            if problem is not None:
                _warn_skipped(qualified, problem)
                continue
            found[adapter.name] = adapter
    return found


def _adapter_problem(adapter: SimulatorAdapter) -> str | None:
    """Why an adapter cannot serve sessions, or ``None`` when it is usable.

    Catches the mistakes a third-party adapter fails *late* on otherwise: a
    blank name (undiscoverable) or a missing Julia env template (bootstrap
    error on first session).
    """
    if not str(adapter.name or "").strip():
        return "its name is empty"
    template = adapter.julia_env_template_path
    if template is not None and not (template / "Project.toml").exists():
        return f"its Julia env template has no Project.toml ({template})"
    return None


def _warn_skipped(source: str, reason: object) -> None:
    print(f"warning: skipping simulator adapter from {source}: {reason}", file=sys.stderr)


def _installed_adapters() -> dict[str, SimulatorAdapter]:
    """Adapters published by installed packages under the entry-point group.

    Each entry point resolves to a ``SimulatorAdapter`` or a zero-argument
    callable returning one. A broken entry point is skipped rather than failing
    the whole registry.
    """
    found: dict[str, SimulatorAdapter] = {}
    try:
        entry_points = importlib_metadata.entry_points(group=SIMULATOR_ENTRY_POINT_GROUP)
    except Exception:
        return found
    for entry_point in entry_points:
        try:
            loaded = entry_point.load()
            adapter = (
                loaded()
                if callable(loaded) and not isinstance(loaded, SimulatorAdapter)
                else loaded
            )
        except Exception as exc:
            _warn_skipped(entry_point.name, exc)
            continue
        if not isinstance(adapter, SimulatorAdapter):
            _warn_skipped(entry_point.name, f"resolved to {type(adapter).__name__}")
            continue
        problem = _adapter_problem(adapter)
        if problem is not None:
            _warn_skipped(entry_point.name, problem)
            continue
        found[adapter.name] = adapter
    return found


@cache
def _registry() -> dict[str, SimulatorAdapter]:
    """Collect adapters from the bundled subpackages and installed packages (cached).

    Installed adapters are applied last, so a separate project can override a
    bundled simulator (e.g. a customised JutulDarcy) by publishing one with the
    same name.
    """
    return {**_bundled_adapters(), **_installed_adapters()}


def names() -> list[str]:
    return sorted(_registry())


def get(name: str) -> SimulatorAdapter:
    try:
        return _registry()[name]
    except KeyError as exc:
        raise KeyError(f"unknown simulator {name!r}; known: {', '.join(names())}") from exc
