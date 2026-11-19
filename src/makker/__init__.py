"""makker: specialized scientific agent for AD-enabled simulators on Jutul."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
    """Best-effort running version, robust across install shapes.

    Preference order:
    1. Installed distribution metadata (wheel, ``uv tool install``, editable) —
       the canonical source, derived from git tags by hatch-vcs at build time.
    2. The ``_version.py`` file hatch-vcs bakes into the wheel, in case metadata
       is unavailable (e.g. running from a source tree that was built but not
       installed).
    3. ``"0.0.0+unknown"`` when neither is present (a bare source checkout with no
       build), so callers always get a string.
    """

    try:
        return version("makker")
    except PackageNotFoundError:
        pass
    try:
        from makker._version import __version__ as baked  # type: ignore[import-not-found]

        return baked
    except Exception:
        return "0.0.0+unknown"


__version__ = _resolve_version()
