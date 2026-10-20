"""Tell users when a newer jutul-agent is available, and how to upgrade.

The agent is actively developed, so a stale install is a common source of
confusion. This module powers two things:

- a one-line "newer version available" notice at launch (``notify_at_launch``),
  driven entirely off a local cache so launch never blocks on the network; the
  cache is refreshed in the background for the next launch.
- the ``jutul-agent upgrade`` command (see ``interfaces.cli.upgrade``), which
  reuses the install-method detection here to run the right upgrade for how the
  user installed.

Everything degrades to silence: no network, an unpublished package, or a
missing ``packaging`` all just mean "no notice", never an error. Set
``JUTUL_AGENT_NO_UPDATE_CHECK=1`` to turn the whole thing off.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Literal

from jutul_agent import __version__
from jutul_agent.paths import state_home

PACKAGE = "jutul-agent"
# Owner/repo used for the GitHub fallback before the package is on PyPI.
GITHUB_REPO = "SINTEF-agentlab/jutul-agent"
DISABLE_ENV_VAR = "JUTUL_AGENT_NO_UPDATE_CHECK"

# Re-check the network at most this often; every other launch reads the cache.
_CACHE_TTL_SECONDS = 24 * 60 * 60
_HTTP_TIMEOUT_SECONDS = 2.0

InstallMethod = Literal["registry", "git", "editable", "unknown"]


# ---------------------------------------------------------------------------
# Install-method detection (PEP 610 direct_url.json).


@dataclass(frozen=True)
class InstallInfo:
    """How jutul-agent was installed, and where (for editable installs)."""

    method: InstallMethod
    location: Path | None = None  # the source checkout, for editable installs


def install_info() -> InstallInfo:
    """Classify the install via the dist's ``direct_url.json`` (PEP 610).

    - ``editable``: an editable/dev checkout (``uv sync`` in the repo) — its
      ``location`` is the source tree, upgraded with ``git pull``.
    - ``git``: ``uv tool install git+…`` — upgraded by re-resolving the ref.
    - ``registry``: a normal wheel install from PyPI (no ``direct_url.json``).
    - ``unknown``: the package isn't importable as a dist (a bare source run).
    """

    try:
        dist = distribution(PACKAGE)
    except PackageNotFoundError:
        return InstallInfo("unknown")

    raw = dist.read_text("direct_url.json")
    if raw is None:
        # No direct_url.json is what a plain registry (PyPI) wheel install looks like.
        return InstallInfo("registry")
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return InstallInfo("unknown")

    if data.get("dir_info", {}).get("editable"):
        return InstallInfo("editable", _url_to_path(data.get("url")))
    if "vcs_info" in data:
        return InstallInfo("git")
    if "dir_info" in data:
        # A non-editable local path install (``uv tool install /path/to/repo``).
        return InstallInfo("git", _url_to_path(data.get("url")))
    return InstallInfo("registry")


def _url_to_path(url: str | None) -> Path | None:
    if not url or not url.startswith("file:"):
        return None
    from urllib.parse import unquote, urlparse

    parsed = urlparse(url)
    raw = unquote(parsed.path)
    # Windows file URLs come through as ``/C:/...``; strip the leading slash.
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    return Path(raw)


def upgrade_command(info: InstallInfo | None = None) -> str:
    """The shell command that upgrades this install, as a user-facing string."""

    info = info or install_info()
    if info.method == "editable":
        return "git pull && uv sync"
    # Both git and registry tool installs upgrade the same way.
    return f"uv tool upgrade {PACKAGE}"


# ---------------------------------------------------------------------------
# Version comparison.


def is_newer(candidate: str, current: str = __version__) -> bool:
    """Whether ``candidate`` is a strictly newer release than ``current``.

    Uses PEP 440 ordering when ``packaging`` is importable (it ships in our
    dependency tree); falls back to a string compare otherwise. Either way a
    parse failure yields ``False`` so a malformed value never nags the user.
    """

    try:
        from packaging.version import InvalidVersion, Version

        try:
            return Version(candidate) > Version(current)
        except InvalidVersion:
            return False
    except ImportError:  # pragma: no cover - packaging is always present in practice
        return candidate != current and candidate > current


# ---------------------------------------------------------------------------
# Fetching the latest published version.


def _fetch_json(url: str) -> dict | list | None:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _fetch_pypi_latest() -> str | None:
    data = _fetch_json(f"https://pypi.org/pypi/{PACKAGE}/json")
    if isinstance(data, dict):
        version = data.get("info", {}).get("version")
        if isinstance(version, str):
            return version
    return None


def _fetch_github_latest() -> str | None:
    """Latest release tag on GitHub (the fallback before the package is on PyPI)."""

    data = _fetch_json(f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest")
    if isinstance(data, dict):
        tag = data.get("tag_name")
        if isinstance(tag, str):
            return tag.lstrip("v")
    # No published release yet: take the highest semver-looking tag.
    tags = _fetch_json(f"https://api.github.com/repos/{GITHUB_REPO}/tags")
    if isinstance(tags, list):
        names = [t.get("name", "").lstrip("v") for t in tags if isinstance(t, dict)]
        names = [n for n in names if n]
        if names:
            try:
                from packaging.version import Version

                return max(names, key=Version)
            except Exception:
                return names[0]
    return None


def fetch_latest_version() -> str | None:
    """The newest version available, trying PyPI first then GitHub.

    PyPI is the source of truth once published; before then it 404s and the
    GitHub fallback serves the latest release tag. Returns ``None`` if neither
    answers (offline, rate-limited, or nothing published yet).
    """

    return _fetch_pypi_latest() or _fetch_github_latest()


# ---------------------------------------------------------------------------
# Cache (so launch reads disk, not the network).


def _cache_path() -> Path:
    return state_home() / "update-check.json"


def _read_cache() -> tuple[float, str] | None:
    """Return ``(checked_at_epoch, latest_version)`` from the cache, or ``None``."""

    try:
        data = json.loads(_cache_path().read_text(encoding="utf-8"))
        return float(data["checked_at"]), str(data["latest"])
    except Exception:
        return None


def _write_cache(latest: str) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"checked_at": time.time(), "latest": latest}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _cache_is_fresh(checked_at: float) -> bool:
    return (time.time() - checked_at) < _CACHE_TTL_SECONDS


def refresh_cache(*, force: bool = False) -> str | None:
    """Fetch the latest version and store it. Returns the version, or ``None``.

    Skips the network when a fresh cache entry already exists unless ``force``.
    """

    if not force:
        cached = _read_cache()
        if cached is not None and _cache_is_fresh(cached[0]):
            return cached[1]
    latest = fetch_latest_version()
    if latest is not None:
        _write_cache(latest)
    return latest


# ---------------------------------------------------------------------------
# The launch notice.


def is_enabled() -> bool:
    return os.environ.get(DISABLE_ENV_VAR, "").strip().lower() not in {"1", "true", "yes"}


def pending_update() -> str | None:
    """The cached latest version if it's newer than what's running, else ``None``."""

    cached = _read_cache()
    if cached is None:
        return None
    latest = cached[1]
    return latest if is_newer(latest) else None


def _refresh_in_background() -> None:
    thread = threading.Thread(target=refresh_cache, name="update-check", daemon=True)
    thread.start()


def refresh_if_stale() -> None:
    """Refresh the cache in the background when it is missing or a day old."""

    if not is_enabled():
        return
    cached = _read_cache()
    if cached is None or not _cache_is_fresh(cached[0]):
        _refresh_in_background()


def notify_at_launch(stream=None) -> None:
    """Print a one-line upgrade notice when a newer version is known.

    Reads only the local cache, so it never adds launch latency, then kicks off a
    background refresh when the cache is stale so the next launch is current.
    Skipped for editable/dev installs (contributors track git) and when disabled.
    """

    import sys

    stream = stream if stream is not None else sys.stderr
    if not is_enabled():
        return

    info = install_info()
    if info.method in {"editable", "unknown"}:
        # Contributors track git. A source run isn't a managed install, so don't nag.
        return

    latest = pending_update()
    if latest is not None:
        print(
            f"A newer jutul-agent is available: {__version__} -> {latest}. "
            f"Upgrade with `jutul-agent upgrade` (or `{upgrade_command(info)}`).",
            file=stream,
        )

    refresh_if_stale()
