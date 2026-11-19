"""Tests for the update-check / install-method logic."""

from __future__ import annotations

import json
import time

import pytest

from makker import update_check as u
from makker.update_check import InstallInfo

# ---------------------------------------------------------------------------
# Version comparison.


@pytest.mark.parametrize(
    ("candidate", "current", "expected"),
    [
        ("1.2.0", "1.1.0", True),
        ("1.1.0", "1.1.0", False),
        ("1.0.0", "1.1.0", False),
        ("1.1.0", "1.1.0.dev3+gabc", True),  # a real release beats a dev build of it
        ("not-a-version", "1.0.0", False),  # malformed never nags
    ],
)
def test_is_newer(candidate: str, current: str, expected: bool) -> None:
    assert u.is_newer(candidate, current) is expected


# ---------------------------------------------------------------------------
# Install-method detection + upgrade command.


def _fake_distribution(direct_url: dict | None):
    class _Dist:
        def read_text(self, name: str) -> str | None:
            if name == "direct_url.json" and direct_url is not None:
                return json.dumps(direct_url)
            return None

    return _Dist()


def test_install_info_registry_when_no_direct_url(monkeypatch) -> None:
    monkeypatch.setattr(u, "distribution", lambda _name: _fake_distribution(None))
    assert u.install_info().method == "registry"


def test_install_info_editable(monkeypatch) -> None:
    url = "file:///home/user/makker"
    monkeypatch.setattr(
        u,
        "distribution",
        lambda _name: _fake_distribution({"url": url, "dir_info": {"editable": True}}),
    )
    info = u.install_info()
    assert info.method == "editable"
    assert info.location is not None


def test_install_info_git(monkeypatch) -> None:
    monkeypatch.setattr(
        u,
        "distribution",
        lambda _name: _fake_distribution(
            {"url": "https://github.com/x/y", "vcs_info": {"vcs": "git"}}
        ),
    )
    assert u.install_info().method == "git"


def test_install_info_unknown_when_not_installed(monkeypatch) -> None:
    from importlib.metadata import PackageNotFoundError

    def _raise(_name):
        raise PackageNotFoundError

    monkeypatch.setattr(u, "distribution", _raise)
    assert u.install_info().method == "unknown"


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("editable", "git pull && uv sync"),
        ("git", "uv tool upgrade makker"),
        ("registry", "uv tool upgrade makker"),
    ],
)
def test_upgrade_command(method: str, expected: str) -> None:
    assert u.upgrade_command(InstallInfo(method)) == expected  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Fetching (parsing only; network is monkeypatched).


def test_fetch_pypi_latest_parses_info_version(monkeypatch) -> None:
    monkeypatch.setattr(u, "_fetch_json", lambda _url: {"info": {"version": "2.3.4"}})
    assert u._fetch_pypi_latest() == "2.3.4"


def test_fetch_github_latest_uses_release_tag(monkeypatch) -> None:
    monkeypatch.setattr(u, "_fetch_json", lambda url: {"tag_name": "v0.5.0"})
    assert u._fetch_github_latest() == "0.5.0"


def test_fetch_latest_prefers_pypi_then_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(u, "_fetch_pypi_latest", lambda: None)
    monkeypatch.setattr(u, "_fetch_github_latest", lambda: "9.9.9")
    assert u.fetch_latest_version() == "9.9.9"


# ---------------------------------------------------------------------------
# Cache + pending_update.


def test_cache_roundtrip_and_pending_update(monkeypatch) -> None:
    u._write_cache("9.9.9")
    cached = u._read_cache()
    assert cached is not None and cached[1] == "9.9.9"
    # 9.9.9 is newer than the running dev version, so it's pending.
    assert u.pending_update() == "9.9.9"

    u._write_cache("0.0.0")
    assert u.pending_update() is None


def test_refresh_cache_uses_fresh_cache_without_network(monkeypatch) -> None:
    u._write_cache("1.2.3")

    def _boom() -> str:
        raise AssertionError("should not hit the network when cache is fresh")

    monkeypatch.setattr(u, "fetch_latest_version", _boom)
    assert u.refresh_cache() == "1.2.3"


def test_refresh_cache_force_fetches(monkeypatch) -> None:
    u._write_cache("1.2.3")
    monkeypatch.setattr(u, "fetch_latest_version", lambda: "4.5.6")
    assert u.refresh_cache(force=True) == "4.5.6"
    assert u._read_cache()[1] == "4.5.6"


def test_refresh_cache_refetches_when_stale(monkeypatch) -> None:
    path = u._cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    stale = time.time() - (u._CACHE_TTL_SECONDS + 1)
    path.write_text(json.dumps({"checked_at": stale, "latest": "1.0.0"}), encoding="utf-8")
    monkeypatch.setattr(u, "fetch_latest_version", lambda: "2.0.0")
    assert u.refresh_cache() == "2.0.0"


# ---------------------------------------------------------------------------
# The launch notice.


def test_notify_skips_editable(monkeypatch, capsys) -> None:
    monkeypatch.setattr(u, "install_info", lambda: InstallInfo("editable"))
    monkeypatch.setattr(u, "_refresh_in_background", lambda: None)
    u._write_cache("9.9.9")  # newer is available, but we still skip for dev installs
    u.notify_at_launch()
    assert capsys.readouterr().err == ""


def test_notify_disabled_by_env(monkeypatch, capsys) -> None:
    monkeypatch.setenv(u.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(u, "install_info", lambda: InstallInfo("registry"))
    u._write_cache("9.9.9")
    u.notify_at_launch()
    assert capsys.readouterr().err == ""


def test_notify_prints_when_update_pending(monkeypatch, capsys) -> None:
    monkeypatch.setattr(u, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(u, "_refresh_in_background", lambda: None)
    u._write_cache("9.9.9")
    u.notify_at_launch()
    err = capsys.readouterr().err
    assert "newer makker is available" in err
    assert "9.9.9" in err
    assert "makker upgrade" in err


def test_notify_silent_when_current(monkeypatch, capsys) -> None:
    monkeypatch.setattr(u, "install_info", lambda: InstallInfo("registry"))
    monkeypatch.setattr(u, "_refresh_in_background", lambda: None)
    u._write_cache("0.0.0")  # older than running version
    u.notify_at_launch()
    assert capsys.readouterr().err == ""
