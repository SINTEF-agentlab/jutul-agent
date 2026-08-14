"""Tests for the precompile ceiling.

What this guards is a failure with no error attached: a machine that runs out of
memory while precompiling dies with a native fault, and every one of these tests
exists because the count that caused it came from Julia's core-based default.
"""

from __future__ import annotations

import os

import pytest

from jutul_agent.julia import precompile


def test_the_ceiling_is_judged_by_memory_not_by_cores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(precompile, "_total_memory_bytes", lambda: 16 * 2**30)
    monkeypatch.setattr(precompile.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(precompile.os, "name", "posix")

    # 16 GiB at 4 GiB a worker, not the 16 a 64-core machine would otherwise get.
    assert precompile.precompile_task_limit() == 4


def test_the_ceiling_never_raises_julias_own_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ceiling, not a setting: memory to spare must not add parallelism."""
    monkeypatch.setattr(precompile, "_total_memory_bytes", lambda: 512 * 2**30)
    monkeypatch.setattr(precompile.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(precompile.os, "name", "posix")

    assert precompile.precompile_task_limit() == 5  # Julia's own CPU_THREADS + 1


def test_windows_counts_the_way_windows_julia_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(precompile, "_total_memory_bytes", lambda: 512 * 2**30)
    monkeypatch.setattr(precompile.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(precompile.os, "name", "nt")

    assert precompile.precompile_task_limit() == 5  # CPU_THREADS/2 + 1


def test_a_machine_too_small_for_one_worker_still_gets_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(precompile, "_total_memory_bytes", lambda: 2 * 2**30)

    assert precompile.precompile_task_limit() == 1


def test_without_a_memory_reading_julias_default_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guessing at the count blind would be worse than Julia's own guess."""
    monkeypatch.setattr(precompile, "_total_memory_bytes", lambda: None)
    monkeypatch.delenv(precompile.PRECOMPILE_TASKS_ENV_VAR, raising=False)

    assert precompile.precompile_task_limit() is None
    assert precompile.PRECOMPILE_TASKS_ENV_VAR not in precompile.julia_environment()


def test_the_ceiling_travels_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """It has to be inherited, not passed: the process it matters most for is the
    one PackageCompiler launches from inside ``create_sysimage``."""
    monkeypatch.setattr(precompile, "_total_memory_bytes", lambda: 8 * 2**30)
    # Pinned, or the ceiling on a one-core runner is Julia's default, not memory's.
    monkeypatch.setattr(precompile.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(precompile.os, "name", "posix")
    monkeypatch.delenv(precompile.PRECOMPILE_TASKS_ENV_VAR, raising=False)

    assert precompile.julia_environment()[precompile.PRECOMPILE_TASKS_ENV_VAR] == "2"


def test_an_operator_who_set_the_variable_keeps_their_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A default for machines nobody tuned, not a policy over one who did."""
    monkeypatch.setattr(precompile, "_total_memory_bytes", lambda: 8 * 2**30)
    monkeypatch.setenv(precompile.PRECOMPILE_TASKS_ENV_VAR, "1")

    assert precompile.julia_environment()[precompile.PRECOMPILE_TASKS_ENV_VAR] == "1"


def test_the_real_machine_reports_a_plausible_amount_of_memory() -> None:
    """The reading is the whole feature: an install where it silently returns
    ``None`` gets Julia's core-based default and none of this applies. It is read
    from the OS rather than psutil for exactly that reason -- psutil arrives only
    with the ``eval`` extra, so on an ordinary install it is not there."""
    total = precompile._total_memory_bytes()

    assert total is not None, "no memory reading on this platform; the ceiling is a no-op"
    assert total > 2**30  # any machine that can build an image has over a GiB


@pytest.mark.skipif(not hasattr(os, "sysconf"), reason="POSIX sysconf")
def test_a_reading_that_fails_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform we cannot read must cost the ceiling, never the build."""

    def _refuse(_name: str) -> int:
        raise OSError("no such configuration parameter")

    monkeypatch.setattr(precompile.os, "sysconf", _refuse)
    monkeypatch.setattr(precompile.os, "name", "posix")

    assert precompile._total_memory_bytes() is None


def test_windows_is_read_through_its_own_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The branch is not an optimisation: ``os.sysconf`` does not exist on Windows,
    so reaching it there would be an ``AttributeError`` rather than a fallback."""
    monkeypatch.setattr(precompile.os, "name", "nt")
    monkeypatch.setattr(precompile, "_windows_total_memory_bytes", lambda: 7 * 2**30)

    assert precompile._total_memory_bytes() == 7 * 2**30
