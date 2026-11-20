"""Tests for the kernel compute-thread policy."""

from __future__ import annotations

import os
from unittest import mock

from makker.julia.threads import (
    HYPRE_THREADS_ENV_VAR,
    THREADS_ENV_VAR,
    blas_thread_env,
    resolve_compute_threads,
    resolve_hypre_threads,
)


def _with_env(value: str | None):
    env = dict(os.environ)
    env.pop(THREADS_ENV_VAR, None)
    if value is not None:
        env[THREADS_ENV_VAR] = value
    return mock.patch.dict(os.environ, env, clear=True)


def _physical(n: int):
    return mock.patch("makker.julia.threads._physical_cores", return_value=n)


def test_default_is_physical_cores_minus_one():
    with _with_env(None), _physical(6):
        assert resolve_compute_threads() == 5


def test_default_clamped_to_one_on_single_core():
    with _with_env(None), _physical(1):
        assert resolve_compute_threads() == 1


def test_auto_env_is_logical_cpu_count():
    with _with_env("auto"), mock.patch("os.cpu_count", return_value=6):
        assert resolve_compute_threads() == 6


def test_explicit_env_integer_overrides_default():
    with _with_env("3"), _physical(32):
        assert resolve_compute_threads() == 3


def test_unparseable_env_falls_back_to_default():
    with _with_env("lots"), _physical(8):
        assert resolve_compute_threads() == 7


def test_non_positive_clamped_to_one():
    with _with_env("0"):
        assert resolve_compute_threads() == 1


def test_cli_value_takes_precedence_over_env():
    with _with_env("4"), _physical(32):
        assert resolve_compute_threads("2") == 2


def test_cli_auto_overrides_env():
    with _with_env("2"), mock.patch("os.cpu_count", return_value=16):
        assert resolve_compute_threads("auto") == 16


def test_blank_cli_defers_to_env():
    with _with_env("5"), _physical(32):
        assert resolve_compute_threads("  ") == 5


def test_blas_left_alone_when_single_threaded():
    with _with_env(None):
        assert blas_thread_env(1) == {}


def test_blas_pinned_when_threading():
    env = dict(os.environ)
    env.pop("OPENBLAS_NUM_THREADS", None)
    with mock.patch.dict(os.environ, env, clear=True):
        assert blas_thread_env(8) == {"OPENBLAS_NUM_THREADS": "1"}


def test_blas_respects_user_choice():
    with mock.patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "4"}):
        assert blas_thread_env(8) == {}


def _without_hypre_env():
    env = dict(os.environ)
    env.pop(HYPRE_THREADS_ENV_VAR, None)
    return mock.patch.dict(os.environ, env, clear=True)


def test_hypre_default_is_physical_minus_one():
    with _without_hypre_env(), _physical(6):
        assert resolve_hypre_threads() == 5


def test_hypre_capped_at_eight():
    with _without_hypre_env(), _physical(32):
        assert resolve_hypre_threads() == 8


def test_hypre_floor_of_one():
    with _without_hypre_env(), _physical(1):
        assert resolve_hypre_threads() == 1


def test_hypre_env_override():
    with mock.patch.dict(os.environ, {HYPRE_THREADS_ENV_VAR: "3"}), _physical(32):
        assert resolve_hypre_threads() == 3
