"""Kernel edge cases: bounded eval buffers, capped tool results, discovery guards."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from jutul_agent.juliakernel import JuliaKernel, KernelConfig
from jutul_agent.juliakernel.connection import (
    EVAL_BUFFER_HEAD,
    EVAL_BUFFER_TAIL,
    KernelConnection,
    PendingEval,
)
from jutul_agent.lab.fakes import make_fake_adapter


async def test_eval_output_buffer_keeps_head_and_tail() -> None:
    """A runaway print loop must not grow harness memory without bound."""
    conn = KernelConnection(asyncio.StreamReader(), asyncio.StreamReader())
    try:
        pending = conn.begin_eval(None)
        chunk = b"x" * (64 * 1024)
        total = 0
        for _ in range(12):  # 768 KiB, past the 512 KiB cap
            conn._route_output("stdout", chunk)
            total += len(chunk)
        assert len(pending.out) == EVAL_BUFFER_HEAD + EVAL_BUFFER_TAIL
        assert pending.omitted["stdout"] == total - (EVAL_BUFFER_HEAD + EVAL_BUFFER_TAIL)
        assert not pending.err  # the other stream is untouched
    finally:
        await conn.aclose()


async def test_build_result_notes_omitted_output() -> None:
    kernel = JuliaKernel(KernelConfig())
    pending = PendingEval(exec_id=1, future=asyncio.get_running_loop().create_future())
    pending.out += b"the tail survived"
    pending.omitted["stdout"] = 12345

    result = kernel._build_result("ok", b"42", pending)

    assert "the tail survived" in result.output
    assert "12,345 bytes of stdout" in result.output
    assert "omitted" in result.output


def test_adapter_problem_catches_late_failures(tmp_path: Path) -> None:
    """The registry validates adapters at discovery, not at first-session time."""
    import dataclasses

    from jutul_agent.simulators.registry import _adapter_problem

    adapter = make_fake_adapter(tmp_path)
    # The fake has no julia_env template on disk.
    problem = _adapter_problem(adapter)
    assert problem is not None and "Project.toml" in problem

    env = adapter.julia_env_template_path
    env.mkdir(parents=True, exist_ok=True)
    (env / "Project.toml").write_text("[deps]\n", encoding="utf-8")
    assert _adapter_problem(adapter) is None

    unnamed = dataclasses.replace(adapter, name="  ")
    assert _adapter_problem(unnamed) == "its name is empty"


def test_bundled_adapters_survive_a_broken_subpackage(monkeypatch, capsys) -> None:
    """One bad simulator folder warns and is skipped; the rest keep working."""
    import pkgutil

    from jutul_agent.simulators import registry

    real_iter = pkgutil.iter_modules

    def with_broken(path=None, prefix=""):
        mods = list(real_iter(path, prefix))
        broken = SimpleNamespace(name="brokensim", ispkg=True)
        return [broken, *mods]

    monkeypatch.setattr(registry.pkgutil, "iter_modules", with_broken)
    found = registry._bundled_adapters()

    err = capsys.readouterr().err
    assert "skipping simulator adapter" in err and "brokensim" in err
    assert "jutuldarcy" in found  # the real ones still load
