"""Tests for the JuliaKernel backend.

The integration tests need only ``julia`` on PATH; the kernel runs against base
Julia, no instantiated env required (a strict improvement over the old backend,
which needed a built env to test at all).
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import shutil
import signal
import socket
from pathlib import Path

import pytest

from jutul_agent.juliakernel import JuliaKernel, KernelConfig, OutputChunk
from jutul_agent.juliakernel.connection import KernelConnection
from jutul_agent.juliakernel.kernel import JuliaStartupError, _thread_flag, describe_exit

_HAS_JULIA = shutil.which("julia") is not None
needs_julia = pytest.mark.skipif(not _HAS_JULIA, reason="requires `julia` on PATH")


# ---- unit tests (no Julia) -------------------------------------------------


def test_build_launch_assembles_flags(tmp_path: Path) -> None:
    kernel = JuliaKernel(
        KernelConfig(
            julia_executable="julia",
            julia_project=tmp_path / "env",
            sysimage=tmp_path / "sys.so",
            threads="auto",
            env={"DISPLAY": ":7"},
        )
    )
    command, args, env = kernel._build_launch(54321)
    # Julia is launched directly (no xvfb-run wrapper); a display arrives via DISPLAY.
    assert command == "julia"
    assert f"--project={tmp_path / 'env'}" in args
    assert f"--sysimage={tmp_path / 'sys.so'}" in args
    assert "--threads=auto" in args
    assert "54321" in args
    assert any(a.endswith("server.jl") for a in args)
    assert env["DISPLAY"] == ":7"
    assert "JK_TOKEN" in env


def test_thread_flag_always_reserves_an_interactive_thread() -> None:
    """The eval loop must end up alone on the interactive thread (SIGINT routing)."""
    assert _thread_flag(None) == "1,1"
    assert _thread_flag("4") == "4,1"
    assert _thread_flag("auto") == "auto"  # auto already implies 1 interactive
    assert _thread_flag("4,2") == "4,2"  # explicit pools are respected


def test_exit_status_names_the_way_the_process_died() -> None:
    """A death has to be attributable: a crash and an OOM kill need different fixes."""
    assert describe_exit(None) == "exit status unknown"
    assert describe_exit(1) == "exit code 1"
    # Windows encodes the fault in the status; the hex form is the searchable one.
    assert "0xC0000005" in describe_exit(0xC0000005)


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX signals")
def test_a_signalled_death_points_at_its_usual_cause() -> None:
    assert "SIGKILL" in describe_exit(-signal.SIGKILL)
    assert "out-of-memory" in describe_exit(-signal.SIGKILL)
    assert "SIGSEGV" in describe_exit(-signal.SIGSEGV)


async def test_startup_error_for_missing_julia() -> None:
    kernel = JuliaKernel(KernelConfig(julia_executable="definitely-not-julia-zzz"))
    with pytest.raises(JuliaStartupError):
        await kernel.__aenter__()
    # _spawn tore down what it opened before raising (no leaked listener socket).
    assert kernel._listener is None


class _Wire:
    """A KernelConnection wired to an in-process socket playing the Julia side."""

    def __init__(self) -> None:
        self.log = io.BytesIO()
        self.conn: KernelConnection = None  # type: ignore[assignment]
        self.writer: asyncio.StreamWriter = None  # type: ignore[assignment]

    async def __aenter__(self) -> _Wire:
        self.out_pipe, self.err_pipe = asyncio.StreamReader(), asyncio.StreamReader()
        self.conn = KernelConnection(self.out_pipe, self.err_pipe, stderr_fh=self.log)
        ours, theirs = socket.socketpair()
        await self.conn.attach_control(ours)
        _, self.writer = await asyncio.open_connection(sock=theirs)
        return self

    async def __aexit__(self, *exc: object) -> None:
        with contextlib.suppress(Exception):
            self.writer.close()
        await self.conn.aclose()

    async def send(self, raw: bytes) -> None:
        self.writer.write(raw)
        await self.writer.drain()


async def test_connection_routes_frames_to_the_pending_eval() -> None:
    """OUT frames fill the in-flight eval's buffers/sink; RES resolves its future."""
    async with _Wire() as wire:
        await wire.send(b"RDY deadbeef 0\n")
        assert await asyncio.wait_for(wire.conn.ready_token(), 5) == "deadbeef"

        chunks: list[OutputChunk] = []
        pending = wire.conn.begin_eval(chunks.append)
        await wire.send(b"OUT stdout 5\nhelloOUT stderr 4\noopsRES 1 ok 1\n2")
        assert await asyncio.wait_for(pending.future, 5) == ("ok", b"2")
        wire.conn.end_eval(pending)
        assert bytes(pending.out) == b"hello"
        assert bytes(pending.err) == b"oops"
        assert [(c.stream, c.text) for c in chunks] == [("stdout", "hello"), ("stderr", "oops")]

        # Output with no eval in flight lands in the log, not in anyone's result.
        await wire.send(b"OUT stdout 4\nlate")
        async with asyncio.timeout(5):
            while b"late" not in wire.log.getvalue():
                await asyncio.sleep(0.01)


async def test_streamed_chunks_keep_split_utf8_characters_whole() -> None:
    """A pipe flush can split a multi-byte character (e.g. the box-drawing
    corner of a results table) across OUT frames; the live sink must see the
    completed character, not replacement marks at the split."""
    async with _Wire() as wire:
        await wire.send(b"RDY t 0\n")
        chunks: list[OutputChunk] = []
        pending = wire.conn.begin_eval(chunks.append)

        corner = "\u2500\u256f".encode()  # "─╯", three bytes each
        head, tail = corner[:4], corner[4:]  # cut inside the second character
        await wire.send(b"OUT stdout %d\n" % len(head) + head)
        await wire.send(b"OUT stdout %d\n" % len(tail) + tail)
        await wire.send(b"RES 1 ok 0\n")
        await asyncio.wait_for(pending.future, 5)
        wire.conn.end_eval(pending)

        text = "".join(chunk.text for chunk in chunks)
        assert text == "\u2500\u256f"
        assert "\ufffd" not in text
        assert bytes(pending.out) == corner


async def test_connection_drops_a_stale_result() -> None:
    """The RES of an abandoned eval must not resolve the next eval's future."""
    async with _Wire() as wire:
        await wire.send(b"RDY t 0\n")
        abandoned = wire.conn.begin_eval(None)
        wire.conn.end_eval(abandoned)  # cancelled and recovered from
        await wire.send(b"RES 1 int 0\n")
        pending = wire.conn.begin_eval(None)
        await wire.send(b"RES 2 ok 1\n5")
        assert await asyncio.wait_for(pending.future, 5) == ("ok", b"5")
        assert not abandoned.future.done()


async def test_connection_poisons_waiters_when_the_socket_closes() -> None:
    """A blocked eval wakes with a clean error instead of hanging forever."""
    async with _Wire() as wire:
        await wire.send(b"RDY t 0\n")
        pending = wire.conn.begin_eval(None)
        wire.writer.close()
        with pytest.raises(RuntimeError, match="exited"):
            await asyncio.wait_for(pending.future, 5)
        assert wire.conn.closed.is_set()
        # Evals started after death fail immediately, with the same message.
        late = wire.conn.begin_eval(None)
        with pytest.raises(RuntimeError, match="exited"):
            await late.future


# ---- integration tests (need Julia) ----------------------------------------


@pytest.mark.integration
@needs_julia
async def test_relative_include_resolves_in_workspace(tmp_path: Path) -> None:
    # A file written to the workspace must be `include`-able by its relative name;
    # without the rewrite, Julia resolves it next to the kernel's own source file.
    (tmp_path / "snippet.jl").write_text('println("loaded snippet")\n')
    async with JuliaKernel(KernelConfig(cwd=tmp_path)) as k:
        r = await k.eval('include("snippet.jl")')
        assert r.error is None, r.error
        assert "loaded snippet" in r.output


@pytest.mark.integration
@needs_julia
async def test_eval_persistence_streaming_and_error() -> None:
    async with JuliaKernel(KernelConfig()) as k:
        r = await k.eval("1 + 1")
        assert r.error is None
        assert r.output == "2"
        assert r.value_repr == "2"

        assert (await k.eval("x = 41; x + 1")).output == "42"  # state persists

        r = await k.eval('println("hello"); nothing')
        assert r.error is None
        assert r.output == "hello"  # no "nothing" noise

        r = await k.eval("sqrt(-1)")
        assert r.error is not None
        assert "DomainError" in r.error

        chunks: list[OutputChunk] = []
        r = await k.eval(
            'for i in 1:4; println("tick $i"); flush(stdout); sleep(0.15); end',
            on_chunk=chunks.append,
        )
        assert chunks, "expected live output chunks"
        assert any("tick" in c.text for c in chunks if c.stream == "stdout")
        assert r.output == "tick 1\ntick 2\ntick 3\ntick 4"


@pytest.mark.integration
@needs_julia
async def test_huge_error_frame_does_not_kill_session() -> None:
    """A giant error payload comes back as a normal error, not a false death.

    Frames are length-prefixed so size alone can't break the transport; the
    server additionally caps the result payload so a pathological error message
    returns bounded and readable.
    """
    async with JuliaKernel(KernelConfig()) as k:
        r = await k.eval('error("X" ^ 200_000)')
        assert r.error is not None
        assert k.running, "the session must survive a huge error frame"
        assert "output truncated" in r.error  # server-side cap kept the frame small
        assert len(r.error) < 80 * 1024
        assert (await k.eval("6 * 7")).output == "42"  # still usable, no reset needed


@pytest.mark.integration
@needs_julia
async def test_error_stacktrace_is_compact() -> None:
    """Errors use Julia's own REPL backtrace with the type limiter.

    ``format_error`` passes the ``:stacktrace_types_limited`` IOContext key, so a
    frame's huge specialized argument types collapse to ``{…}`` while the call chain
    and small types are kept.
    """
    async with JuliaKernel(KernelConfig()) as k:
        # A nested call so there are real frames with argument types.
        r = await k.eval("f(x) = sqrt(x); g(x) = f(x); g(-1.0)")
        assert r.error is not None
        assert "DomainError" in r.error
        assert "Stacktrace:" in r.error
        assert " @ " in r.error  # frames carry `@ file:line`
        assert len(r.error) < 8 * 1024  # compact, not a type-signature dump


@pytest.mark.integration
@needs_julia
async def test_error_keeps_output_printed_before_the_throw() -> None:
    """Output the user code printed before it threw is kept, not discarded."""
    async with JuliaKernel(KernelConfig()) as k:
        r = await k.eval('println("progress: step 1"); error("boom")')
        assert r.error is not None
        assert "boom" in r.error
        # The breadcrumb the agent needs survives in both the assembled output
        # and the structured stdout.
        assert "progress: step 1" in r.output
        assert "progress: step 1" in r.stdout


@pytest.mark.integration
@needs_julia
async def test_undisplayable_value_reports_its_type() -> None:
    """A value whose show/string both throw yields its type, not '<unprintable value>'."""
    async with JuliaKernel(KernelConfig()) as k:
        await k.eval(
            "struct _Unshowable end\n"
            'Base.show(io::IO, ::MIME"text/plain", ::_Unshowable) = error("no")\n'
            'Base.show(io::IO, ::_Unshowable) = error("no")'
        )
        r = await k.eval("_Unshowable()")
        assert r.error is None
        assert "_Unshowable" in r.output
        assert "cannot be displayed" in r.output


@pytest.mark.integration
@needs_julia
async def test_interrupt_survives_and_recovers() -> None:
    async with JuliaKernel(KernelConfig()) as k:
        task = asyncio.create_task(
            k.eval("for i in 1:200; println(i); flush(stdout); sleep(0.05); end")
        )
        await asyncio.sleep(1.0)
        await k.interrupt()
        r = await task
        assert r.interrupted
        assert k.running  # the server survived the interrupt
        assert (await k.eval("21 * 2")).output == "42"  # and still works


@pytest.mark.integration
@needs_julia
async def test_cancelled_eval_interrupts_and_preserves_state() -> None:
    """Cancelling a running eval interrupts it and keeps the session + state alive.

    Instead of restarting Julia (losing packages and variables), the kernel
    soft-interrupts the eval and drains its result frame, so the process stays in
    protocol sync. POSIX uses SIGINT; Windows uses CTRL_BREAK to the kernel's
    process group, which Julia (exit_on_sigint=false) turns into the same
    catchable InterruptException.
    """
    async with JuliaKernel(KernelConfig()) as k:
        await k.eval("kept = 123")  # state that must survive the cancel
        task = asyncio.create_task(k.eval("for i in 1:1000; sleep(0.1); end; 1"))
        await asyncio.sleep(0.3)  # let it enter the loop
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert k.running, "the session must survive a cancelled eval"
        assert k.cancel_preserved_state, "state should be preserved, not restarted"
        # No protocol desync (results align with their code) and state is intact.
        # Bounded waits: recovery must come from the interrupt, not from the
        # 100-second loop quietly running to completion.
        assert (await asyncio.wait_for(k.eval("kept + 1"), timeout=15)).output == "124"
        assert (await asyncio.wait_for(k.eval("2 + 2"), timeout=15)).output == "4"


@pytest.mark.integration
@needs_julia
async def test_reset_and_restart_clear_state() -> None:
    async with JuliaKernel(KernelConfig()) as k:
        await k.eval("y = 7")
        await k.reset()
        assert (await k.eval("@isdefined(y)")).output == "false"

        await k.eval("z = 9")
        await k.restart()
        assert k.running
        assert (await k.eval("@isdefined(z)")).output == "false"


@pytest.mark.integration
@needs_julia
async def test_ccall_output_is_captured() -> None:
    """Output written by C code (fd 1, bypassing Julia's IO) is still captured.

    The server redirects the fds themselves (dup2), so printf from a solver's C
    or Fortran dependency lands in the eval's output like any println.
    """
    async with JuliaKernel(KernelConfig()) as k:
        r = await k.eval(
            'ccall(:printf, Cint, (Cstring,), "from-c\\n"); Libc.flush_cstdio(); nothing'
        )
        assert r.error is None
        assert "from-c" in r.output


@pytest.mark.integration
@needs_julia
async def test_interrupt_during_heavy_printing_repeatedly() -> None:
    """SIGINT lands in the eval; never swallowed by the output pumps; under load.

    Regression test for interrupt misdelivery: the pumps live on the default
    thread pool, away from the eval loop's interactive thread, so an interrupt
    arriving mid-flood must reliably stop the eval and leave the session in
    protocol sync. Repeated because misdelivery was a race, not a certainty.
    """
    async with JuliaKernel(KernelConfig()) as k:
        for i in range(3):
            task = asyncio.create_task(k.eval("for i in 1:10_000_000; println(i); end"))
            await asyncio.sleep(0.4)
            await k.interrupt()
            r = await asyncio.wait_for(task, timeout=15)
            assert r.interrupted, f"iteration {i}: expected an interrupt result"
            clean = await asyncio.wait_for(k.eval('"clean"'), timeout=15)
            assert clean.value_repr == '"clean"', f"iteration {i}: protocol out of sync"


@pytest.mark.integration
@needs_julia
async def test_background_task_output_stays_out_of_the_result() -> None:
    """Output printed by a task after its eval returned isn't in that result."""
    async with JuliaKernel(KernelConfig()) as k:
        r = await k.eval('@async (sleep(0.6); println("LATE")); nothing')
        assert "LATE" not in r.output
        await asyncio.sleep(1.0)  # the stray print lands while idle, into the log
        assert (await k.eval("1 + 1")).value_repr == "2"


@pytest.mark.integration
@needs_julia
async def test_killed_process_surfaces_as_error_not_a_hang(tmp_path: Path) -> None:
    """If the process dies mid-eval, the blocked eval wakes with a clean error.

    The connection poisons the pending eval's future when the control socket or
    the process pipes close, so the await raises instead of hanging forever. The
    error names the exit status and points at the log, because "the kernel
    exited" alone leaves the caller guessing between a crash and a kill.
    """
    log = tmp_path / "julia.log"
    async with JuliaKernel(KernelConfig(stderr_file=log)) as k:
        task = asyncio.create_task(k.eval("sleep(60)"))
        await asyncio.sleep(0.5)
        assert k._proc is not None
        k._proc.kill()  # hard kill mid-eval, not a survivable SIGINT
        with pytest.raises(RuntimeError) as excinfo:
            await task
        assert not k.running
        message = str(excinfo.value)
        assert "exited unexpectedly" in message
        assert ("SIGKILL" in message) or ("exit code" in message)  # Windows has no signals
        assert str(log) in message

        # The next call is the one a user is most likely to see, since a death
        # often lands under a tool call they have already moved on from; it has
        # to explain itself too, not just report that nothing is running.
        with pytest.raises(RuntimeError) as later:
            await k.eval("1 + 1")
        assert ("SIGKILL" in str(later.value)) or ("exit code" in str(later.value))
