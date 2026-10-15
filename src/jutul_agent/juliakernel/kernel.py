"""A supervised, persistent Julia runtime with live output and interrupt.

``JuliaKernel`` owns one Julia subprocess and talks to it over a single
loopback-TCP control connection carrying length-prefixed frames: code out,
live output and one authoritative ``ok``/``err``/``int`` result back per eval
(see :mod:`.connection` for the framing). Output is captured *inside* the Julia
process at the fd level, so an eval's result frame arrives only after all of
its output; completion is one event, ordered by TCP. Evaluation runs
in-process in that Julia process, so a ``Distributed`` worker pool launched
from user code uses it the normal way (the process is the cluster master). The
kernel is the supervisor: it spawns, interrupts (an ``INT`` frame, then SIGINT
on POSIX or CTRL_BREAK on Windows), and respawns the process, so a reset never
takes the surrounding Python session down with it. It counts the processes it
has started (``generation``), records why the latest one replaced the one
before (``last_restart``), and tells whoever asked (``on_restart``), so that what
set the last process up can set the new one up too, before other evals reach it.
The package depends only on the standard library.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import os
import secrets
import signal
import socket
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Self

from .config import KernelConfig
from .connection import KernelConnection, KernelDied, PendingEval
from .result import EvalResult, OnChunk
from .text import render_terminal_output

_SERVER_JL = Path(__file__).resolve().parent / "server.jl"
# A cancelled eval is first interrupted through the server, which wakes a waiting
# eval at once; one still running after this long is computing, and gets SIGINT.
_INTERRUPT_SIGNAL_DELAY = 1.0
# After interrupting a cancelled eval we wait this long for its result frame;
# past this we treat it as wedged and restart.
_INTERRUPT_DRAIN_TIMEOUT = 10.0
# The pipes usually reach EOF a moment before the process is reaped, so allow the
# exit status a short wait rather than reporting it as unknown.
_EXIT_STATUS_TIMEOUT = 2.0
# How much of Julia's own last output to quote back; a crash banner and the top of
# its backtrace fit, a whole solver log does not.
_DEATH_TAIL_CHARS = 600
# Set while the restart callbacks run, so every task they start carries it: their
# evals set the fresh process up, and so are the ones that do not wait for that.
_SETTING_UP: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "julia_kernel_setting_up", default=False
)


class JuliaStartupError(RuntimeError):
    """The Julia process died or never connected before the kernel was ready.

    Carries the diagnostic context a bare transport error throws away: which
    Julia and project were used, the tail of Julia's own stderr (the actual
    error, usually a package-load failure), and where the full log lives.
    """

    def __init__(
        self,
        summary: str,
        *,
        julia_executable: str,
        julia_project: Path | None,
        stderr_tail: str = "",
        log_file: Path | None = None,
    ) -> None:
        self.summary = summary
        self.julia_executable = julia_executable
        self.julia_project = julia_project
        self.stderr_tail = stderr_tail
        self.log_file = log_file
        super().__init__(self.format())

    def format(self) -> str:
        lines = [
            f"Julia failed to start before the kernel was ready: {self.summary}",
            f"  julia:         {self.julia_executable}",
            f"  julia project: {self.julia_project if self.julia_project else '(default)'}",
        ]
        if self.stderr_tail.strip():
            lines.append("  Julia said:")
            lines.extend(f"    {line}" for line in self.stderr_tail.strip().splitlines()[-40:])
        if self.log_file is not None:
            lines.append(f"  full log:      {self.log_file}")
        return "\n".join(lines)


@dataclass(frozen=True)
class KernelRestart:
    """When a fresh Julia process replaced a running one, and why."""

    at: float  # wall-clock time (``time.time()``) the fresh process was ready
    reason: str  # completes "Julia restarted because ..."


# Told of each restart; may return an awaitable that sets the fresh process up.
RestartCallback = Callable[[KernelRestart], Awaitable[object] | None]


class JuliaKernel:
    """A ``JuliaSession``-compatible persistent Julia runtime."""

    def __init__(self, config: KernelConfig | None = None) -> None:
        self._config = config or KernelConfig()
        self._proc: asyncio.subprocess.Process | None = None
        self._listener: socket.socket | None = None
        self._conn: KernelConnection | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._stderr_fh: IO[bytes] | None = None
        self._token = ""
        self._lock = asyncio.Lock()
        self._evals = 0  # running or waiting for the lock
        self._cancel_preserved_state = True
        self._generation = 0
        self._last_restart: KernelRestart | None = None
        self._restart_callbacks: list[RestartCallback] = []
        # What the callbacks are doing to set the fresh process up; other evals wait.
        self._setups: list[asyncio.Future[object]] = []

    # ---- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> Self:
        await self._spawn()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self._teardown(graceful=True)

    @property
    def running(self) -> bool:
        return (
            self._proc is not None
            and self._proc.returncode is None
            and self._conn is not None
            and not self._conn.closed.is_set()
        )

    @property
    def busy(self) -> bool:
        """Whether an eval is running or waiting for its turn.

        Work nobody waits on (a plot re-fit) checks this to run only on an idle
        kernel, so that a deadline it sets counts its own time, not the queue's.
        """
        return self._evals > 0

    @property
    def generation(self) -> int:
        """How many Julia processes this kernel has started; a change means the
        session's state (variables, packages, live plots) was lost."""
        return self._generation

    @property
    def last_restart(self) -> KernelRestart | None:
        """Why and when the current process replaced the one before, if it did."""
        return self._last_restart

    def on_restart(self, callback: RestartCallback) -> Callable[[], None]:
        """Call ``callback`` each time a fresh process replaces the running one.

        Whether ``reset``, ``restart`` or a cancelled eval's recovery started it,
        the callback gets the ``KernelRestart`` once the new process is ready,
        before the call that restarted it returns. It must not block: work for the
        new process goes in a task. To set the new process up before anything else
        runs on it, return an awaitable that finishes once that is done: other
        evals wait for it, even those queued before the restart, while the evals
        of the tasks the callback started go ahead.
        Returns a function that removes the callback.
        """
        self._restart_callbacks.append(callback)

        def remove() -> None:
            with contextlib.suppress(ValueError):
                self._restart_callbacks.remove(callback)

        return remove

    async def reset(self) -> EvalResult:
        """Respawn a fresh Julia process (cooperative; never raises)."""
        with contextlib.suppress(BaseException):
            await self._teardown(graceful=True)
        await self._spawn()
        self._restarted("it was reset")
        return EvalResult(output="Julia restarted with a fresh session.")

    async def restart(self, reason: str = "it was restarted") -> None:
        """Force the process down (SIGKILL) and respawn, ignoring responsiveness.

        ``reason`` completes "Julia restarted because ..." in ``last_restart``.
        """
        await self._teardown(graceful=False)
        await self._spawn()
        self._restarted(reason)

    def _restarted(self, reason: str) -> None:
        """Record why the fresh process replaced the last one, and tell the callbacks."""
        restart = KernelRestart(time.time(), reason)
        self._last_restart = restart
        token = _SETTING_UP.set(True)
        try:
            for callback in list(self._restart_callbacks):
                try:
                    setup = callback(restart)
                    if setup is not None:
                        self._setups.append(asyncio.ensure_future(setup))
                except Exception as exc:
                    # The restart has happened either way; a broken callback is
                    # reported the way asyncio reports one, and the rest still hear.
                    asyncio.get_running_loop().call_exception_handler(
                        {"message": "a Julia restart callback failed", "exception": exc}
                    )
        finally:
            _SETTING_UP.reset(token)

    def _setting_up(self) -> bool:
        """Whether a fresh process is being set up, by work this task is not part of."""
        if _SETTING_UP.get():
            return False
        self._setups = [setup for setup in self._setups if not setup.done()]
        return bool(self._setups)

    async def interrupt(self) -> bool:
        """Try to soft-interrupt the running eval; return whether a signal was sent.

        The eval cancels as an ``InterruptException`` while the session and its
        loaded state survive. On POSIX that's SIGINT to the kernel's own process
        group (``start_new_session`` isolated it from ours). On Windows there is no
        per-process SIGINT, but ``CTRL_BREAK_EVENT`` to the kernel's process group
        (created via ``CREATE_NEW_PROCESS_GROUP``) reaches Julia's console handler,
        which — with ``exit_on_sigint(false)``, set in server.jl — delivers the same
        catchable ``InterruptException`` rather than killing the process. Both stay
        off our own process group, so the agent is never hit.

        Returns ``False`` only if no signal could be sent (e.g. no console to
        target on Windows); the caller then restarts to stop a runaway eval.
        """
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return False
        with contextlib.suppress(ProcessLookupError, OSError, NotImplementedError, ValueError):
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # The whole process group, not just the master: Distributed
                # workers an ensemble spawned must see the interrupt too, or a
                # compute-bound worker outlives the drain window and forces a
                # full restart. ``start_new_session`` made the kernel its own
                # group leader, so pgid == pid.
                os.killpg(proc.pid, signal.SIGINT)
            return True
        return False

    # ---- evaluation --------------------------------------------------------

    async def eval(self, code: str, on_chunk: OnChunk | None = None) -> EvalResult:
        """Evaluate ``code`` in the persistent session.

        ``on_chunk`` (optional) receives :class:`OutputChunk` fragments live as
        the eval produces them; the returned :class:`EvalResult` carries the full
        cleaned output plus the structured value/error.
        """
        if self._proc is None:
            raise RuntimeError("JuliaKernel must be used inside an `async with` block")
        self._evals += 1
        try:
            while True:
                while self._setting_up():
                    await asyncio.wait(self._setups)
                result = await self._eval_locked(code, on_chunk)
                if result is not None:
                    return result
        finally:
            self._evals -= 1

    async def _eval_locked(self, code: str, on_chunk: OnChunk | None) -> EvalResult | None:
        async with self._lock:
            # A restart while this waited for the lock may have begun setting the
            # fresh process up, which goes first: ``None`` to wait for it and queue
            # again.
            if self._setting_up():
                return None
            # Checked under the lock: an eval queued behind a cancelled one can find
            # that the cancel's recovery restarted Julia, with a new connection.
            if not self.running:
                # Every call after a death lands here, so it explains itself the same
                # way the one that witnessed the death does; otherwise only the first
                # caller learns why and the rest see a bare "not running".
                raise await self._explain_death(KernelDied("the Julia kernel exited unexpectedly"))
            conn = self._conn
            assert conn is not None
            pending = conn.begin_eval(on_chunk)
            try:
                await conn.send_exec(pending, code)
                # Shielded: cancelling this task must not cancel the result slot
                # itself; the recovery below still needs the result frame to
                # land in it (a cancelled future could never be resolved).
                status, payload = await asyncio.shield(pending.future)
            except KernelDied as exc:
                raise await self._explain_death(exc) from None
            except (ConnectionError, OSError) as exc:
                # A torn control socket mid-send or while waiting for the result frame
                # is operationally the same as the kernel dying: callers need the
                # supervised process context, not a bare socket error.
                raise await self._explain_death(
                    KernelDied(f"control connection failed ({type(exc).__name__}: {exc})")
                ) from None
            except asyncio.CancelledError:
                # The caller was cancelled mid-eval. Interrupt the eval and wait
                # for its one result frame so the session stays usable, rather
                # than restarting. Shielded so the cancellation can't abort the
                # recovery itself.
                await asyncio.shield(self._recover_from_cancel(pending))
                raise
            finally:
                conn.end_eval(pending)
        return self._build_result(status, payload, pending)

    async def _explain_death(self, exc: KernelDied) -> KernelDied:
        """Restate a kernel death with the exit status and Julia's last output.

        The transport only knows that the connection went away, which leaves a
        crash, an out-of-memory kill and a plain exit indistinguishable to the
        caller even though each needs a different response. The supervisor owns
        the process, so it can say which one happened and quote the output that
        came just before.
        """
        proc = self._proc
        code = proc.returncode if proc is not None else None
        if proc is not None and code is None:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(_EXIT_STATUS_TIMEOUT):
                    code = await proc.wait()
        parts = [f"{exc}: {describe_exit(code)}"]
        tail = self._conn.stderr_tail.decode("utf-8", "replace").strip() if self._conn else ""
        if tail:
            parts.append(f"Julia's last output:\n{tail[-_DEATH_TAIL_CHARS:]}")
        if self._config.stderr_file is not None:
            parts.append(f"Full log: {self._config.stderr_file}")
        return KernelDied("\n".join(parts))

    async def _recover_from_cancel(self, pending: PendingEval) -> None:
        """Leave the session usable after an eval was cancelled mid-flight.

        If the eval is still running (its future is unresolved), interrupt it and
        wait for its result frame; the connection drops that stale result. The
        interrupt comes in two steps. The server is asked first (an ``INT``
        frame): it wakes a waiting eval with the exception itself, which SIGINT
        cannot be relied on to do: SIGINT lands in whichever task is current on
        the eval's thread, and while the eval waits, that can be any other task
        running there (see server.jl). An eval still going after
        ``_INTERRUPT_SIGNAL_DELAY`` is computing, which only the signal reaches.
        If the signal can't be sent, the eval ignores both, or the process dies,
        fall back to a restart (the only case where REPL state is lost): a
        running solve we can't interrupt must not be left to keep going.
        """
        conn = self._conn
        if not pending.future.done() and conn is not None:
            with contextlib.suppress(Exception):
                await conn.send_interrupt(pending)
            await asyncio.wait({pending.future}, timeout=_INTERRUPT_SIGNAL_DELAY)
        # `and` short-circuits, so interrupt() is only attempted while the eval is
        # still running; a False return (no soft interrupt available) means restart.
        if not pending.future.done() and not await self.interrupt():
            await self._restart_after_cancel("a cancelled evaluation could not be interrupted")
            return
        try:
            async with asyncio.timeout(_INTERRUPT_DRAIN_TIMEOUT):
                await pending.future
            self._cancel_preserved_state = True
        except Exception:
            await self._restart_after_cancel(
                "a cancelled evaluation did not stop within "
                f"{_INTERRUPT_DRAIN_TIMEOUT:g} s of being interrupted"
            )

    async def _restart_after_cancel(self, reason: str) -> None:
        self._cancel_preserved_state = False
        with contextlib.suppress(Exception):
            await self.restart(reason)

    @property
    def cancel_preserved_state(self) -> bool:
        """Whether the last cancelled eval kept REPL state (vs. forced a restart)."""
        return self._cancel_preserved_state

    def _build_result(self, status: str, payload: bytes, pending: PendingEval) -> EvalResult:
        out = render_terminal_output(bytes(pending.out).decode("utf-8", "replace"))
        err = render_terminal_output(bytes(pending.err).decode("utf-8", "replace"))
        text = payload.decode("utf-8", "replace")
        # REPL-style text the user code produced, whatever the outcome: stdout, then
        # stderr under a heading. On ok this gets the value's repr appended; on an
        # error or interrupt it is the output printed before the eval stopped, kept
        # so the caller can surface it next to the error.
        parts: list[str] = []
        if out.strip():
            parts.append(out.rstrip("\n"))
        if err.strip():
            parts.append("[stderr]\n" + err.rstrip("\n"))
        if pending.omitted:
            dropped = " and ".join(
                f"{count:,} bytes of {stream}" for stream, count in sorted(pending.omitted.items())
            )
            parts.append(
                f"[note: the output exceeded the capture buffer; {dropped} were "
                "omitted from the middle (the start and the end are kept)]"
            )
        if status == "int":
            return EvalResult(
                output="\n".join(parts),
                error="InterruptException: evaluation was interrupted",
                stdout=out,
                stderr=err,
                interrupted=True,
            )
        if status == "err":
            return EvalResult(output="\n".join(parts), error=text, stdout=out, stderr=err)
        # ok; append the value's repr.
        if text:
            parts.append(text)
        return EvalResult(
            output="\n".join(parts),
            value_repr=text or None,
            stdout=out,
            stderr=err,
        )

    # ---- spawn / teardown --------------------------------------------------

    async def _spawn(self) -> None:
        loop = asyncio.get_running_loop()

        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._listener.setblocking(False)
        port = self._listener.getsockname()[1]
        self._token = secrets.token_hex(16)

        if self._config.stderr_file is not None:
            self._config.stderr_file.parent.mkdir(parents=True, exist_ok=True)
            _keep_previous_log(self._config.stderr_file)
            self._stderr_fh = open(self._config.stderr_file, "wb")  # noqa: SIM115

        command, args, env = self._build_launch(port)
        # Put the kernel in its own process group so an interrupt targets only it,
        # never the agent. POSIX: a new session (setsid) for SIGINT. Windows:
        # CREATE_NEW_PROCESS_GROUP, which is what lets CTRL_BREAK_EVENT reach this
        # child (and its Distributed workers) without hitting our console.
        if os.name == "nt":
            group_kwargs: dict[str, object] = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            group_kwargs = {"start_new_session": True}
        try:
            self._proc = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._config.cwd) if self._config.cwd is not None else None,
                env=env,
                **group_kwargs,
            )
        except FileNotFoundError as exc:
            await self._teardown(graceful=False)  # close the listener / stderr log we opened
            raise JuliaStartupError(
                f"`{command}` is not on PATH",
                julia_executable=self._config.julia_executable,
                julia_project=self._config.julia_project,
            ) from exc

        assert self._proc.stdout is not None and self._proc.stderr is not None
        self._conn = KernelConnection(
            self._proc.stdout, self._proc.stderr, stderr_fh=self._stderr_fh
        )
        self._watch_task = asyncio.create_task(self._watch_proc(self._proc, self._conn))

        try:
            conn = await self._accept_control(loop, port)
            await self._conn.attach_control(conn)
            await self._await_ready()
        except JuliaStartupError:
            await self._teardown(graceful=False)
            raise
        except BaseException as exc:
            tail = self._conn.stderr_tail.decode("utf-8", "replace") if self._conn else ""
            await self._teardown(graceful=False)
            raise JuliaStartupError(
                "the Julia process exited before it was ready (see Julia output below)",
                julia_executable=self._config.julia_executable,
                julia_project=self._config.julia_project,
                stderr_tail=tail,
                log_file=self._config.stderr_file,
            ) from exc
        self._generation += 1

    async def _accept_control(self, loop: asyncio.AbstractEventLoop, port: int) -> socket.socket:
        assert self._listener is not None and self._conn is not None
        accept = asyncio.ensure_future(loop.sock_accept(self._listener))
        dead = asyncio.ensure_future(self._conn.closed.wait())
        done, _ = await asyncio.wait(
            {accept, dead},
            timeout=self._config.startup_timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if accept not in done:
            accept.cancel()
            dead.cancel()
            why = (
                "the Julia process exited" if self._conn.closed.is_set() else "timed out connecting"
            )
            raise JuliaStartupError(
                f"{why} before the control channel came up",
                julia_executable=self._config.julia_executable,
                julia_project=self._config.julia_project,
                stderr_tail=self._conn.stderr_tail.decode("utf-8", "replace"),
                log_file=self._config.stderr_file,
            )
        dead.cancel()
        conn, _ = accept.result()
        return conn

    async def _await_ready(self) -> None:
        assert self._conn is not None
        try:
            # The accept above is timed, but a client that connects and then
            # stays silent would otherwise hold startup forever (the listener
            # accepts exactly one connection, first come first served).
            async with asyncio.timeout(self._config.startup_timeout):
                token = await self._conn.ready_token()
        except TimeoutError:
            raise JuliaStartupError(
                "timed out waiting for the control handshake",
                julia_executable=self._config.julia_executable,
                julia_project=self._config.julia_project,
                stderr_tail=self._conn.stderr_tail.decode("utf-8", "replace"),
                log_file=self._config.stderr_file,
            ) from None
        if token != self._token:
            raise JuliaStartupError(
                "the control handshake failed (token mismatch)",
                julia_executable=self._config.julia_executable,
                julia_project=self._config.julia_project,
            )

    async def _watch_proc(self, proc: asyncio.subprocess.Process, conn: KernelConnection) -> None:
        with contextlib.suppress(Exception):
            await proc.wait()
        conn.closed.set()

    async def _teardown(self, *, graceful: bool) -> None:
        proc = self._proc
        if proc is not None and proc.returncode is None:
            if graceful and self._conn is not None and self._conn.has_control:
                self._conn.close_control()  # clean EOF exit for the server
                with contextlib.suppress(TimeoutError, Exception):
                    await asyncio.wait_for(proc.wait(), timeout=5)
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        if self._watch_task is not None:
            self._watch_task.cancel()
            with contextlib.suppress(BaseException):
                await self._watch_task
            self._watch_task = None
        if self._conn is not None:
            await self._conn.aclose()
            self._conn = None
        if self._listener is not None:
            with contextlib.suppress(Exception):
                self._listener.close()
            self._listener = None
        if self._stderr_fh is not None:
            with contextlib.suppress(Exception):
                self._stderr_fh.close()
            self._stderr_fh = None
        self._proc = None
        self._token = ""

    def _build_launch(self, port: int) -> tuple[str, list[str], dict[str, str]]:
        cfg = self._config
        args: list[str] = []
        if cfg.julia_project is not None:
            args.append(f"--project={cfg.julia_project}")
        if cfg.sysimage is not None:
            args.append(f"--sysimage={cfg.sysimage}")
        args.append(f"--threads={_thread_flag(cfg.threads)}")
        args.extend(cfg.extra_args)
        args.extend([str(_SERVER_JL), str(port)])

        env = dict(os.environ)
        if cfg.env:
            env.update(cfg.env)
        env["JK_TOKEN"] = self._token

        return cfg.julia_executable, args, env


def _keep_previous_log(log: Path) -> None:
    """Move the last process's log aside (``<stem>.prev<suffix>``) before a respawn
    starts a fresh one, so that why the last process went away stays readable."""
    with contextlib.suppress(OSError):
        if log.stat().st_size > 0:
            os.replace(log, log.with_name(f"{log.stem}.prev{log.suffix}"))


def describe_exit(code: int | None) -> str:
    """How the process ended, in the terms of whichever platform ended it.

    Public because the kernel is not the only place that has to explain a dead
    Julia: the system-image build reports one too, and a crash should read the
    same way whichever of them saw it.

    POSIX reports a signal as a negative code; SIGKILL is called out because the
    common source of one nobody sent is the out-of-memory killer. Windows has no
    signals and encodes a fault in the status itself, so a large code is also
    given in hex, the form its documentation and search results use.
    """
    if code is None:
        return "exit status unknown"
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        if -code == getattr(signal, "SIGKILL", None):
            return f"killed by {name}, usually the system out-of-memory killer"
        return f"killed by {name}, which normally means a crash inside native code"
    if code > 0xFF:
        return f"exit code {code} (0x{code:08X}), which normally means a crash inside native code"
    return f"exit code {code}"


def _thread_flag(threads: str | None) -> str:
    """The ``--threads`` value, always reserving an interactive thread.

    The server pins its eval loop to the interactive thread and its output
    pumps to the default pool; SIGINT delivery relies on that separation (a
    pump sharing the eval's thread could swallow the InterruptException). So
    a plain thread count N becomes ``N,1``. ``auto`` already implies one
    interactive thread, and an explicit ``N,M`` is respected as given.
    """
    if threads is None:
        return "1,1"
    if threads == "auto" or "," in threads:
        return threads
    return f"{threads},1"
