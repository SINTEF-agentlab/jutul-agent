"""The parent endpoint of the kernel wire protocol.

One loopback TCP connection carries the whole protocol as typed,
length-prefixed frames; ``server.jl`` is the other end and the two must agree
on the header format (an ASCII ``"TYPE [args...] NBYTES\\n"`` line followed by
exactly NBYTES raw payload bytes). A single reader task demultiplexes frames:
``OUT`` feeds the in-flight eval's buffers and live sink, ``RES`` resolves its
future. An eval therefore has exactly one completion event, and "all output
arrived before the result" is guaranteed by TCP ordering plus the server's
drain step; there is nothing to stitch together here.

The Julia process's own stdout/stderr pipes carry no protocol: log pumps tail
them for startup diagnostics (the boot preamble, a crash banner) and append
them to the kernel log file.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import socket
from dataclasses import dataclass, field
from typing import IO

from .result import OnChunk, OutputChunk

_STDERR_TAIL_CAP = 16 * 1024
_LOG_READ_SIZE = 64 * 1024


class KernelDied(RuntimeError):
    """The Julia process went away while an eval was in flight or waiting.

    Distinct from the other transport errors so the supervisor, which owns the
    process and therefore its exit status, can add why it went away.
    """


@dataclass
class PendingEval:
    """One in-flight eval: its frame id, completion future, and output."""

    exec_id: int
    future: asyncio.Future[tuple[str, bytes]]  # (status, result payload)
    on_chunk: OnChunk | None = None
    out: bytearray = field(default_factory=bytearray)
    err: bytearray = field(default_factory=bytearray)
    _decoders: dict[str, codecs.IncrementalDecoder] = field(default_factory=dict, repr=False)

    def decode_chunk(self, stream: str, body: bytes) -> str:
        """Decode one streamed chunk without splitting multi-byte characters.

        Pipe flushes land anywhere, including inside a UTF-8 sequence (the
        box-drawing characters of a results table are three bytes each), so
        decoding each chunk on its own emits replacement characters at the
        split. The per-stream incremental decoder holds the partial sequence
        until its remaining bytes arrive with the next chunk.
        """
        decoder = self._decoders.get(stream)
        if decoder is None:
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            self._decoders[stream] = decoder
        return decoder.decode(body)


class KernelConnection:
    """The control connection to a running Julia server, plus its log pipes."""

    def __init__(
        self,
        stdout_reader: asyncio.StreamReader,
        stderr_reader: asyncio.StreamReader,
        *,
        stderr_fh: IO[bytes] | None = None,
    ) -> None:
        self.closed = asyncio.Event()
        self._death_reason: str | None = None
        self._stderr_tail = bytearray()
        self._stderr_fh = stderr_fh
        loop = asyncio.get_running_loop()
        self._ready: asyncio.Future[str] = loop.create_future()
        self._current: PendingEval | None = None
        self._next_id = 1
        self._ctrl_writer: asyncio.StreamWriter | None = None
        self._tasks = [
            asyncio.create_task(self._log_pump(stdout_reader, is_stderr=False)),
            asyncio.create_task(self._log_pump(stderr_reader, is_stderr=True)),
        ]

    async def attach_control(self, conn: socket.socket) -> None:
        """Open the control connection over ``conn`` and start reading frames."""
        reader, writer = await asyncio.open_connection(sock=conn)
        self._ctrl_writer = writer
        self._tasks.append(asyncio.create_task(self._read_frames(reader)))

    @property
    def has_control(self) -> bool:
        return self._ctrl_writer is not None

    @property
    def stderr_tail(self) -> bytes:
        """The most recent stderr bytes, kept to diagnose a startup failure."""
        return bytes(self._stderr_tail)

    # ---- protocol ------------------------------------------------------------

    async def ready_token(self) -> str:
        """The token from the server's RDY frame (raises if the kernel died first)."""
        return await self._ready

    def begin_eval(self, on_chunk: OnChunk | None) -> PendingEval:
        """Register the next eval; its frames route to the returned record."""
        if self._current is not None:
            raise RuntimeError("an eval is already in flight")
        pending = PendingEval(
            exec_id=self._next_id,
            future=asyncio.get_running_loop().create_future(),
            on_chunk=on_chunk,
        )
        self._next_id += 1
        if self.closed.is_set():
            pending.future.set_exception(KernelDied("the Julia kernel exited unexpectedly"))
            return pending
        self._current = pending
        return pending

    def end_eval(self, pending: PendingEval) -> None:
        """Stop routing frames to ``pending`` (a late RES for it is dropped)."""
        if self._current is pending:
            self._current = None

    async def send_exec(self, pending: PendingEval, code: str) -> None:
        """Frame ``code`` as the EXE for ``pending`` and write it out."""
        assert self._ctrl_writer is not None
        body = code.encode("utf-8")
        head = f"EXE {pending.exec_id} {len(body)}\n".encode("ascii")
        self._ctrl_writer.write(head + body)
        await self._ctrl_writer.drain()

    # ---- teardown --------------------------------------------------------------

    def close_control(self) -> None:
        """Close the control writer, giving the server a clean EOF exit."""
        if self._ctrl_writer is not None:
            with contextlib.suppress(Exception):
                self._ctrl_writer.close()

    async def aclose(self) -> None:
        """Stop the reader and log pumps and close the control writer."""
        self.close_control()
        self._poison()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    # ---- reader / pumps ---------------------------------------------------------

    async def _read_frames(self, reader: asyncio.StreamReader) -> None:
        """Demultiplex control frames until EOF, then poison every waiter."""
        try:
            while True:
                head = await reader.readline()
                if not head:
                    break
                parts = head.split()
                if len(parts) < 2:
                    continue
                kind = parts[0]
                body = await reader.readexactly(int(parts[-1]))
                if kind == b"OUT" and len(parts) == 3:
                    self._route_output(parts[1].decode("ascii", "replace"), body)
                elif kind == b"RES" and len(parts) == 4:
                    self._route_result(int(parts[1]), parts[2].decode("ascii", "replace"), body)
                elif kind == b"RDY" and len(parts) == 3 and not self._ready.done():
                    self._ready.set_result(parts[1].decode("ascii", "replace"))
        except (asyncio.IncompleteReadError, ConnectionError, OSError, ValueError) as exc:
            # A torn frame or dropped socket: treated as the kernel dying, with
            # the cause kept for the poison message.
            self._death_reason = f"{type(exc).__name__}: {exc}"
        finally:
            self._poison()

    def _route_output(self, stream: str, body: bytes) -> None:
        pending = self._current
        if pending is None:
            # Output with no eval in flight (e.g. a background task printing
            # between evals): keep it out of the next eval's result, but let it
            # surface in the kernel log.
            self._note_log(body, to_tail=stream == "stderr")
            return
        buf = pending.out if stream == "stdout" else pending.err
        buf += body
        if pending.on_chunk is not None and body:
            text = pending.decode_chunk(stream, body)
            if text:
                with contextlib.suppress(Exception):  # a bad sink must not wedge the reader
                    pending.on_chunk(OutputChunk(text=text, stream=stream))

    def _route_result(self, exec_id: int, status: str, body: bytes) -> None:
        pending = self._current
        if pending is None or pending.exec_id != exec_id:
            return  # the result of an abandoned eval; already recovered from
        self._current = None
        if not pending.future.done():
            pending.future.set_result((status, body))

    def _poison(self) -> None:
        """Wake every waiter with "the kernel exited" instead of hanging."""
        self.closed.set()
        message = "the Julia kernel exited unexpectedly"
        if self._death_reason:
            message += f" ({self._death_reason})"
        if not self._ready.done():
            self._ready.set_exception(KernelDied(message))
            self._ready.exception()  # consumed here if nobody awaits ready_token
        pending = self._current
        self._current = None
        if pending is not None and not pending.future.done():
            pending.future.set_exception(KernelDied(message))

    async def _log_pump(self, reader: asyncio.StreamReader, *, is_stderr: bool) -> None:
        """Tail one of the process's own pipes into the log (no protocol here)."""
        try:
            while True:
                chunk = await reader.read(_LOG_READ_SIZE)
                if not chunk:
                    break
                self._note_log(chunk, to_tail=is_stderr)
        finally:
            if self._death_reason is None:
                self._death_reason = f"{'stderr' if is_stderr else 'stdout'} pipe closed"
            # EOF on an inherited pipe means the process is gone.
            self._poison()

    def _note_log(self, chunk: bytes, *, to_tail: bool) -> None:
        if to_tail:
            self._stderr_tail += chunk
            if len(self._stderr_tail) > _STDERR_TAIL_CAP:
                del self._stderr_tail[:-_STDERR_TAIL_CAP]
        if self._stderr_fh is not None:
            with contextlib.suppress(Exception):
                self._stderr_fh.write(chunk)
                self._stderr_fh.flush()
