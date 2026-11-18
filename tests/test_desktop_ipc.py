"""The desktop app's transports: frames over pipes instead of sockets.

Three seams, each tested against the real code on real pipes:

- ``interfaces/desktop``: the ASGI application driven from frames (what uvicorn
  does for ``jutul-agent web``).
- ``juliakernel`` in pipe mode: the control channel and the auxiliary channel
  over pipes the Julia process inherits, with nothing listening.
- ``agent/plot_pipe``: the backend end of the live-plot pipe.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from typing import Any

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

from jutul_agent.agent.plot_pipe import PlotPipe, PlotPipeError
from jutul_agent.interfaces.desktop import framing
from jutul_agent.interfaces.desktop.asgi_bridge import serve
from jutul_agent.juliakernel import JuliaKernel, KernelConfig
from jutul_agent.juliakernel.config import IPC_ENV_VAR, pipes_enabled
from jutul_agent.juliakernel.connection import KernelDied
from jutul_agent.juliakernel.kernel import JuliaStartupError

needs_julia = pytest.mark.skipif(shutil.which("julia") is None, reason="requires `julia` on PATH")
posix_only = pytest.mark.skipif(os.name == "nt", reason="pipe mode is POSIX-only for now")


# ---- framing -----------------------------------------------------------------


def test_frames_round_trip_with_binary_bodies() -> None:
    stream = io.BytesIO()
    stream.write(framing.encode({"t": "http", "id": 1}, b"\x00\xff body"))
    stream.write(framing.encode({"t": "ping", "id": 2}))
    stream.seek(0)
    assert framing.read(stream) == ({"t": "http", "id": 1}, b"\x00\xff body")
    assert framing.read(stream) == ({"t": "ping", "id": 2}, b"")
    assert framing.read(stream) is None  # a clean end, not an error


def test_a_torn_or_foreign_stream_is_an_error_not_a_frame() -> None:
    whole = framing.encode({"t": "x"}, b"abcdef")
    with pytest.raises(framing.FrameError):
        framing.read(io.BytesIO(whole[:-2]))
    with pytest.raises(framing.FrameError):
        framing.read(io.BytesIO(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"))


# ---- the ASGI bridge ---------------------------------------------------------


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/hello")
    def hello(name: str = "you") -> dict[str, str]:
        return {"hello": name}

    @app.post("/echo")
    async def echo(payload: dict[str, Any]) -> dict[str, Any]:
        return {"got": payload}

    @app.get("/stream")
    def stream() -> StreamingResponse:
        return StreamingResponse(iter([b"one,", b"two,", b"three"]), media_type="text/plain")

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("deliberate")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        while True:
            text = await websocket.receive_text()
            if text == "bye":
                await websocket.close(code=1000)
                return
            await websocket.send_text(text.upper())
            await websocket.send_bytes(text.encode())

    @app.websocket("/refused")
    async def refused(websocket: WebSocket) -> None:
        await websocket.close(code=1008)

    return app


class _Shell:
    """Plays the Electron shell: writes frames to the bridge and reads its answers."""

    def __init__(self) -> None:
        to_backend_r, to_backend_w = os.pipe()
        from_backend_r, from_backend_w = os.pipe()
        self.backend_in = os.fdopen(to_backend_r, "rb", 0)
        self.backend_out = os.fdopen(from_backend_w, "wb")
        self._out = os.fdopen(to_backend_w, "wb", 0)
        self._in = os.fdopen(from_backend_r, "rb", 0)
        self.frames: asyncio.Queue[tuple[dict[str, Any], bytes]] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> _Shell:
        loop = asyncio.get_running_loop()
        self.task = asyncio.create_task(serve(_app(), self.backend_in, self.backend_out))

        def pump() -> None:
            with contextlib.suppress(Exception):
                while (frame := framing.read(self._in)) is not None:
                    loop.call_soon_threadsafe(self.frames.put_nowait, frame)

        # A daemon thread: a blocked read must never keep the test process alive.
        threading.Thread(target=pump, daemon=True).start()
        header, _ = await self.next()
        assert header["t"] == "ready"
        return self

    async def __aexit__(self, *exc: object) -> None:
        with contextlib.suppress(Exception):
            self._out.close()  # the shell going away is what stops the backend
        assert self.task is not None
        await asyncio.wait_for(self.task, 10)
        self.close()

    def close(self) -> None:
        """Release both pipes, which also ends the reader thread."""
        for stream in (self._out, self.backend_out, self.backend_in, self._in):
            with contextlib.suppress(Exception):
                stream.close()

    def send(self, header: dict[str, Any], body: bytes = b"") -> None:
        self._out.write(framing.encode(header, body))

    async def next(self) -> tuple[dict[str, Any], bytes]:
        return await asyncio.wait_for(self.frames.get(), 10)

    async def http(self, ident: int, method: str, path: str, **kw: Any) -> tuple[int, bytes]:
        body = kw.pop("body", b"")
        self.send({"t": "http", "id": ident, "method": method, "path": path, **kw}, body)
        status, chunks = 0, []
        while True:
            header, data = await self.next()
            assert header["id"] == ident
            if header["t"] == "http.start":
                status = header["status"]
            else:
                chunks.append(data)
                if not header["more"]:
                    return status, b"".join(chunks)


async def test_http_requests_reach_the_app_without_a_socket() -> None:
    async with _Shell() as shell:
        status, body = await shell.http(1, "GET", "/hello", query="name=desktop")
        assert (status, json.loads(body)) == (200, {"hello": "desktop"})

        status, body = await shell.http(
            2,
            "POST",
            "/echo",
            headers=[["content-type", "application/json"]],
            body=b'{"a": [1, 2]}',
        )
        assert (status, json.loads(body)) == (200, {"got": {"a": [1, 2]}})

        status, body = await shell.http(3, "GET", "/stream")
        assert (status, body) == (200, b"one,two,three")

        status, _ = await shell.http(4, "GET", "/nowhere")
        assert status == 404


async def test_a_crashing_route_answers_500_and_the_bridge_lives_on() -> None:
    async with _Shell() as shell:
        status, _ = await shell.http(1, "GET", "/boom")
        assert status == 500
        status, _ = await shell.http(2, "GET", "/hello")
        assert status == 200


async def test_websocket_messages_flow_both_ways() -> None:
    async with _Shell() as shell:
        shell.send({"t": "ws.open", "id": 7, "path": "/ws", "query": "", "headers": []})
        assert (await shell.next())[0] == {"t": "ws.accept", "id": 7, "subprotocol": None}
        shell.send({"t": "ws.msg", "id": 7, "binary": False}, b"plot")
        assert await shell.next() == ({"t": "ws.msg", "id": 7, "binary": False}, b"PLOT")
        assert await shell.next() == ({"t": "ws.msg", "id": 7, "binary": True}, b"plot")
        shell.send({"t": "ws.msg", "id": 7, "binary": False}, b"bye")
        header, _ = await shell.next()
        assert (header["t"], header["code"]) == ("ws.close", 1000)


async def test_a_refused_websocket_closes_before_any_accept() -> None:
    async with _Shell() as shell:
        shell.send({"t": "ws.open", "id": 3, "path": "/refused", "query": "", "headers": []})
        header, _ = await shell.next()
        assert (header["t"], header["code"]) == ("ws.close", 1008)


async def test_ping_and_explicit_shutdown() -> None:
    shell = _Shell()
    await shell.__aenter__()
    shell.send({"t": "ping", "id": 9})
    assert (await shell.next())[0] == {"t": "pong", "id": 9}
    shell.send({"t": "shutdown"})
    assert shell.task is not None
    await asyncio.wait_for(shell.task, 10)
    shell.close()


def test_the_backend_never_loads_a_network_server() -> None:
    """The bridge replaces uvicorn; importing the desktop backend must not pull it in."""
    code = (
        "import sys\n"
        "import jutul_agent.interfaces.desktop.backend\n"
        "import jutul_agent.interfaces.desktop.asgi_bridge\n"
        "print('uvicorn' in sys.modules)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "False"


# ---- the kernel over inherited pipes -----------------------------------------


def test_pipe_mode_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(IPC_ENV_VAR, raising=False)
    assert not pipes_enabled() and not KernelConfig().pipes
    monkeypatch.setenv(IPC_ENV_VAR, "pipe")
    assert pipes_enabled() and KernelConfig().pipes


@pytest.mark.skipif(os.name != "nt", reason="the refusal is specific to Windows")
async def test_pipe_mode_refuses_rather_than_opening_a_port() -> None:
    with pytest.raises(JuliaStartupError):
        await JuliaKernel(KernelConfig(pipes=True)).__aenter__()


@needs_julia
@posix_only
@pytest.mark.integration
async def test_kernel_runs_over_pipes_with_nothing_listening() -> None:
    async with JuliaKernel(KernelConfig(pipes=True)) as kernel:
        assert kernel._listener is None  # no socket was ever bound
        result = await kernel.eval('println("hi"); 6 * 7')
        assert result.error is None and result.output == "hi\n42"

        # The auxiliary pipe is a second, independent byte channel.
        reader, writer = kernel.aux_streams()
        echo = await kernel.eval(
            'let fds = parse.(Int, split(ENV["JK_AUX_FDS"], ","))\n'
            "    global aux_in = Base.PipeEndpoint(RawFD(fds[1]))\n"
            "    global aux_out = Base.PipeEndpoint(RawFD(fds[2]))\n"
            '    @async write(aux_out, uppercase(readline(aux_in)) * "\\n")\n'
            "    nothing\n"
            "end"
        )
        assert echo.error is None
        writer.write(b"over the pipe\n")
        assert await asyncio.wait_for(reader.readline(), 10) == b"OVER THE PIPE\n"

        # A restart is a new process with new pipes; the old ones are gone.
        await kernel.restart()
        assert kernel.aux_streams() is not None and kernel.aux_streams()[0] is not reader
        assert (await kernel.eval("1 + 1")).output == "2"
    assert kernel.aux_streams() is None


@needs_julia
@posix_only
@pytest.mark.integration
async def test_a_killed_pipe_kernel_is_reported_and_restartable() -> None:
    async with JuliaKernel(KernelConfig(pipes=True)) as kernel:
        # The pipe reading as closed is how the death is noticed, and the
        # supervisor still says how the process ended.
        with pytest.raises(KernelDied, match="exit code 3"):
            await kernel.eval("exit(3)")
        await kernel.restart()
        assert (await kernel.eval("40 + 2")).output == "42"


# ---- the live-plot pipe --------------------------------------------------------


class _FakeJulia:
    """The kernel's end of the plot pipe: answers GET, echoes view messages."""

    def __init__(self) -> None:
        to_julia_r, to_julia_w = os.pipe()
        from_julia_r, from_julia_w = os.pipe()
        self._fds = (to_julia_r, to_julia_w, from_julia_r, from_julia_w)
        self.seen: list[tuple[str, str, bytes]] = []

    async def open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Returns the backend's streams; keeps Julia's for itself."""
        to_julia_r, to_julia_w, from_julia_r, from_julia_w = self._fds
        self.reader, _ = await _streams(to_julia_r, None)
        _, self.writer = await _streams(None, from_julia_w)
        reader, _ = await _streams(from_julia_r, None)
        _, writer = await _streams(None, to_julia_w)
        self.task = asyncio.create_task(self._run())
        return reader, writer

    def send(self, kind: str, ident: str, body: bytes = b"") -> None:
        self.writer.write(f"{kind} {ident} {len(body)}\n".encode() + body)

    async def _run(self) -> None:
        self.send("HEL", "1")
        while head := await self.reader.readline():
            kind, ident, size = head.decode().split()
            body = await self.reader.readexactly(int(size))
            self.seen.append((kind, ident, body))
            if kind == "GET" and body == b"/viz/known":
                self.send("HTM", ident, b"<html>figure</html>")
            elif kind == "GET":
                self.send("ERR", ident, b"404 no such plot route")
            elif kind == "MSG":
                self.send("MSG", ident, body[::-1])

    def die(self) -> None:
        self.task.cancel()
        self.writer.close()


async def _streams(read_fd: int | None, write_fd: int | None) -> tuple[Any, Any]:
    loop = asyncio.get_running_loop()
    reader = writer = None
    if read_fd is not None:
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(read_fd, "rb", 0)
        )
    if write_fd is not None:
        transport, protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, os.fdopen(write_fd, "wb", 0)
        )
        writer = asyncio.StreamWriter(transport, protocol, None, loop)
    return reader, writer


async def _until(condition: Any) -> None:
    async with asyncio.timeout(10):
        while not condition():
            await asyncio.sleep(0.01)


@posix_only
async def test_plot_pipe_renders_pages_and_relays_view_messages() -> None:
    julia = _FakeJulia()
    pipe = PlotPipe()
    assert not pipe.connected
    pipe.bind(*await julia.open())
    await _until(lambda: pipe.connected)

    assert await pipe.render("/viz/known") == b"<html>figure</html>"
    with pytest.raises(PlotPipeError) as missing:
        await pipe.render("/viz/unknown")
    assert missing.value.status == 404

    got: list[bytes] = []
    closed = asyncio.Event()

    async def on_message(data: bytes) -> None:
        got.append(data)

    async def on_close() -> None:
        closed.set()

    pipe.attach("view1", on_message, on_close)
    pipe.send("view1", b"abc")
    await _until(lambda: got == [b"cba"])
    assert ("OPN", "view1", b"") in julia.seen

    # The kernel going away fails what waits on it and closes its views.
    julia.die()
    await asyncio.wait_for(closed.wait(), 10)
    assert not pipe.connected
    with pytest.raises(PlotPipeError):
        await pipe.render("/viz/known")
    await pipe.aclose()


# ---- the plot tool's choice of server -------------------------------------------


class _RecordingJulia:
    """A kernel stand-in that records what it is asked to evaluate."""

    def __init__(self, aux: Any = None) -> None:
        self.code: list[str] = []
        self._aux = aux

    def aux_streams(self) -> Any:
        return self._aux

    async def eval(self, code: str, on_chunk: Any = None) -> Any:
        from jutul_agent.agent import plot_julia_src as jl
        from jutul_agent.juliakernel import EvalResult

        self.code.append(code)
        return EvalResult(output=jl.WEB_PIPE_MARKER)


class _PlotSession:
    session_id = "s1"
    base_path = ""
    web_plot_port: int | None = None
    web_plot_pipe: Any = None

    def __init__(self, julia: _RecordingJulia) -> None:
        self.julia = julia


@posix_only
async def test_pipe_mode_serves_plots_on_the_pipe_and_never_starts_a_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jutul_agent.agent.plot_julia import _start_live_server

    monkeypatch.setenv(IPC_ENV_VAR, "pipe")
    julia_end = _FakeJulia()
    session = _PlotSession(_RecordingJulia(aux=await julia_end.open()))
    assert await _start_live_server(session) is None
    assert session.web_plot_port is None and session.web_plot_pipe is not None
    (code,) = session.julia.code
    assert 'JutulAgentPipeViews.serve(raw"/live/s1/")' in code
    assert "Bonito.Server" not in code
    await session.web_plot_pipe.aclose()
    julia_end.die()


async def test_pipe_mode_without_a_pipe_fails_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from jutul_agent.agent.plot_julia import _start_live_server

    monkeypatch.setenv(IPC_ENV_VAR, "pipe")
    session = _PlotSession(_RecordingJulia(aux=None))
    assert "no pipe" in (await _start_live_server(session) or "")
    assert session.julia.code == []  # nothing was started, least of all a port
