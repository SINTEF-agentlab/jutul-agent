"""Drive an ASGI application from frames instead of from a network socket.

This is the part uvicorn plays for ``jutul-agent web``, with the HTTP parsing
taken out: the desktop shell has already split a request into method, path,
headers and body, so a frame maps straight onto an ASGI scope. Each request and
each WebSocket runs as its own task, exactly as under a server, which is what
lets the FastAPI app be reused unchanged.

Frames in (shell -> backend)::

    {"t":"http", "id", "method", "path", "query", "headers":[[k,v],...]} + body
    {"t":"http.abort", "id"}                    the page stopped waiting
    {"t":"ws.open", "id", "path", "query", "headers", "subprotocols":[...]}
    {"t":"ws.msg", "id", "binary":bool} + body
    {"t":"ws.close", "id", "code"}
    {"t":"ping", "id"}
    {"t":"shutdown"}

Frames out (backend -> shell)::

    {"t":"ready", "pid", "version"}             once, after startup
    {"t":"http.start", "id", "status", "headers"}
    {"t":"http.body", "id", "more":bool} + body
    {"t":"ws.accept", "id", "subprotocol"}
    {"t":"ws.msg", "id", "binary":bool} + body
    {"t":"ws.close", "id", "code", "reason"}    before an accept: refused
    {"t":"pong", "id"}

Ids are chosen by the shell and are unique per kind for the life of the pipe.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import sys
import threading
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, BinaryIO
from urllib.parse import unquote

from jutul_agent.interfaces.desktop import framing

_log = logging.getLogger("jutul_agent.desktop")

ASGIApp = Callable[..., Awaitable[None]]
# Seconds the app's own shutdown (closing every session, and so every Julia
# kernel) may take before the process exits anyway.
SHUTDOWN_GRACE = 20.0


class FrameWriter:
    """Serializes frames onto a blocking stream from a dedicated thread.

    The event loop must never block on a full pipe (the shell reads at its own
    pace, and a large plot payload can exceed the pipe buffer), so frames are
    queued and written off-loop. One thread also means one writer: frames from
    concurrent requests can't interleave mid-frame.
    """

    def __init__(self, stream: BinaryIO, on_broken: Callable[[], None]) -> None:
        self._stream = stream
        self._on_broken = on_broken
        self._queue: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="desktop-ipc-writer", daemon=True)
        self._thread.start()

    def send(self, header: dict[str, Any], body: bytes = b"") -> None:
        self._queue.put(framing.encode(header, body))

    def close(self, timeout: float = 5.0) -> None:
        """Flush what is queued, then stop."""
        self._queue.put(None)
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            data = self._queue.get()
            if data is None:
                return
            try:
                self._stream.write(data)
                self._stream.flush()
            except (OSError, ValueError):
                # The shell is gone. Nothing more can be delivered; tell the
                # bridge so it shuts the sessions down instead of lingering.
                self._on_broken()
                return


class _Http:
    """One in-flight HTTP request."""

    def __init__(self, body: bytes) -> None:
        self.body: bytes | None = body
        self.disconnected = asyncio.Event()
        self.started = False
        self.finished = False
        self.task: asyncio.Task[None] | None = None


class _WebSocket:
    """One WebSocket: the events the app has yet to receive."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.accepted = False
        self.closed = False
        self.task: asyncio.Task[None] | None = None


def _headers(raw: Any, extra: dict[str, str]) -> list[tuple[bytes, bytes]]:
    """ASGI headers from the frame's pairs, with ``extra`` overriding by name."""
    out: list[tuple[bytes, bytes]] = []
    if isinstance(raw, list):
        for pair in raw:
            if isinstance(pair, list) and len(pair) == 2 and str(pair[0]).lower() not in extra:
                out.append(
                    (
                        str(pair[0]).lower().encode("latin-1", "replace"),
                        str(pair[1]).encode("latin-1", "replace"),
                    )
                )
    out.extend((k.encode("latin-1"), v.encode("latin-1")) for k, v in extra.items())
    return out


class AsgiBridge:
    """Runs ``app`` for the requests and sockets arriving as frames."""

    def __init__(self, app: ASGIApp, writer: FrameWriter, *, host: str = "localhost") -> None:
        self._app = app
        self._writer = writer
        # The Host every request is given. The app's origin guard checks it, and
        # the shell serves the UI under this same name, so the page's own
        # requests read as the server's own.
        self._host = host
        self._http: dict[int, _Http] = {}
        self._ws: dict[int, _WebSocket] = {}
        self._state: dict[str, Any] = {}
        self._lifespan_in: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._lifespan_task: asyncio.Task[None] | None = None
        self._lifespan_ok = False
        self._started = asyncio.Event()
        self._stopped = asyncio.Event()
        self.stopping = asyncio.Event()

    # ---- lifespan --------------------------------------------------------------

    async def startup(self) -> None:
        """Run the app's lifespan startup, as a server would before serving."""

        async def receive() -> dict[str, Any]:
            return await self._lifespan_in.get()

        async def send(message: dict[str, Any]) -> None:
            kind = message["type"]
            if kind == "lifespan.startup.complete":
                self._lifespan_ok = True
                self._started.set()
            elif kind == "lifespan.startup.failed":
                _log.error("application startup failed: %s", message.get("message", ""))
                self._started.set()
            elif kind in ("lifespan.shutdown.complete", "lifespan.shutdown.failed"):
                self._stopped.set()

        async def run() -> None:
            scope = {
                "type": "lifespan",
                "asgi": {"version": "3.0", "spec_version": "2.0"},
                "state": self._state,
            }
            try:
                await self._app(scope, receive, send)
            except BaseException:
                # An app without lifespan support raises here; that is allowed.
                _log.debug("lifespan ended: %s", traceback.format_exc())
            finally:
                self._started.set()
                self._stopped.set()

        self._lifespan_task = asyncio.create_task(run())
        await self._lifespan_in.put({"type": "lifespan.startup"})
        await self._started.wait()

    async def shutdown(self) -> None:
        """Cancel what is in flight, then let the app close its sessions."""
        tasks = [h.task for h in self._http.values() if h.task is not None]
        for ws in self._ws.values():
            # A socket hears a disconnect rather than a cancellation, so its
            # handler unwinds the way it does when a browser tab closes.
            ws.inbox.put_nowait({"type": "websocket.disconnect", "code": 1001})
            if ws.task is not None:
                tasks.append(ws.task)
        for h in self._http.values():
            if h.task is not None:
                h.task.cancel()
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=5.0)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._lifespan_task is not None and self._lifespan_ok:
            await self._lifespan_in.put({"type": "lifespan.shutdown"})
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopped.wait(), SHUTDOWN_GRACE)
        if self._lifespan_task is not None:
            self._lifespan_task.cancel()
            await asyncio.gather(self._lifespan_task, return_exceptions=True)

    # ---- dispatch --------------------------------------------------------------

    def dispatch(self, header: dict[str, Any], body: bytes) -> None:
        """Handle one incoming frame. Called on the event loop."""
        kind = header.get("t")
        ident = header.get("id")
        if kind == "shutdown":
            self.stopping.set()
        elif kind == "ping":
            self._writer.send({"t": "pong", "id": ident})
        elif not isinstance(ident, int):
            _log.warning("dropping a %r frame without an id", kind)
        elif kind == "http":
            request = _Http(body)
            self._http[ident] = request
            request.task = asyncio.create_task(self._run_http(ident, header, request))
        elif kind == "http.abort":
            request = self._http.get(ident)
            if request is not None:
                request.disconnected.set()
        elif kind == "ws.open":
            ws = _WebSocket()
            self._ws[ident] = ws
            ws.inbox.put_nowait({"type": "websocket.connect"})
            ws.task = asyncio.create_task(self._run_ws(ident, header, ws))
        elif kind == "ws.msg":
            ws = self._ws.get(ident)
            if ws is not None:
                event: dict[str, Any] = {"type": "websocket.receive"}
                if header.get("binary"):
                    event["bytes"] = body
                else:
                    event["text"] = body.decode("utf-8", "replace")
                ws.inbox.put_nowait(event)
        elif kind == "ws.close":
            ws = self._ws.get(ident)
            if ws is not None:
                ws.closed = True  # the shell already knows; don't echo a close back
                ws.inbox.put_nowait(
                    {"type": "websocket.disconnect", "code": int(header.get("code") or 1005)}
                )

    def _scope(self, kind: str, header: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(header.get("path") or "/")
        return {
            "type": kind,
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "scheme": "http" if kind == "http" else "ws",
            "path": unquote(raw_path),
            "raw_path": raw_path.encode("ascii", "replace"),
            "query_string": str(header.get("query") or "").encode("ascii", "replace"),
            "root_path": "",
            "headers": _headers(header.get("headers"), {"host": self._host}),
            # No peer address exists on a pipe; loopback is the honest stand-in
            # for "this computer".
            "client": ("127.0.0.1", 0),
            "server": (self._host, 80),
            "state": dict(self._state),
        }

    # ---- HTTP ------------------------------------------------------------------

    async def _run_http(self, ident: int, header: dict[str, Any], request: _Http) -> None:
        scope = self._scope("http", header)
        scope["method"] = str(header.get("method") or "GET").upper()

        async def receive() -> dict[str, Any]:
            if request.body is not None:
                body, request.body = request.body, None
                return {"type": "http.request", "body": body, "more_body": False}
            await request.disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if request.disconnected.is_set():
                return  # nobody is listening; let the app finish quietly
            if message["type"] == "http.response.start":
                request.started = True
                self._writer.send(
                    {
                        "t": "http.start",
                        "id": ident,
                        "status": int(message["status"]),
                        "headers": [
                            [k.decode("latin-1"), v.decode("latin-1")]
                            for k, v in message.get("headers", [])
                        ],
                    }
                )
            elif message["type"] == "http.response.body":
                more = bool(message.get("more_body", False))
                self._writer.send(
                    {"t": "http.body", "id": ident, "more": more},
                    bytes(message.get("body", b"")),
                )
                if not more:
                    request.finished = True
                    request.disconnected.set()

        try:
            await self._app(scope, receive, send)
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.error("unhandled error in %s %s\n%s", scope["method"], scope["path"], _tb())
        finally:
            self._http.pop(ident, None)
            if not request.finished and not request.disconnected.is_set():
                # The app ended without a complete response (it raised, or was
                # cancelled by a shutdown). The page must not wait forever.
                if not request.started:
                    self._writer.send(
                        {
                            "t": "http.start",
                            "id": ident,
                            "status": 500,
                            "headers": [["content-type", "text/plain; charset=utf-8"]],
                        }
                    )
                    self._writer.send(
                        {"t": "http.body", "id": ident, "more": False},
                        b"Internal Server Error",
                    )
                else:
                    self._writer.send({"t": "http.body", "id": ident, "more": False})

    # ---- WebSocket -------------------------------------------------------------

    async def _run_ws(self, ident: int, header: dict[str, Any], ws: _WebSocket) -> None:
        scope = self._scope("websocket", header)
        protocols = header.get("subprotocols")
        scope["subprotocols"] = [str(p) for p in protocols] if isinstance(protocols, list) else []

        def close(code: int, reason: str = "") -> None:
            if not ws.closed:
                ws.closed = True
                self._writer.send({"t": "ws.close", "id": ident, "code": code, "reason": reason})

        async def send(message: dict[str, Any]) -> None:
            kind = message["type"]
            if kind == "websocket.accept":
                ws.accepted = True
                self._writer.send(
                    {"t": "ws.accept", "id": ident, "subprotocol": message.get("subprotocol")}
                )
            elif kind == "websocket.send":
                if ws.closed:
                    # Same contract a server gives the app: sending to a peer
                    # that has gone raises, so its pump loops stop.
                    raise OSError("the desktop window closed this WebSocket")
                data = message.get("bytes")
                if data is not None:
                    self._writer.send({"t": "ws.msg", "id": ident, "binary": True}, bytes(data))
                else:
                    text = message.get("text") or ""
                    self._writer.send(
                        {"t": "ws.msg", "id": ident, "binary": False}, text.encode("utf-8")
                    )
            elif kind == "websocket.close":
                close(int(message.get("code") or 1000), str(message.get("reason") or ""))

        try:
            await self._app(scope, ws.inbox.get, send)
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.error("unhandled error in WebSocket %s\n%s", scope["path"], _tb())
            close(1011)
        finally:
            self._ws.pop(ident, None)
            close(1000 if ws.accepted else 1006)


def _tb() -> str:
    return traceback.format_exc()


def _reader_thread(
    stream: BinaryIO,
    loop: asyncio.AbstractEventLoop,
    bridge: AsgiBridge,
) -> threading.Thread:
    """Read frames on a thread (pipes are not selectable on every platform) and
    hand each to the loop. The end of the stream means the shell is gone, which
    stops the backend: it must never outlive the window it serves."""

    def run() -> None:
        try:
            while True:
                frame = framing.read(stream)
                if frame is None:
                    break
                loop.call_soon_threadsafe(bridge.dispatch, *frame)
        except (framing.FrameError, OSError, ValueError) as exc:
            print(f"desktop backend: input stream failed: {exc}", file=sys.stderr)
        finally:
            with contextlib.suppress(RuntimeError):  # the loop already closed
                loop.call_soon_threadsafe(bridge.stopping.set)

    thread = threading.Thread(target=run, name="desktop-ipc-reader", daemon=True)
    thread.start()
    return thread


async def serve(
    app: ASGIApp,
    stdin: BinaryIO,
    stdout: BinaryIO,
    *,
    version: str = "",
    on_ready: Callable[[AsgiBridge], None] | None = None,
) -> None:
    """Serve ``app`` over the two streams until the shell asks to stop or goes away."""
    loop = asyncio.get_running_loop()
    stop_from_thread = lambda: loop.call_soon_threadsafe(bridge.stopping.set)  # noqa: E731
    writer = FrameWriter(stdout, on_broken=stop_from_thread)
    bridge = AsgiBridge(app, writer)
    await bridge.startup()
    _reader_thread(stdin, loop, bridge)
    if on_ready is not None:
        on_ready(bridge)
    writer.send({"t": "ready", "pid": os.getpid(), "version": version})
    try:
        await bridge.stopping.wait()
    finally:
        await bridge.shutdown()
        writer.close()
