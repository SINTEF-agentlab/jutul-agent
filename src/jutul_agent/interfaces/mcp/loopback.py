"""The loopback viewer: a session's plots and reports, at URLs a browser can open.

The MCP client shows text and images, not a live WebGL figure, so the MCP server
also serves the same viewer routes as the web server (``server.viewer``) on a
port of its own, bound to 127.0.0.1. Tool results link into it.

Anything on the machine can reach a loopback port, so the links carry the
session id and the MCP server makes that id unguessable: knowing it is the
capability to view the session. Requests must also name the loopback host itself
(a rebinding page names its own domain), and a websocket must come from the
viewer's own pages.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jutul_agent.session import Session

_LOOPBACK = "127.0.0.1"
_STARTUP_TIMEOUT = 10.0
_SHUTDOWN_TIMEOUT = 5.0


class _OwnHostOnly:
    """ASGI guard: answer only requests addressed to this viewer's own origin."""

    def __init__(self, app: Any, port: int) -> None:
        self._app = app
        self._hosts = {f"{_LOOPBACK}:{port}", f"localhost:{port}"}
        self._origins = {f"http://{host}" for host in self._hosts}

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = {
                key.decode("latin-1").lower(): value.decode("latin-1")
                for key, value in scope["headers"]
            }
            host_ok = headers.get("host", "").lower() in self._hosts
            origin = headers.get("origin")
            origin_ok = origin is None or origin.lower() in self._origins
            if scope["type"] == "websocket" and not (host_ok and origin_ok):
                await send({"type": "websocket.close", "code": 1008})
                return
            if scope["type"] == "http" and not host_ok:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send({"type": "http.response.body", "body": b"forbidden"})
                return
        await self._app(scope, receive, send)


def _embedded_server(config: Any) -> Any:
    """A uvicorn server that leaves signal handling to the process hosting it."""
    import uvicorn

    class _Embedded(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self) -> Iterator[None]:
            yield

    return _Embedded(config)


class LoopbackViewer:
    """The viewer routes for one session, served on an ephemeral loopback port."""

    def __init__(
        self,
        *,
        live_session: Callable[[str], Session | None],
        output_dir: Callable[[str], Path | None],
        port: int = 0,
    ) -> None:
        self._live_session = live_session
        self._output_dir = output_dir
        self._port = port
        self._server: Any = None
        self._task: asyncio.Task[None] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{_LOOPBACK}:{self._port}"

    async def start(self) -> None:
        import uvicorn
        from fastapi import FastAPI

        from jutul_agent.interfaces.server.viewer import add_viewer_routes

        # Bind first, so the port is known before anything links to it.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((_LOOPBACK, self._port))
        self._port = sock.getsockname()[1]

        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        add_viewer_routes(app, live_session=self._live_session, output_dir=self._output_dir)
        config = uvicorn.Config(
            _OwnHostOnly(app, self._port),
            log_config=None,  # leave the host process's logging alone
            access_log=False,
            lifespan="off",
            timeout_graceful_shutdown=2,
        )
        self._server = _embedded_server(config)
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _STARTUP_TIMEOUT
        while not self._server.started:
            if self._task.done():
                self._task.result()  # surface why it stopped
                raise RuntimeError("the viewer stopped while starting")
            if loop.time() > deadline:
                raise RuntimeError("the viewer did not start")
            await asyncio.sleep(0.01)

    async def aclose(self) -> None:
        if self._task is None:
            return
        task, self._task = self._task, None
        self._server.should_exit = True
        # ``asyncio.wait`` never raises the server task's own outcome, so a
        # cancellation reaching here is the caller's and propagates.
        done, _ = await asyncio.wait({task}, timeout=_SHUTDOWN_TIMEOUT)
        if not done:
            task.cancel()
            await asyncio.wait({task})
        if not task.cancelled():
            task.exception()  # retrieved: a failed shutdown is not worth a warning
