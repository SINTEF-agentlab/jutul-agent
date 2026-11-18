"""The backend end of a session's live-plot pipe (desktop app only).

``jutul-agent web`` serves live figures from a Bonito HTTP server inside the
Julia kernel, which the app server reverse-proxies. The desktop app may not
have a listening endpoint of any kind, so there the figures travel over the
kernel's auxiliary pipe instead: anonymous pipes the Julia process inherited at
spawn (``JuliaKernel.aux_streams``). ``JutulAgent.JutulAgentPipeViews`` is the
other end, and its docstring has the frame protocol. The app's ``/live/...``
routes relay between this object and the window.

One ``PlotPipe`` belongs to one session and outlives its kernel: after a Julia
restart it is bound to the new process's pipe.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
from collections.abc import Awaitable, Callable
from typing import Any

# How long the kernel may take to render a page. A long eval holds the thread the
# figure lives on, so this is sized like the reverse proxy's read timeout.
RENDER_TIMEOUT = 300.0

OnMessage = Callable[[bytes], Awaitable[None]]
OnClose = Callable[[], Awaitable[None]]


class PlotPipeError(RuntimeError):
    """A page could not be rendered; ``status`` is the HTTP status to answer with."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(reason)
        self.status = status


class PlotPipe:
    """Listens for the kernel's plot connection and relays pages and messages."""

    def __init__(self) -> None:
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._requests: dict[str, asyncio.Future[bytes]] = {}
        self._views: dict[str, tuple[OnMessage, OnClose]] = {}
        self._ids = itertools.count(1)
        self._serving = False

    @property
    def connected(self) -> bool:
        """Whether a kernel is serving figures on the pipe right now."""
        return self._serving and self._writer is not None and not self._writer.is_closing()

    def bind(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Use this kernel's auxiliary pipe (a no-op if already bound to it)."""
        if reader is self._reader:
            return
        # A new Julia process: whatever waited on the old one can't be answered.
        self._drop_connection()
        self._reader, self._writer = reader, writer
        self._reader_task = asyncio.create_task(self._pump(reader, writer))

    async def aclose(self) -> None:
        task, self._reader_task = self._reader_task, None
        self._drop_connection()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # ---- the kernel's end --------------------------------------------------------

    async def _pump(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                head = await reader.readline()
                if not head:
                    break
                parts = head.split()
                if len(parts) != 3:
                    continue
                kind, ident = parts[0], parts[1].decode("ascii", "replace")
                body = await reader.readexactly(int(parts[2]))
                if kind == b"HEL":
                    # A (new) server on the Julia side: views of an earlier one
                    # are gone with it.
                    self._fail_waiters()
                    self._serving = True
                elif kind in (b"HTM", b"ERR"):
                    future = self._requests.pop(ident, None)
                    if future is None or future.done():
                        continue
                    if kind == b"HTM":
                        future.set_result(body)
                    else:
                        status, _, reason = body.decode("utf-8", "replace").partition(" ")
                        code = int(status) if status.isdigit() else 500
                        future.set_exception(PlotPipeError(code, reason or "render failed"))
                elif kind == b"MSG":
                    view = self._views.get(ident)
                    if view is not None:
                        with contextlib.suppress(Exception):  # a gone page must not stop the pump
                            await view[0](body)
                elif kind == b"CLS":
                    view = self._views.pop(ident, None)
                    if view is not None:
                        with contextlib.suppress(Exception):
                            await view[1]()
        except (asyncio.IncompleteReadError, ConnectionError, OSError, ValueError):
            pass
        finally:
            if self._reader is reader:
                self._drop_connection()

    def _drop_connection(self) -> None:
        # The streams belong to the kernel, which closes them with its process.
        self._reader = self._writer = None
        self._serving = False
        self._fail_waiters()

    def _fail_waiters(self) -> None:
        requests, self._requests = self._requests, {}
        for future in requests.values():
            if not future.done():
                future.set_exception(PlotPipeError(502, "the live plot server went away"))
                future.exception()  # consumed here if the waiter already gave up
        views, self._views = self._views, {}
        for _on_message, on_close in views.values():
            # Scheduled, not awaited: this also runs from non-async teardown.
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(_quietly(on_close()))

    def _send(self, kind: str, ident: str, body: bytes = b"") -> None:
        writer = self._writer
        if writer is None or writer.is_closing():
            raise PlotPipeError(502, "the live plot server is not connected")
        writer.write(f"{kind} {ident} {len(body)}\n".encode("ascii") + body)

    # ---- what the app's /live routes call ----------------------------------------

    async def render(self, route: str) -> bytes:
        """The HTML page of the figure served on ``route``."""
        ident = str(next(self._ids))
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self._requests[ident] = future
        try:
            self._send("GET", ident, route.encode("utf-8"))
            return await asyncio.wait_for(future, RENDER_TIMEOUT)
        except TimeoutError:
            raise PlotPipeError(504, "the live plot server did not answer") from None
        finally:
            self._requests.pop(ident, None)

    def attach(self, view: str, on_message: OnMessage, on_close: OnClose) -> None:
        """Connect a page to its view; raises ``PlotPipeError`` if no kernel is there."""
        self._send("OPN", view)
        self._views[view] = (on_message, on_close)

    def send(self, view: str, data: bytes) -> None:
        if view in self._views:
            self._send("MSG", view, data)

    def detach(self, view: str) -> None:
        if self._views.pop(view, None) is not None:
            with contextlib.suppress(PlotPipeError):
                self._send("CLS", view)


async def _quietly(awaitable: Awaitable[Any]) -> None:
    with contextlib.suppress(Exception):
        await awaitable


def ensure(session: Any) -> PlotPipe | None:
    """The session's plot pipe, bound to its current kernel; ``None`` when the
    kernel has no auxiliary pipe (it was not started in pipe mode)."""
    aux_streams = getattr(session.julia, "aux_streams", None)
    streams = aux_streams() if aux_streams is not None else None
    if streams is None:
        return None
    pipe = getattr(session, "web_plot_pipe", None)
    if pipe is None:
        pipe = PlotPipe()
        session.web_plot_pipe = pipe
    pipe.bind(*streams)
    return pipe
