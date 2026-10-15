"""The routes a browser needs to view a session's output, mountable on any app.

A session's artifacts (plots, reports), its live plots (reverse-proxied to the
session's Bonito server), and the popout page that stages a live plot in a
window of its own. ``add_viewer_routes`` installs them around two lookups, so they
work for whichever front end owns the sessions: the web server includes them
next to its session API, and the MCP server serves them on a loopback port of
its own so a live plot can be opened from any browser on the machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from jutul_agent.trace import schema

if TYPE_CHECKING:
    from jutul_agent.session import Session

# Timeouts for the live-plot proxy routes. The plot server shares Julia's
# interactive thread with the kernel's eval loop, so a long eval (a solve, a slow
# figure build) can leave requests unanswered for minutes while the OS still
# accepts the TCP connection. Waiting serves those requests when the eval yields;
# a short timeout would turn every busy spell into a 502 or a dropped socket.
# Connects stay quick so a port nothing listens on still fails fast.
_LIVE_WS_OPEN_TIMEOUT = 120.0
_LIVE_HTTP_TIMEOUT = {"connect": 5.0, "read": 300.0, "write": 30.0, "pool": 30.0}

# A live plot's route id: a slot or ``plot-<hex>`` stem, optionally with the
# popout suffix a replay appended. What ``popout_wrapper`` accepts; anything
# else 404s rather than being echoed into a page.
_ROUTE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}(?:--pop\d+)?")


def _fit_target(stage: list[int], authored: Any) -> list[int]:
    """The size a re-fit resizes a live figure to, for a ``stage``-sized panel.

    The Python twin of the serve-time fit in ``jl._fig_size_block``, whose one
    rule is stated in full there: aspect clamped near the authored shape
    (letterbox rather than distort), fitted inside the panel at full text size,
    then grown so neither dimension falls below the squash floor. The two
    fractions are read from that module so the twins cannot drift apart.
    Anchoring to the authored size, not the current one, makes refitting
    idempotent. Without a recorded authored size the stage itself is target.
    """
    from jutul_agent.agent import plot_julia_src as jl

    sw, sh = stage
    if not (isinstance(authored, (list, tuple)) and len(authored) == 2):
        return [sw, sh]
    aw, ah = int(authored[0]), int(authored[1])
    if aw <= 0 or ah <= 0:
        return [sw, sh]
    ra = aw / ah
    r = min(max(sw / sh, ra * jl.FIT_ASPECT_FRACTION), ra / jl.FIT_ASPECT_FRACTION)
    w1 = min(sw, round(sh * r))
    h1 = round(w1 / r)
    s = min(
        max(1.0, aw * jl.FIT_SQUASH_FLOOR / w1, ah * jl.FIT_SQUASH_FLOOR / h1), jl.FIT_MAX_SCALE
    )
    return [round(w1 * s), round(h1 * s)]


def _sized(value: Any) -> list[int] | None:
    """A recorded ``[w, h]`` as plain ints, or ``None`` if it is not one."""
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        return None
    try:
        w, h = int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None
    return [w, h] if w > 0 and h > 0 else None


def _sane_size(width: Any, height: Any) -> list[int] | None:
    """A client-supplied pixel size as ``[w, h]``, or ``None`` if implausible.

    Guards every place the browser reports a measurement (the canvas hint, a
    replot target): a missing field, a junk type, or a degenerate rectangle
    yields ``None``, and callers fall back to not resizing at all.
    """
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        return None
    if not (100 <= w <= 8192 and 100 <= h <= 8192):
        return None
    return [w, h]


# The popout wrapper page (see ``popout_wrapper``). Placeholders instead of an
# f-string so the JS braces stay readable. The figure is embedded at its real
# pixel size and CSS-scaled down to the window (never upscaled: a canvas grown
# by CSS blurs). When the scaled figure covers the window poorly (the same
# gate the inline canvas uses) the wrapper POSTs a re-fit and then stages the
# size the server *echoes*, never the size it asked for. Until a refit lands,
# the scaled centered presentation is the always-correct fallback.
_POPOUT_WRAPPER_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>jutul-agent plot</title>
<style>
  /* `color-scheme` is what stops the white flash on a window resize, and no
     background declaration can substitute for it: while the window grows, the
     browser fills the newly exposed area with the window's base canvas,
     painted before any CSS applies, and that base is white until the page
     declares which schemes it supports. Declaring both, and painting the page
     in the matching system colour, makes base and page the same colour in
     either scheme, so there is nothing left to flash. */
  html { color-scheme: light dark; }
  html, body { margin: 0; height: 100%; overflow: hidden; background: Canvas; }
  /* Transparent, so the figure (whose page is transparent all the way down)
     composites onto the page colour above rather than onto white. */
  #stage { position: absolute; transform-origin: top left; background: transparent; }
  iframe { display: block; border: 0; background: transparent; }
</style>
</head>
<body>
<div id="stage"><iframe id="frame" src="__SRC__" allow="fullscreen"></iframe></div>
<script>
  let W = __W__, H = __H__;
  const REFIT = "__REFIT__";
  function fit() {
    const s = Math.min(innerWidth / W, innerHeight / H, 1);
    const stage = document.getElementById("stage");
    const frame = document.getElementById("frame");
    frame.style.width = W + "px";
    frame.style.height = H + "px";
    stage.style.transform = s < 1 ? "scale(" + s + ")" : "none";
    stage.style.left = Math.max(0, (innerWidth - W * s) / 2) + "px";
    stage.style.top = Math.max(0, (innerHeight - H * s) / 2) + "px";
  }
  let timer;
  addEventListener("resize", () => {
    fit();
    clearTimeout(timer);
    timer = setTimeout(async () => {
      // The scaled figure leaves too much of the window unused (grown past it,
      // or a shape mismatch): have the kernel re-fit the live figure to the
      // window, then stage the size it echoes; the floors may have held.
      const s = Math.min(innerWidth / W, innerHeight / H, 1);
      const coverage = (W * s * H * s) / (innerWidth * innerHeight);
      if (coverage >= 0.75) return;
      try {
        const resp = await fetch(
          REFIT + "&w=" + innerWidth + "&h=" + innerHeight, { method: "POST" });
        if (resp.ok) {
          const d = await resp.json();
          if (d && d.width >= 100 && d.height >= 100) { W = d.width; H = d.height; fit(); }
        }
      } catch (e) { /* best-effort: the scaled presentation stands */ }
    }, 1000);
  });
  fit();
</script>
</body>
</html>
"""


# What a live plot's link shows once the plot is gone, in place of the bare error
# its server answers: the iframe a popout or the canvas embeds shows this page.
_PLOT_GONE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>jutul-agent plot</title>
<style>
  html { color-scheme: light dark; }
  body { margin: 0; height: 100vh; display: grid; place-items: center;
         background: Canvas; color: CanvasText;
         font: max(16px, 2vw)/1.5 system-ui, sans-serif; }
  main { max-width: 34em; padding: 1em; text-align: center; }
</style>
</head>
<body><main>
<p><strong>This live plot has ended.</strong></p>
<p>__WHY__</p>
</main></body>
</html>
"""


def time_ago(at: float) -> str:
    """How long ago ``at`` (a ``time.time()``) was, in words, such as "3 min ago".

    A duration rather than a time of day: the server's clock can be in another
    time zone than the person reading it (a WSL distro on UTC, say).
    """
    minutes = int(max(0.0, time.time() - at) // 60)
    if minutes < 1:
        return "under a minute ago"
    if minutes < 60:
        return f"{minutes} min ago"
    return f"{minutes // 60} h {minutes % 60} min ago"


def _plot_gone(session: Session | None) -> HTMLResponse:
    """The page for a live plot that is no longer served, saying why if known."""
    restart = getattr(getattr(session, "julia", None), "last_restart", None)
    if session is None:
        why = "The session that drew it is no longer running."
    elif restart is not None:
        why = (
            f"Julia restarted {time_ago(restart.at)} because {restart.reason}, and a live "
            "plot ends with the Julia session that drew it. Ask for the plot again."
        )
    else:
        why = "It was closed, or the Julia session that drew it has ended."
    return HTMLResponse(_PLOT_GONE_HTML.replace("__WHY__", html.escape(why)), status_code=404)


def _resolve_artifact(output_dir: Path, path: str):
    """The artifact file for ``path``, or ``None`` if it escapes the artifacts dir."""
    base = (output_dir / "artifacts").resolve()
    target = (base / path).resolve()
    if not target.is_file() or not target.is_relative_to(base):
        return None
    return target


def add_viewer_routes(
    target: FastAPI | APIRouter,
    *,
    live_session: Callable[[str], Session | None],
    output_dir: Callable[[str], Path | None],
) -> None:
    """Install the viewer routes on ``target``, resolving sessions through lookups.

    ``live_session`` maps a session id to its running ``Session`` (``None`` when
    it isn't live): live plots and popout re-fits need the kernel. ``output_dir``
    maps an id to the directory its artifacts live in, which may outlive the
    running session. The routes are added in place (not as an included
    sub-router), so they sit in ``target``'s own route table, in order.
    """

    @target.get("/sessions/{session_id}/artifacts/{path:path}")
    def get_artifact(session_id: str, path: str) -> FileResponse:
        out_dir = output_dir(session_id)
        if out_dir is None:
            raise HTTPException(status_code=404, detail="no such session")
        target = _resolve_artifact(out_dir, path)
        if target is None:
            raise HTTPException(status_code=404, detail="no such artifact")
        return FileResponse(target)

    @target.get("/popout/{session_id}")
    def popout_wrapper(session_id: str, route: str, w: int, h: int) -> HTMLResponse:
        """The page a popout window hosts: the live figure, staged and scaled
        under the same presentation contract as the inline canvas stage (see
        ``_POPOUT_WRAPPER_HTML``), so resizing or fullscreening the popup can
        never leave the scene mis-drawn with widgets floating outside it."""
        if _ROUTE_ID_RE.fullmatch(route) is None:
            raise HTTPException(status_code=404, detail="no such plot route")
        if not (100 <= w <= 8192 and 100 <= h <= 8192):
            raise HTTPException(status_code=404, detail="implausible figure size")
        src = f"/live/{quote(session_id)}/viz/{quote(route)}"
        refit = f"/popout/{quote(session_id)}/refit?route={quote(route)}"
        return HTMLResponse(
            _POPOUT_WRAPPER_HTML.replace("__SRC__", src)
            .replace("__REFIT__", refit)
            .replace("__W__", str(w))
            .replace("__H__", str(h))
        )

    @target.post("/popout/{session_id}/refit")
    async def popout_refit(session_id: str, route: str, w: int, h: int) -> Response:
        """Re-fit a popout's live figure to its window, in place: the same fit
        rule as everywhere else, anchored to the record's authored size. The
        wrapper stages the echoed size this answers as JSON. Best-effort: a
        missing session, route, or figure answers 204 and the wrapper's scaled
        presentation stands."""
        from jutul_agent.agent.plot_julia import _plot_id_of, refit_web

        if _ROUTE_ID_RE.fullmatch(route) is None or _sane_size(w, h) is None:
            raise HTTPException(status_code=404, detail="no such plot route")
        session = live_session(session_id)
        if session is None:
            return Response(status_code=204)
        # The popout's route is its record's plot id plus a `--popN` suffix.
        base = re.sub(r"--pop\d+$", "", route)
        payload = next(
            (
                event.payload
                for event in reversed(session.trace.iter_events())
                if event.kind == schema.ARTIFACT and _plot_id_of(event.payload) == base
            ),
            None,
        )
        authored = (payload or {}).get("authored_px") or (payload or {}).get("size_px")
        echoed = None
        with contextlib.suppress(Exception):
            _err, echoed = await refit_web(
                session, route, _fit_target([w, h], authored), _sized(authored)
            )
        if echoed is None:
            return Response(status_code=204)
        return JSONResponse({"width": echoed[0], "height": echoed[1]})

    @target.api_route("/live/{session_id}/{path:path}", methods=["GET", "HEAD"])
    async def live_plot_proxy(session_id: str, path: str, request: Request) -> Response:
        """Reverse-proxy a session's live Bonito plot server through this port.

        A live plot's Bonito server binds a fresh, unpredictable localhost port
        per session; handing the browser that raw ``127.0.0.1:<port>`` URL only
        works when the browser shares this machine's network namespace. Under any
        port-forwarded setup (an SSH tunnel, VS Code/Cursor remote, Docker) only
        this app server's own port is known to whatever is forwarding connections,
        so the raw port is a permanent, unfixable connection-refused. Bonito's
        ``proxy_url`` (set in ``plot_julia_src.web_server_start``) already writes
        every URL it hands the browser as ``/live/<session_id>/...``, so this route
        only needs to strip that prefix and forward to the real port.
        """
        session = live_session(session_id)
        port = session.web_plot_port if session is not None else None
        # A plot's own page, as opposed to its assets: a browser shows what this
        # answers, so a plot that is gone says so in words.
        plot_page = (
            request.method == "GET"
            and path.startswith("viz/")
            and _ROUTE_ID_RE.fullmatch(path.removeprefix("viz/")) is not None
        )
        if port is None:
            if plot_page:
                return _plot_gone(session)
            raise HTTPException(status_code=404, detail="no live plot server for this session")

        import httpx

        # Answer a readiness probe here rather than forwarding it. Bonito renders the
        # figure's app for *any* request reaching its route, HEAD included, and each
        # render leaves behind a WGLMakie screen that nothing removes, because no browser
        # attaches to a HEAD. The canvas polls this route to find out when the live server
        # is up, so a figure that takes a moment collects a fistful of dead screens — and
        # those shadow the real view, since Makie resolves a scene's screen by taking the
        # first backend match, breaking picking and the click-to-select built on it.
        #
        # A readiness check only has to answer "is the server there", which a connection
        # settles without rendering anything.
        if request.method == "HEAD":
            try:
                _reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", port), timeout=5.0
                )
            except (TimeoutError, OSError) as exc:
                raise HTTPException(
                    status_code=502, detail=f"live plot server unreachable: {exc}"
                ) from exc
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return Response(status_code=200)

        hop_by_hop = {"host", "content-length", "connection"}
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in hop_by_hop]
        async with httpx.AsyncClient() as client:
            try:
                upstream = await client.request(
                    request.method,
                    f"http://127.0.0.1:{port}/{path}",
                    params=request.query_params,
                    headers=headers,
                    content=await request.body(),
                    timeout=httpx.Timeout(**_LIVE_HTTP_TIMEOUT),
                )
            except httpx.TransportError as exc:
                # Refused: nothing listens on the port, so Julia restarted since it
                # was bound. (A timeout is a busy Julia, not a dead plot.)
                if plot_page and isinstance(exc, httpx.ConnectError):
                    return _plot_gone(session)
                raise HTTPException(
                    status_code=502, detail=f"live plot server unreachable: {exc}"
                ) from exc
        if plot_page and upstream.status_code == 404:
            return _plot_gone(session)  # the server runs, but no longer has this plot
        strip = {"content-length", "content-encoding", "transfer-encoding", "connection"}
        response_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in strip}
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    @target.websocket("/live/{session_id}/{path:path}")
    async def live_plot_ws_proxy(websocket: WebSocket, session_id: str, path: str) -> None:
        """Pump a live plot's widget websocket (a slider, a field selector) through
        the same reverse proxy as ``live_plot_proxy``, for the same reason: the
        browser has no route to Bonito's own port, only to this one."""
        session = live_session(session_id)
        port = session.web_plot_port if session is not None else None
        if port is None:
            # Refuse before ``accept`` so the upgrade is answered with a plain HTTP
            # error instead of a socket that opens and immediately closes. The plot
            # client's reconnect budgets ~30s of backed-off attempts but resets that
            # budget on every successful open, so accept-then-close would keep a
            # dead session's leftover tab reconnecting forever.
            with contextlib.suppress(RuntimeError, WebSocketDisconnect, OSError):
                await websocket.close(code=1011)
            return

        import websockets
        from websockets.exceptions import ConnectionClosed, WebSocketException

        upstream_url = f"ws://127.0.0.1:{port}/{path}"
        if websocket.url.query:
            upstream_url += f"?{websocket.url.query}"

        # Dial the plot server first and accept the browser only once the upstream
        # handshake has succeeded, for the same reason as the refusal above: an
        # accept before a failed dial reads as progress to the client and resets
        # its retry budget. Dialing first, a gone upstream is a refused handshake
        # the client backs off from and gives up on, while a merely busy one holds
        # the browser in CONNECTING (which the client treats as "wait", not
        # "retry") until the kernel yields. The generous ``open_timeout`` covers a
        # long eval delaying the upstream handshake; the TCP connect itself still
        # fails fast when nothing listens.
        #
        # No keepalive on this hop. The client library's default is to ping the
        # upstream every 20s and tear the connection down when a pong does not come
        # back in another 20s, but the peer here is the Julia process: its plot
        # server's handler tasks share a thread pool with whatever the kernel is
        # computing, so a long solve can leave them unscheduled for minutes. Pinging
        # measures how busy Julia is, not whether the connection is alive, and every
        # timeout kills a working socket that the browser then reconnects, in a loop
        # for as long as the solve runs. The socket itself still reports a plot
        # server that actually goes away.
        try:
            async with websockets.connect(
                upstream_url,
                max_size=None,
                ping_interval=None,
                open_timeout=_LIVE_WS_OPEN_TIMEOUT,
            ) as upstream:
                await websocket.accept(subprotocol=websocket.headers.get("sec-websocket-protocol"))

                async def to_upstream() -> None:
                    try:
                        while True:
                            message = await websocket.receive()
                            if message["type"] == "websocket.disconnect":
                                return
                            data = message.get("bytes")
                            if data is not None:
                                await upstream.send(data)
                                continue
                            text = message.get("text")
                            if text is not None:
                                await upstream.send(text)
                    finally:
                        with contextlib.suppress(Exception):
                            await upstream.close()

                async def from_upstream() -> None:
                    with contextlib.suppress(ConnectionClosed):
                        async for message in upstream:
                            if isinstance(message, bytes):
                                await websocket.send_bytes(message)
                            else:
                                await websocket.send_text(message)

                pumps = {asyncio.create_task(to_upstream()), asyncio.create_task(from_upstream())}
                done, pending = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                # Wait for the cancellation to actually land: otherwise the loser can
                # still be mid-``send`` when we close the client socket below, racing
                # a "send after close" ASGI error instead of a clean cancel. Gather the
                # winner as well, so that a pump which ended by raising has its
                # exception retrieved here: an unretrieved one stays on the task until
                # it is collected and then prints itself as a bare, unattributed
                # traceback, typically long after the connection it belongs to.
                await asyncio.gather(*done, *pending, return_exceptions=True)
        except (WebSocketDisconnect, OSError, WebSocketException):
            # OSError: the dial failed or timed out. WebSocketException: the plot
            # server answered the upgrade with an HTTP error (e.g. a route removed
            # by close_plots while a tab still held it). The close below then
            # refuses the still-unaccepted browser handshake.
            pass
        finally:
            # Closing is best-effort on a peer that may already be gone: starlette
            # turns uvicorn's ClientDisconnected into WebSocketDisconnect (a raw
            # OSError before accept), and a double close raises RuntimeError.
            with contextlib.suppress(RuntimeError, WebSocketDisconnect, OSError):
                await websocket.close()
