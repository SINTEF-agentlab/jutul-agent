"""Refuse browser requests to a loopback server that came from other sites."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit

from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose


def is_loopback(host: str | None) -> bool:
    """Whether ``host`` (a hostname or IP literal, without port) names this machine."""
    if not host:
        return False
    host = host.strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class LoopbackGuard:
    """ASGI middleware for a server bound to loopback.

    Binding to loopback keeps other machines out, but not the pages open in the
    user's own browser. Two routes remain, and this closes both. DNS rebinding: a
    page whose own hostname resolves to 127.0.0.1, so its requests carry that
    foreign name in ``Host``. Cross-site requests, notably WebSocket handshakes,
    which the same-origin policy does not restrict, and which carry the foreign
    page in ``Origin``. Requests without an ``Origin`` (the CLI, scripts) and
    pages served from any loopback port pass.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and not self._allowed(scope):
            if scope["type"] == "websocket":
                await WebSocketClose(code=1008)(scope, receive, send)
            else:
                await PlainTextResponse("Forbidden: not a local request", 403)(scope, receive, send)
            return
        await self.app(scope, receive, send)

    @staticmethod
    def _allowed(scope: Scope) -> bool:
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", ())
        }
        if not is_loopback(urlsplit(f"//{headers.get('host', '')}").hostname):
            return False
        origin = headers.get("origin")
        return origin is None or is_loopback(urlsplit(origin).hostname)
