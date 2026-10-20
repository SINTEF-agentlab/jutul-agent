"""Which browser pages may use this server.

A page that can drive a session can run code on this computer: the agent runs
Julia and shell commands. Browsers let any site send requests to 127.0.0.1 and,
without CORS getting in the way, open WebSockets to it. So the server itself
turns away pages that aren't its own:

- A request from a browser page carries ``Origin``. A WebSocket handshake, or a
  request that changes something (anything but GET, HEAD and OPTIONS), is only
  answered when that origin is this server's: the ``Host`` the request was sent
  to, the ``X-Forwarded-Host`` a reverse proxy passed on, or an origin allowed
  with ``--allowed-origin``. Another site can't read the answer to a GET, since
  the server sends no CORS headers.
- A server that serves this computer only also checks ``Host`` on every request:
  it must be a loopback name, or the host of an allowed origin. Otherwise a site
  could point its own domain at 127.0.0.1 and pass as the server's own page.

Requests without ``Origin`` (scripts and other programs) are not affected: they
are not run by a page someone else wrote. The checks run as ASGI middleware, so
they cover every route and WebSocket, including the live-plot proxy.
"""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Iterable, Mapping
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_log = logging.getLogger("uvicorn.error")


def is_loopback_host(host: str) -> bool:
    """Whether a bind address or URL host name only reaches this computer."""
    host = host.strip("[]")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _hostname(host: str) -> str:
    """The name part of a ``Host`` header (``[::1]:8742`` gives ``::1``)."""
    return urlsplit(f"//{host}").hostname or ""


def normalize_origin(origin: str) -> str:
    """``scheme://host[:port]`` in lower case, as browsers send it.

    Raises ``ValueError`` for anything else, so a typo in ``--allowed-origin``
    fails at launch rather than refusing the page it meant to let in.
    """
    parts = urlsplit(origin.strip().rstrip("/"))
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"not an origin: {origin!r} (expected e.g. https://example.org)")
    if parts.path or parts.query or parts.fragment:
        raise ValueError(f"an origin has no path: {origin!r}")
    return f"{parts.scheme}://{parts.netloc}".lower()


def request_is_local(headers: Mapping[str, str]) -> bool:
    """Whether a request comes from this server's own page on this computer.

    Stricter than the middleware, for the few requests only someone at this
    computer should make (updating jutul-agent): the page must be the server's
    own, and served under a loopback name.
    """
    host = headers.get("host", "")
    origin = headers.get("origin")
    if not host or not origin:
        return False
    return is_loopback_host(_hostname(host)) and urlsplit(origin).netloc == host


class OriginGuard:
    """ASGI middleware applying the checks in the module docstring."""

    def __init__(
        self, app: ASGIApp, *, local_only: bool = False, allowed_origins: Iterable[str] = ()
    ) -> None:
        self.app = app
        # Check Host too: the server answers to this computer only.
        self.local_only = local_only
        self.allowed = {normalize_origin(origin) for origin in allowed_origins}
        self.allowed_hosts = {urlsplit(origin).netloc for origin in self.allowed}
        self._reported: set[str] = set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        reason = self.refusal(scope["type"], scope.get("method", ""), Headers(scope=scope))
        if reason is None:
            await self.app(scope, receive, send)
        elif scope["type"] == "websocket":
            # Closing before accepting answers the handshake with a 403.
            await receive()
            await send({"type": "websocket.close", "code": 1008})
        else:
            await PlainTextResponse(reason, status_code=403)(scope, receive, send)

    def refusal(self, kind: str, method: str, headers: Headers) -> str | None:
        """Why this request is refused, or ``None`` to let it through."""
        host = headers.get("host", "").lower()
        if self.local_only and host and not self._host_ok(host):
            self._report(
                f"host {host}",
                f"Refused a request for {host}: this server only answers to localhost. "
                "Open it at 127.0.0.1, or start it with --allowed-origin for that address.",
            )
            return "this server only answers to localhost"
        origin = headers.get("origin")
        if origin is None or (kind == "http" and method in _SAFE_METHODS):
            return None
        if self._origin_ok(origin.lower(), host, headers.get("x-forwarded-host", "")):
            return None
        self._report(
            origin,
            f"Refused a request from {origin}, which isn't this server's own page. If it "
            f"should use this server, start jutul-agent with --allowed-origin {origin}",
        )
        return "requests from other sites are refused"

    def _host_ok(self, host: str) -> bool:
        return is_loopback_host(_hostname(host)) or host in self.allowed_hosts

    def _origin_ok(self, origin: str, host: str, forwarded_host: str) -> bool:
        if origin in self.allowed:
            return True
        netloc = urlsplit(origin).netloc
        # A page's own requests name its host; behind a proxy that rewrites Host,
        # the original rides in X-Forwarded-Host (the first one, if chained). A
        # browser page can't set either header on another site's request.
        forwarded = forwarded_host.split(",")[0].strip().lower()
        return bool(netloc) and netloc in (host, forwarded)

    def _report(self, key: str, message: str) -> None:
        """Say once per origin or host why its requests are refused."""
        if key not in self._reported:
            self._reported.add(key)
            _log.warning(message)
