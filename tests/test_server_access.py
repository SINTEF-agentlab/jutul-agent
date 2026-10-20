"""Only the server's own pages may drive it (interfaces/server/access.py)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from jutul_agent.interfaces.server.access import OriginGuard, normalize_origin

SERVER = "http://127.0.0.1:8742"


async def _ok(_request) -> PlainTextResponse:
    return PlainTextResponse("ok")


async def _echo(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_text("hello")
    await websocket.close()


def _client(**guard) -> TestClient:
    app = Starlette(
        routes=[Route("/thing", _ok, methods=["GET", "POST"]), WebSocketRoute("/ws", _echo)]
    )
    return TestClient(OriginGuard(app, **guard), base_url=SERVER)


def _connects(client: TestClient, **headers) -> bool:
    # The test client sends Host: testserver on websockets, whatever its base URL.
    headers.setdefault("host", "127.0.0.1:8742")
    try:
        with client.websocket_connect("/ws", headers=headers) as ws:
            return ws.receive_text() == "hello"
    except WebSocketDisconnect as exc:
        assert exc.code == 1008
        return False


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        (SERVER, True),  # its own page
        (None, True),  # a script or another program
        ("https://evil.example", False),  # another site in the same browser
        ("http://127.0.0.1:9999", False),  # another local server's page
        ("null", False),  # a sandboxed frame or a file:// page
    ],
)
def test_websockets_only_from_the_servers_own_page(origin, allowed) -> None:
    headers = {"origin": origin} if origin else {}
    assert _connects(_client(), **headers) is allowed


def test_requests_that_change_something_need_the_servers_own_page() -> None:
    client = _client()
    assert client.post("/thing", headers={"origin": SERVER}).status_code == 200
    assert client.post("/thing").status_code == 200
    refused = client.post("/thing", headers={"origin": "https://evil.example"})
    assert refused.status_code == 403
    # Another site can't read the answer to a GET, so it isn't refused.
    assert client.get("/thing", headers={"origin": "https://evil.example"}).status_code == 200


def test_a_local_server_answers_to_localhost_only() -> None:
    # A site that points its own domain at 127.0.0.1 is "same origin" with itself.
    client = _client(local_only=True)
    rebound = {"host": "evil.example:8742", "origin": "http://evil.example:8742"}
    assert client.get("/thing", headers=rebound).status_code == 403
    assert not _connects(client, **rebound)
    for host in ("127.0.0.1:8742", "localhost:8742", "[::1]:8742"):
        assert client.get("/thing", headers={"host": host}).status_code == 200


def test_an_allowed_origin_may_use_the_server() -> None:
    client = _client(local_only=True, allowed_origins=["http://jutul.test:8742/"])
    page = {"host": "jutul.test:8742", "origin": "http://jutul.test:8742"}
    assert client.post("/thing", headers=page).status_code == 200
    assert _connects(client, **page)
    # From a page on another origin, with this server's own host.
    other = _client(allowed_origins=["https://app.example.org"])
    assert _connects(other, origin="https://app.example.org")


def test_behind_a_reverse_proxy() -> None:
    # The proxy keeps the public Host: the page is the server's own.
    client = _client()
    public = {"host": "sso.example.org", "origin": "https://sso.example.org"}
    assert _connects(client, **public)
    # The proxy rewrites Host, and passes the original on.
    forwarded = {
        "host": "127.0.0.1:8742",
        "x-forwarded-host": "sso.example.org",
        "origin": "https://sso.example.org",
    }
    assert _connects(client, **forwarded)
    assert not _connects(client, **{**forwarded, "origin": "https://evil.example"})


@pytest.mark.parametrize(
    ("value", "normalized"),
    [("https://App.Example.org/", "https://app.example.org"), (SERVER, SERVER)],
)
def test_origin_values(value, normalized) -> None:
    assert normalize_origin(value) == normalized


@pytest.mark.parametrize("value", ["app.example.org", "https://example.org/app", "ftp://x"])
def test_bad_origin_values_fail_at_launch(value) -> None:
    from jutul_agent.interfaces.cli import web

    with pytest.raises(ValueError):
        normalize_origin(value)
    with pytest.raises(SystemExit):
        web.build_parser().parse_args(["--allowed-origin", value])


def test_the_server_is_guarded(tmp_path) -> None:
    from jutul_agent.interfaces.server.app import create_app
    from jutul_agent.interfaces.server.manager import SessionManager

    async def unused_factory(**_kwargs):
        raise AssertionError("no session should start")

    app = create_app(SessionManager(host_factory=unused_factory), default_sim="jutuldarcy")
    with TestClient(app) as client:
        response = client.post("/sessions", json={}, headers={"origin": "https://evil.example"})
        assert response.status_code == 403
        stream = "/sessions/2026-01-01-0000-abcd/stream"
        with pytest.raises(WebSocketDisconnect) as refused:  # noqa: SIM117
            with client.websocket_connect(stream, headers={"origin": "https://evil.example"}):
                pass
        assert refused.value.code == 1008
