"""The loopback guard: other sites' pages cannot drive a local server."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from jutul_agent.interfaces.server.guard import LoopbackGuard, is_loopback


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    app.add_middleware(LoopbackGuard)
    return TestClient(app, base_url="http://127.0.0.1:8742")


def test_is_loopback() -> None:
    for host in (
        "localhost",
        "LOCALHOST",
        "app.localhost",
        "127.0.0.1",
        "127.8.9.1",
        "::1",
        "[::1]",
    ):
        assert is_loopback(host), host
    for host in (None, "", "evil.example", "localhost.evil.example", "0.0.0.0", "192.168.1.10"):
        assert not is_loopback(host), host


def test_local_requests_pass(client: TestClient) -> None:
    assert client.get("/ping").status_code == 200
    assert client.get("/ping", headers={"host": "localhost:8742"}).status_code == 200
    # The app's own pages, or a local host app on another port.
    for origin in ("http://127.0.0.1:8742", "http://localhost:3000"):
        assert client.get("/ping", headers={"origin": origin}).status_code == 200


def test_rebinding_and_cross_site_requests_are_refused(client: TestClient) -> None:
    # DNS rebinding: a foreign name that resolves to loopback.
    assert client.get("/ping", headers={"host": "evil.example:8742"}).status_code == 403
    # A cross-site request, and the opaque origin of a sandboxed frame.
    for origin in ("https://evil.example", "null"):
        response = client.post("/ping", headers={"origin": origin})
        assert response.status_code == 403, origin


def test_websocket_origin(client: TestClient) -> None:
    # An absolute URL: the test client's WebSocket ignores base_url for Host.
    url = "ws://127.0.0.1:8742/ws"
    with client.websocket_connect(url, headers={"origin": "http://localhost:8742"}) as ws:
        assert ws.receive_text() == "hello"
    with (
        pytest.raises(WebSocketDisconnect),
        client.websocket_connect(url, headers={"origin": "https://evil.example"}) as ws,
    ):
        ws.receive_text()
