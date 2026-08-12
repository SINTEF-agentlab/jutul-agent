"""The server serves the built React UI (web_dist) with correct asset MIME types.

These guard the bundled web UI: that the pre-built single-page app is served at
``/``, and that its JavaScript bundle is served with a JS MIME type (a browser
refuses a ``type="module"`` script otherwise, and on Windows the registry often
maps ``.js`` to ``text/plain``).
"""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from jutul_agent.interfaces.server.app import WEB_DIST_DIR, _ui_dir, create_app


def test_built_app_is_present_and_served() -> None:
    assert (WEB_DIST_DIR / "index.html").is_file(), (
        "run `npm run build` in interfaces/server/webapp"
    )
    assert _ui_dir() == WEB_DIST_DIR


def test_serves_the_single_page_app() -> None:
    with TestClient(create_app(ui=True)) as client:
        body = client.get("/").text
        assert '<div id="root">' in body
        match = re.search(r'assets/index-[^"\']+\.js', body)
        assert match, "index.html should reference the hashed JS bundle"
        resp = client.get("/" + match.group(0))
        assert resp.status_code == 200
        # The module script must be served as JavaScript or the browser won't run it.
        assert resp.headers["content-type"].startswith("text/javascript")


def test_no_ui_mount_when_disabled() -> None:
    with TestClient(create_app(ui=False)) as client:
        # API routes still work; the SPA is just not mounted at "/".
        assert client.get("/models").status_code == 200
        assert client.get("/").status_code == 404


def test_popout_wrapper_stages_the_live_route() -> None:
    # The popup's wrapper page embeds the live route at the figure's real size
    # and scales it to the window; it never echoes an arbitrary route string.
    with TestClient(create_app(ui=False)) as client:
        resp = client.get("/popout/sid-1", params={"route": "res--pop2", "w": 1600, "h": 900})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert 'src="/live/sid-1/viz/res--pop2"' in resp.text
        assert "let W = 1600, H = 900;" in resp.text
        # The wrapper scales down into a small window; when the scaled figure
        # covers the window poorly it asks the kernel to re-fit and stages the
        # size the server *echoes*, never the size it asked for.
        assert "scale(" in resp.text
        assert "coverage" in resp.text
        assert "W = d.width; H = d.height;" in resp.text
        assert "/popout/sid-1/refit?route=res--pop2" in resp.text

        # The refit endpoint validates like the wrapper and is 204 best-effort
        # (no session here — the fallback presentation stands).
        assert (
            client.post(
                "/popout/sid-1/refit", params={"route": "res--pop2", "w": 1200, "h": 900}
            ).status_code
            == 204
        )
        assert (
            client.post(
                "/popout/sid-1/refit", params={"route": "../x", "w": 1200, "h": 900}
            ).status_code
            == 404
        )

        # A route id that is not a plot route (path tricks, markup) 404s.
        for bad in ("../../etc", "a/b", "<script>", "x" * 80):
            assert (
                client.get("/popout/sid-1", params={"route": bad, "w": 800, "h": 600}).status_code
                == 404
            )
        # As does an implausible size.
        assert (
            client.get("/popout/sid-1", params={"route": "res", "w": 9, "h": 600}).status_code
            == 404
        )
