"""Julia source generation for the plotting tools.

``web_server_start`` is the one place two connectivity bugs can hide:

- A port-mismatch: Bonito silently retries on ``port+1``, ``port+2``, ... when
  the requested port is already taken (a real risk given ``_free_port()``'s
  bind-then-release race), and updates the server object's own ``.port`` to
  wherever it actually bound. If the generated code echoed the requested port
  back instead of reading ``.port``, every plot's ``live_url`` for the rest of
  the session would point at a dead port.
- An unreachable raw port: the browser is handed Bonito's own ephemeral port
  directly unless ``proxy_url`` is set, which breaks under any port-forwarded
  setup (SSH tunnel, VS Code/Cursor remote, Docker) where only the app's own
  port is known to whatever is forwarding ports -- a persistent "connection
  refused" no client-side retry can recover from, since the port genuinely
  isn't reachable from where the browser sits.
"""

from __future__ import annotations

from jutul_agent.agent.plot_julia_src import web_server_start


def test_web_server_start_reports_the_actual_bound_port() -> None:
    code = web_server_start(12345, "2026-01-01-0000-abcd")
    assert "__JUTUL_WEB_PORT__ = __JUTUL_WEB_SERVER__.port" in code
    assert "__JUTUL_WEB_PORT__ = 12345" not in code


def test_web_server_start_sets_a_site_relative_proxy_url() -> None:
    code = web_server_start(12345, "2026-01-01-0000-abcd")
    assert 'proxy_url = raw"/live/2026-01-01-0000-abcd/"' in code
