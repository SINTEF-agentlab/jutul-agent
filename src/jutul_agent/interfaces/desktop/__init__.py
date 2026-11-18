"""The desktop interface: the web UI in an Electron window, with no HTTP server.

The browser UI and the FastAPI application are the ones ``jutul-agent web``
serves; only the transport differs. Nothing here listens on a network port:

- The Electron main process spawns ``python -m jutul_agent.interfaces.desktop.backend``
  and speaks length-prefixed frames with it over the child's stdin/stdout
  (``framing.py``). The backend drives the same ASGI app in-process
  (``asgi_bridge.py``), so every route and the per-session stream behave exactly
  as they do behind uvicorn.
- The backend reaches each Julia kernel over anonymous pipes the Julia process
  inherits at spawn, and the kernel serves live plots over a second such pipe
  (``juliakernel/config.py`` holds the switch; ``agent/plot_pipe.py`` and
  ``JutulAgent.JutulAgentPipeViews`` are the two ends of the plot pipe).

No process binds, listens or accepts: there is no TCP port and no Unix socket.

The only network traffic left is the agent's own: the LLM provider.

See ``docs/desktop.md`` for the process layout and the frame protocol.
"""

from __future__ import annotations
