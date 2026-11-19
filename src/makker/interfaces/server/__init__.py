"""The server interface: drive an agent session over HTTP and WebSocket.

This lets a webapp, or any other graphical application, embed makker as a
conversational driver. The point of the package is that the server reuses the
same session core the CLI and TUI run on, rather than reimplementing the agent,
so the agent behaves identically across every front end.

A front end depends only on the documented HTTP and WebSocket contract (see
docs/server-interface.md). How this package is organised behind that contract is
an implementation detail and may change.
"""

from __future__ import annotations
