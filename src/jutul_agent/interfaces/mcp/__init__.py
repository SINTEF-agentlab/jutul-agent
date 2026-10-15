"""The MCP interface: one session's Julia tools, served to an MCP client.

``jutul-agent mcp`` runs an MCP server over stdio for an IDE or desktop agent
that brings its own model, chat, and approvals. It serves the session's tools
(``build_tools``, the list the bundled agent gets), streams Julia's output as
progress, and links plots and reports to a loopback viewer (the web server's
viewer routes) where a live figure stays interactive.

``runtime`` owns the session, ``server`` maps it to the protocol, and
``loopback`` serves the viewer.
"""
