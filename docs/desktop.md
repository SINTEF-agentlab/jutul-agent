# The desktop app

`jutul-agent desktop` runs the web UI in its own window, as a desktop app, with
**no server**: no process of the app binds, listens on, or accepts a connection
on any TCP port or Unix socket. The only network traffic is the agent's own
calls to the LLM provider.

It is the same UI and the same application as `jutul-agent web`. Only the
transport between the pieces differs, so the two stay in step by construction.

## Running it

```sh
uv sync                              # or: uv pip install -r requirements-desktop.txt
uv run jutul-agent desktop --install # once: Electron, into this Python environment
uv run jutul-agent desktop           # in a folder set up with `jutul-agent init`
```

`jutul-agent desktop` takes `--workspace DIR` and the session options of
`jutul-agent web` (`--sim`, `--model`, `--approval-mode`, `--threads`, ...). The
port options (`--host`, `--port`, `--allowed-origin`, `--base-path`) have no
meaning here.

Electron is installed with npm into `<python env>/share/jutul-agent-desktop`, so
the app lives and dies with the environment and nothing is installed globally.
Installing needs Node.js 22.12 or newer; running does not. `$JUTUL_AGENT_ELECTRON`
points the launcher at another Electron binary.

## Process layout

Every arrow is a pipe or Electron's own IPC.

```
Electron renderer    the web UI (server/web_dist, unmodified) and live-plot iframes
   │  fetch, iframes, downloads   → the jutul:// protocol handler
   │  WebSocket                   → preload shim → Electron IPC
Electron main        electron/main.js, electron/backend.js
   │  frames on the child's stdin/stdout
Python backend       jutul_agent.interfaces.desktop.backend
   │  anonymous pipes inherited at spawn (control + auxiliary)
Julia kernel         juliakernel/server.jl, JutulAgent.JutulAgentPipeViews
```

| Hop | `jutul-agent web` | `jutul-agent desktop` |
| --- | --- | --- |
| UI ↔ application | HTTP + WebSocket to uvicorn | `jutul://` protocol handler and an IPC WebSocket shim, then frames over the backend's stdin/stdout |
| application ↔ Julia | loopback TCP connection | two anonymous pipes inherited by the Julia process |
| live plots | Bonito HTTP server in Julia, reverse-proxied | Bonito pages rendered on request and relayed over a second inherited pipe pair |

### Renderer to backend

The window loads `jutul://localhost/`. `main.js` registers that scheme and
answers every request for it by forwarding it to the backend, so `fetch`,
iframes, images and downloads work as they do against a server. A custom scheme
has no WebSocket upgrade, so `preload.js` replaces `WebSocket` in the page with
one that carries the same messages over Electron IPC (same-origin URLs only;
anything else gets the real `WebSocket`).

The backend is `python -m jutul_agent.interfaces.desktop.backend`. It builds the
application with the same code as `jutul-agent web` (`cli/web.py:build_app`) and
drives it in-process from frames (`asgi_bridge.py`), which is the job uvicorn
does for the web flavour. uvicorn is never imported.

A frame is an 8-byte prefix (two big-endian `uint32`: header length, body
length), a JSON header, and a raw body (`framing.py`, `electron/framing.js`).
The frame kinds are listed at the top of `asgi_bridge.py`.

stdout is the frame channel, so the backend moves it to a private descriptor at
startup and points file descriptor 1 at stderr. A stray `print` then lands in
the log (which the shell shows) instead of corrupting a frame.

### Backend to Julia

`JUTUL_AGENT_IPC=pipe` (set by the backend) switches `juliakernel` to pipe mode.
The parent creates four anonymous pipes, the Julia process inherits one end of
each, and `server.jl` is started with `fd:<read>,<write>` instead of a port. The
control protocol on the pipes is unchanged. The second pair is the kernel's
*auxiliary channel* (`JuliaKernel.aux_streams()`, `ENV["JK_AUX_FDS"]` in Julia).

Pipe mode never falls back to a socket. Where pipes cannot be inherited
(Windows, for now) the kernel refuses to start.

### Live plots

`JutulAgent.JutulAgentPipeViews` serves live figures over the auxiliary channel.
Its `PipeServer` takes `Bonito.route!` / `Bonito.delete_route!` like the
`Bonito.Server` it stands in for, so the code that serves, replaces and closes
figures is shared. On a request it renders the figure's page as one HTML document
with its scripts inlined (`Bonito.NoServer`), connected through a `PipeConnection`
(Bonito's interface for custom connections). `agent/plot_pipe.py` is the backend
end; the application's existing `/live/...` routes relay between it and the
window.

If the pipe cannot be used, plots fall back to the static export. They never
fall back to a port.

## When something crashes

| What dies | What happens |
| --- | --- |
| Python backend | In-flight requests fail and sockets close at once. The backend is restarted with a backoff (0.5 s up to 8 s; given up after five crashes in a row). The page stays, shows a banner, and its own reconnect resumes the session from disk. The Julia state is lost with the backend. |
| Backend never starts | No restart loop: a status page shows the backend's log with *Try again*, *Choose another folder*, and a simulator choice. |
| Julia kernel | Unchanged from the web flavour: the eval fails with how the process ended, and the kernel is restarted on the next use. |
| Renderer | The window reloads. The backend still holds the session and its Julia state. |
| A plot's page or iframe | Its view is closed in Julia after a grace period; reopening the plot renders a new view. |
| Electron itself | The backend's stdin closes, it shuts its sessions down and exits; each Julia process exits when its pipes close. Nothing is orphaned. |

## What still opens a socket

- **The LLM provider**: outgoing connections from the backend. That is the point.
- **MPI**: a simulator that initializes MPI (JutulDarcy does, through HYPRE, at the
  first solve) gets whatever its MPI library opens. MPICH opens a listening TCP
  socket. This is the simulator's own library, in every front end, and outside
  jutul-agent's transport.
- **Distributed workers**: `run_ensemble` uses Julia's `Distributed`, whose
  workers talk to the master over TCP.

To check a running app, list its process tree and run
`lsof -nP -a -p <pids> -i` and `lsof -nP -a -p <pids> -U`.

## Limits

- Pipe mode is POSIX-only. On Windows `jutul-agent desktop` cannot start a
  session yet (it refuses rather than opening a port).
- Updating from the UI is off; use `jutul-agent upgrade`.
- There is no single-instance lock (Electron's is a listening Unix socket), so a
  second launch is a second, independent app.
- Not packaged as an installer; it runs from a Python environment.
