# makker demo app

A small, runnable example of extending the agent: a custom simulator and tools
added through the extension seam, served with the bundled web UI. Copy it as a
starting point for wiring in your own simulator and tools, or for embedding the
agent in another application. The wire contract is documented in
[docs/server-interface.md](../../docs/server-interface.md).

What it demonstrates:

- **A custom simulator**, `DemoSim`, added as an adapter (not via the built-in
  registry), with its own Julia environment.
- **Capability composition**: a web `Capability` adds two tools and a prompt
  fragment through the extension seam.
- **An interactive plot.** The `plot_response` tool renders a WGLMakie figure and
  exports it to self-contained interactive HTML with `Bonito.export_static`. The
  agent records it as an artifact, and the bundled UI pins it in the canvas. This
  is the fiddly part to get right, so it is the main thing to copy.
- **A UI action.** The `set_param` tool emits a `ui` action. The bundled UI shows
  it as a note; a host app embedding the agent can instead apply it to its own
  controls by setting `window.onMakkerUi`.

## Run it

The web stack ships in the core install, so just start the app:

```bash
python examples/demo-app/demo.py       # first run instantiates the Julia env
```

Open <http://127.0.0.1:8742>. Pick a model if needed (any provider key in your
environment), then try: *"plot the response for p=3"*, or *"set the parameter to 5
and plot it"*. The interactive plot appears in the canvas on the right.

## Layout

- `DemoSim/` is a tiny Julia library: one function, `response(p)`, no physics.
- `julia_env/` is the Julia environment (DemoSim, WGLMakie, Bonito).
- `demo.py` is the whole example: the adapter, the web capability (the
  `plot_response` and `set_param` tools), and the server wiring.

## Extending it

- Swap `DemoSim` for a real simulator: point the adapter at it and add it to the
  Julia env.
- Add tools, skills, or subagents by extending the `Capability` in `demo.py`, or
  publish your own under the `makker.extensions` entry point.
- Add a new canvas surface (a map, a custom chart) by registering a panel in the
  web UI; see [docs/web-ui.md](../../docs/web-ui.md).
