# The web UI

`jutul-agent`, run with no command, serves a browser interface for a session
(`jutul-agent web` is the same command). It is the default way to work with the
agent, and the surface most users and extensions build on. It speaks the same [server interface](server-interface.md) any
front end would, and reuses the same agent core as the command line and the
terminal UI.

## Running it

Start it from a simulator workspace:

```sh
jutul-agent
```

This serves the UI at `http://127.0.0.1:8742` by default. Open that address in a
browser. The first session in a folder builds the simulator's Julia environment,
which can take a few minutes; later sessions start in seconds. Use `--port` to
change the port and `--sim` to pick the simulator; see the
[CLI reference](cli.md) for the full set of options.

## The layout

The interface has two panes.

The conversation is on the left. You type a prompt, the agent runs the simulator,
writes and runs Julia, and streams its work back as text, collapsible reasoning,
and tool cards (the code it ran, file edits as diffs, a plan as a checklist, and
captured output).

The canvas is on the right. Interactive plots and written reports are pinned here
as tabs, so results stay visible while the conversation continues. The canvas opens
when the first view is produced. You can switch tabs, resize the split, pop a view
out into its own browser tab, or close the panel and reopen it from the top bar.

Type `/` in the message box for commands (switch the model, set the approval
policy, compact the conversation, download a transcript, and more). Past sessions
are listed in the left sidebar and reopen where you left off.

The foot of the sidebar names the jutul-agent and simulator versions. When a new
release or an environment update is due, a link there opens the update dialog,
which runs the update and reloads the page when it is done (see
[Upgrading](usage.md#upgrading)).

## How it is built and shipped

The UI is a React and TypeScript application, built with Vite. The source lives in
`src/jutul_agent/interfaces/server/webapp`. It is compiled ahead of time into
`src/jutul_agent/interfaces/server/web_dist`, which is committed and shipped in the
package. Users install with `uv` or `pip` and never need Node; the server serves
the prebuilt bundle.

Contributors who change the UI rebuild that bundle:

```sh
cd src/jutul_agent/interfaces/server/webapp
npm install        # first time only
npm run build      # writes ../web_dist
npm test           # unit and component tests (vitest)
```

The webapp README has the full development workflow, including a dev server with
hot reload that proxies to a running backend.

## Extending the canvas

The canvas is the main place to extend the UI. Each pinned view has a `kind`
(`plot`, `report`, and so on), and the canvas looks up a panel component for that
kind. The built-in kinds render in an iframe (live plots and reports) or as an
image. A new surface registers its own panel:

```ts
import { registerPanel } from "./canvas/registry";

registerPanel("map", MapPanel);
```

`MapPanel` is an ordinary React component that receives the view and renders it.
For example, a MapLibre map that places geothermal wells for Fimbul would be a
`map` panel. The agent emits a view of that kind over the
[wire protocol](server-interface.md) (a `viz` message carries the `kind`, and a
tool can drive the panel through `ui` actions), and the canvas mounts the
registered panel. The canvas core does not change.

This keeps an extension small and self-contained: a Julia or Python tool on the
backend produces the data, a wire `kind` names the surface, and a registered panel
draws it.

## Producing an interactive plot from a tool

A view does not always need a custom panel. The simplest interactive plot is a tool
that renders a Makie figure and writes it as self-contained HTML, which the canvas
shows in an iframe. Getting the WGLMakie and Bonito wiring right is the fiddly part,
so the demo example is the reference: its `plot_response` tool renders a WGLMakie
figure and exports it with `Bonito.export_static`, then records it as an artifact
that the canvas pins automatically. See `examples/demo-app`. The built-in
`plot_julia` tool serves a live, fully interactive figure the same way, kept
offscreen so no native window opens.
