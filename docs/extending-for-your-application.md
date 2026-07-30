# Building your application on jutul-agent

You can build a complete application on top of jutul-agent, with its own
simulator, skills, tools, and web front end, **without forking jutul-agent or
editing any of its code**. Your project is a normal Python package that depends
on `jutul-agent` and registers what it adds through entry points. Upgrading
jutul-agent then never costs you a merge.

There are three things you can contribute, independently or together:

| You want to add | Mechanism |
| --- | --- |
| Your own simulator (adapter, skills, Julia env) | a `jutul_agent.simulators` entry point |
| Tools, skills, subagents, or prompt text (per simulator and per interface) | a `jutul_agent.extensions` entry point (a `Capability`) |
| A web app for your users | a front end over the documented WebSocket/REST protocol |

The rest of this page shows each one. The agent loop, the server, the Julia
kernel, and the wire protocol all stay in jutul-agent; you only add data and
small pieces of glue.

## Add your own simulator

A simulator is a folder with an adapter, skill markdown, and a Julia environment
template (see [adding a simulator](adding-a-simulator.md) for the folder's
contents). To ship one from your own package, lay it out the same way inside your
package and publish its `SimulatorAdapter` under the `jutul_agent.simulators`
entry-point group:

```
mypackage/
  mysim/
    adapter.py        # MYSIM = SimulatorAdapter(name="mysim", module_dir=Path(__file__).parent, ...)
    skills/           # your SKILL.md folders
    julia_env/        # Project.toml (+ optional warm package)
```

```toml
# in your pyproject.toml
[project.entry-points."jutul_agent.simulators"]
mysim = "mypackage.mysim.adapter:MYSIM"
```

Install your package alongside jutul-agent and `mysim` appears everywhere the
built-in simulators do (the registry, `--sim mysim`, the server's
`/simulators`). The adapter's `module_dir` is what anchors your `skills/` and
`julia_env/`, so they are read from your package. The shared `JutulAgent` Julia
runtime is copied from the jutul-agent install automatically, so your
`julia_env/Project.toml` only needs your simulator's packages plus the
`[sources]` entries (copy them from a bundled simulator).

**Building on an existing simulator.** Because an installed adapter overrides a
bundled one of the same `name`, you can publish a customised `jutuldarcy` (your
own skills and Julia config, based on JutulDarcy) and it replaces the built-in
one. Or give it a new name to offer it alongside.

## Add tools, skills, subagents, and prompt text

How a simulator is used differs by interface: a web app might let the agent drive
controls or a diagram that a terminal never shows. So the unit of customization
is a `Capability` (see [the server interface](server-interface.md#capabilities)),
which can contribute tools, skill directories, subagents, and a prompt fragment,
optionally scoped to a surface (`tui`, `web`, `cli`).

Publish a `Capability` (or a function returning one) under the
`jutul_agent.extensions` entry-point group:

```python
# mypackage/capability.py
from jutul_agent.agent.capabilities import Capability


def my_capability() -> Capability:
    return Capability(
        name="mysim-web",
        tools=(make_my_tool,),  # factories taking the Session
        skill_dirs=(("/abs/path/skills", "MySim web"),),
        prompt_fragment="How to use the app's controls...",
        surfaces=("web",),  # only on the web surface; omit for all
    )
```

```toml
[project.entry-points."jutul_agent.extensions"]
mysim_web = "mypackage.capability:my_capability"
```

Installed capabilities are discovered automatically and composed into the agent
alongside the base tools and the active simulator. Two further options exist for
specific needs:

- **UI-control tools** emit a `ui` command that your front end applies (move a
  slider, highlight a node, pan a map), and the front end sends `ui_event`s back
  that the agent reads. The envelope is generic; the actions are yours.
- **Declarative HTTP tools** let an application in any language expose its
  routines: send tool specs (name, description, schema, endpoint) when you create
  a session and the agent gets tools that call them. No Python plug-in needed.

## Build your web app

The server exposes a stable HTTP + WebSocket contract (see
[the server interface](server-interface.md)). Your front end can be anything that
speaks it. Two starting points:

- **Use the bundled UI.** `jutul-agent web --sim mysim` serves a complete chat
  interface. Good for getting going and for internal tools.
- **Build your own.** Code against the protocol with whatever stack your app
  needs (React, Svelte, MapLibre, ...). The bundled UI and the `examples/`
  app are reference implementations to copy.

Interactive plots come back over the protocol as a `viz` (a self-contained
WebGL/WGLMakie HTML you embed) and as image artifacts; the agent builds the
figures, so they work for any simulator.

## What you never touch

You do not edit jutul-agent's registry, builder, prompts, server, or kernel.
Adding a simulator is an entry point and a folder; adding behavior is an entry
point and a `Capability`; adding a UI is code against the protocol. That is the
seam, and it is the supported way to build a product on top of jutul-agent.
