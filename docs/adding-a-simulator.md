# Adding a simulator

A simulator is a self-contained folder of data: an adapter, a Julia environment
template, and skills. Nothing else changes, because the registry discovers the
folder on its own and the agent code reads it generically.

```
<name>/
  __init__.py         # exports the adapter
  adapter.py          # SimulatorAdapter instance
  julia_env/          # Project.toml + JutulAgent<Sim>/ warm package
  skills/             # one folder per skill, each with a SKILL.md
```

The folder goes in one of two places, depending on who it is for:

- **In-tree**, under `src/jutul_agent/simulators/<name>/` in a checkout of this
  repo, to contribute a simulator to jutul-agent itself. The rest of this page
  follows this route.
- **In your own package**, to add a simulator for your own use without forking.
  The folder is laid out the same way; you publish its adapter under an entry
  point so an install discovers it. See
  [building your application](extending-for-your-application.md).

Either way the registry finds the adapter automatically, so there is no list to
edit and no core code to change.

## 1. The adapter

`adapter.py` declares everything the harness needs to know:

```python
from pathlib import Path
from jutul_agent.simulators.base import SimulatorAdapter

MYSIM = SimulatorAdapter(
    name="mysim",
    display_name="MySim",
    module_dir=Path(__file__).resolve().parent,
    package_imports=("Jutul", "MySim"),
    primary_package="MySim",
    domain_hints="What the simulator is for, in one or two sentences.",
    warm_package="JutulAgentMySim",
    example_prompts=("Run a small case and show me the result.", ...),
)
```

`module_dir` anchors the convention: the base class derives
`julia_env_template_path` and `skills_dir` from it. `package_imports` is what
the agent is told it can `using`, and `primary_package` is what `doctor`
checks is actually resolved in the env. `example_prompts` are starter tasks the
web UI offers on its welcome screen (optional; lead with a simulate-and-plot
one). Adapters can also contribute simulator subagents through
`subagent_factories`.

That is all the registration needed: dropping the folder in place is enough for
the registry to find it.

## 2. The Julia environment template

`julia_env/Project.toml` lists the packages a workspace gets. Keep compat
loose and do not commit a `Manifest.toml`, so that the env resolves when
the workspace is instantiated. Declare the shared `JutulAgent` package as a
relative `[sources]` path dependency (copy the entry from an existing
simulator). Its single source lives in `src/jutul_agent/julia_runtime/` and
is synced into the env at bootstrap.

The kernel itself needs no dependency in the env. Its Julia server is
standard-library only.

## 3. The warm package

`julia_env/JutulAgentMySim/` is a small Julia package whose only job is
precompilation. Start from an existing simulator's and adapt the workload:

- `@recompile_invalidations` around the imports, so the simulator and the
  plotting stack are compiled together.
- Every Makie backend a session can hold at once (`WGLMakie` as well as
  `GLMakie`), imported inside that block.
- A `_warm()` function running the smallest representative solve and the
  plotters your skills teach, each figure saved through GLMakie *and*
  CairoMakie (the plot tool builds with one and writes its durable record with
  the other, and neither is warm from the other).
- A `@compile_workload` whose whole body is `try _warm() catch end`.

This is what makes the difference between a first solve in seconds and one in
minutes, and the same for the first plot. Set the adapter's `warm_package` to
its name, and it is loaded in the background at session start.

Three things are easy to get wrong here, and all of them fail silently:

- **Bake the plotter the skills teach, not a cheaper relative of it.** The
  workload only warms the call it actually makes. A skill that teaches an
  interactive explorer is not served by baking the plain mesh plotter
  underneath it.
- **Activate GLMakie before building a figure.** A Makie backend activates
  itself on load, so whichever you imported last is current, and the
  interactive plotters refuse to build under a non-interactive one.
- **Bake in the world the session runs in.** Loading a Makie backend
  invalidates code specialised against the backends already loaded, and that
  reaches well past plotting into the solver. A backend the session loads
  later than the warm package therefore discards part of the bake, solve
  included, so the warm package has to load them all up front.

The `try` is required (a context-less precompile has no GL and must still bake
the solve), which is exactly why the workload lives in `_warm()`:
`tests/test_simulators_smoke.py` calls it directly, without the swallow, so a
workload that has stopped baking anything fails a test instead of quietly
making every first call slow again.

## 4. Skills

Add at least `skills/<name>-overview/SKILL.md`: what the package does, the
canonical entry points, the standard workflow, where the examples live. Write
for a model that reads the package source at its real `pkgdir` path and has a
live REPL. Point at things to read and probe rather than duplicating the
documentation. See [improving the agent](improving-the-agent.md) for how
skills are surfaced and when to use a skill versus the system prompt.

Frontmatter is YAML and must parse (quote a `description:` that contains a
colon). A repo test checks every skill, and a malformed one is skipped at
runtime with only a warning.

## 5. CI

Add the simulator to the `simulators.yml` matrix. It instantiates the env
template and runs the simulator's integration smoke on PRs and weekly, which
catches upstream releases that break the template.

## Trying it

```sh
mkdir try-mysim && cd try-mysim
uv run jutul-agent init --sim mysim
uv run jutul-agent web
```

To develop against a local checkout of the simulator package:

```sh
uv run jutul-agent init --sim mysim --source-path /path/to/MySim.jl
```

The checkout resolves at its real path (writable, since it is a dev checkout
rather than a shared-depot install), so the agent can read and edit the
package source itself.
