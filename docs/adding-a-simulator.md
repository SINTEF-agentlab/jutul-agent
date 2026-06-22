# Adding a simulator

A simulator is a folder of data under `src/jutul_agent/simulators/`, plus one
registry entry. The agent code does not change.

```
simulators/<name>/
  __init__.py         # exports the adapter
  adapter.py          # SimulatorAdapter instance
  julia_env/          # Project.toml + JutulAgent<Sim>/ warm package
  skills/             # one folder per skill, each with a SKILL.md
```

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
)
```

`module_dir` anchors the convention: the base class derives
`julia_env_template_path` and `skills_dir` from it. `package_imports` is what
the agent is told it can `using`, and `primary_package` is what `doctor`
checks is actually resolved in the env. Adapters can also contribute simulator
subagents through `subagent_factories`.

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
- A `@compile_workload` that runs the smallest representative solve and a
  plot save.

This is what makes the difference between a first solve in seconds and one in
minutes. Set the adapter's `warm_package` to its name, and it is loaded in
the background at session start.

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

## 5. Add the Simulator

### Built-in Simulator (Development)

Add your simulator folder to the `src/jutul_agent/simulators` directory

Add the simulator to the `simulators.yml` matrix. It instantiates the env
template and runs the simulator's integration smoke on PRs and weekly, which
catches upstream releases that break the template.

### Local Simulator

Add the simulator to your jutul-agent by using the `jutul-agent simulator add` command.
This will create a symlink between your installation of jutul-agent and the simulator folder,
such that if you change any of the files in `/path/to/simulator`, jutul-agent automatically gets updated
```
jutul-agent simulator add /path/to/simulator
```

## Trying it



```sh
mkdir try-mysim && cd try-mysim
uv run jutul-agent init --sim mysim --precompile
uv run jutul-agent
```

To develop against a local checkout of the simulator package:

```sh
uv run jutul-agent init --sim mysim --source-path /path/to/MySim.jl
```

The checkout resolves at its real path (writable, since it is a dev checkout
rather than a shared-depot install), so the agent can read and edit the
package source itself.
