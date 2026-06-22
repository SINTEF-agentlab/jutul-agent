# jutul-agent

[![CI](https://github.com/SINTEF-agentlab/jutul-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/SINTEF-agentlab/jutul-agent/actions/workflows/ci.yml)
[![Simulators](https://github.com/SINTEF-agentlab/jutul-agent/actions/workflows/simulators.yml/badge.svg)](https://github.com/SINTEF-agentlab/jutul-agent/actions/workflows/simulators.yml)
[![Docs](https://img.shields.io/badge/docs-website-0f4c5c)](https://sintef-agentlab.github.io/jutul-agent/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A scientific AI agent for differentiable simulators built on the
[Jutul](https://github.com/sintefmath/Jutul.jl) framework. Ask for a
simulation in plain language. The agent sets it up, runs it, analyses and
plots the results, and iterates: fixing mistakes and refining the next run.

<p align="center">
  <a href="https://github.com/sintefmath/JutulDarcy.jl"><img src="https://raw.githubusercontent.com/sintefmath/JutulDarcy.jl/main/docs/src/assets/logo_wide.png" alt="JutulDarcy" height="42"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/BattMoTeam/BattMo.jl"><img src="https://raw.githubusercontent.com/BattMoTeam/BattMo.jl/main/docs/src/assets/battmologo_text.png" alt="BattMo" height="74"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/sintefmath/Fimbul.jl"><img src="https://raw.githubusercontent.com/sintefmath/Fimbul.jl/main/docs/src/assets/logo_text_wide.png" alt="Fimbul" height="50"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/sintefmath/Mocca.jl"><img src="https://raw.githubusercontent.com/sintefmath/Mocca.jl/main/docs/src/assets/mocca_small_transparent.png" alt="Mocca" height="62"></a>
</p>

What makes it work for scientific computing:

- A persistent Julia REPL per session. State, loaded packages, and compiled
  methods carry across turns, and a first solve is fast because each
  simulator ships a precompiled warm package.
- The agent reads real source. Every package in the environment is on disk at
  its real `pkgdir` path (read-only in the shared depot), so answers come from
  the installed version, not from memory.
- Everything is recorded. Each session writes a trace of every message, tool
  call, and artifact. Transcripts and the benchmark grade against it.
- Models are interchangeable: OpenAI, Anthropic, Google, or local models via
  Ollama, switchable mid-session.

## Supported simulators

| `--sim` | Package | Domain |
|---|---|---|
| `jutuldarcy` | [JutulDarcy.jl](https://github.com/sintefmath/JutulDarcy.jl) | Reservoir / multi-phase flow |
| `battmo` | [BattMo.jl](https://github.com/BattMoTeam/BattMo.jl) | Lithium-ion (and other) battery cells |
| `fimbul` | [Fimbul.jl](https://github.com/sintefmath/Fimbul.jl) | Geothermal (ATES, BTES, doublet, EGS) |
| `mocca` | [Mocca.jl](https://github.com/sintefmath/Mocca.jl) | Adsorption-based CO₂ capture (PSA / VSA / TSA) |

## Install

You need two tools on PATH:

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which
  installs Python and manages the tool
- Julia 1.10 or newer, via [juliaup](https://github.com/JuliaLang/juliaup)

On headless Linux, plotting also needs `xvfb`.

Install jutul-agent as a uv tool. This puts a `jutul-agent` command on your
PATH that works from any folder:

```sh
uv tool install git+https://github.com/SINTEF-agentlab/jutul-agent
```

Upgrade any time with `jutul-agent upgrade`.

API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`) can go in
your environment or a `.env`. jutul-agent also prompts for a missing key and
saves it. Local models through [Ollama](https://ollama.com) need no key.
Details in the
[installation and usage guide](https://sintef-agentlab.github.io/jutul-agent/usage/).

## Adding your own simulators and skills

Add a predefined skill by entering the path to a skills directory, with an optional simulator argument
if you want to specify which simulators should have access to the skill.

```sh
jutul-agent skill add /path/to/skill [--sim mysim]
jutul-agent skill add /path/to/skill
jutul-agent skill list
```

Add a predefined simulator to jutul-agent by entering the path to a simulator directory.

```sh
jutul-agent simulator add /path/to/simulator
jutul-agent simulator list
```

## First run

jutul-agent works in the directory you launch it from: that directory
becomes the workspace where it reads and writes files. Start it from a
fresh project folder.

```sh
mkdir my-battery-run && cd my-battery-run
jutul-agent init --sim battmo --precompile
jutul-agent
```

```
> Set up a constant-current discharge for the chen_2020 cell and plot the voltage curve.
```

`--sim` takes any simulator from the table above, and the workflow is the
same for all of them. The first `--precompile` takes a while (Julia compiles
the simulator and the plotting stack), after which sessions start in
seconds. If anything fails, `jutul-agent doctor` diagnoses the setup
and prints a fix per finding.

## Documentation

The full documentation lives at
[sintef-agentlab.github.io/jutul-agent](https://sintef-agentlab.github.io/jutul-agent/)
(source in [docs/](docs/)): using the agent, how it works (architecture,
the Julia kernel, the trace), and extending it (adding a simulator,
improving the agent, evaluation).

## Development

Work from a clone instead of a tool install. `uv run` resolves the
`jutul-agent` command from the checkout, so run every command through it
(`uv run jutul-agent ...`); upgrade with `git pull && uv sync`.

```sh
git clone https://github.com/SINTEF-agentlab/jutul-agent
cd jutul-agent
uv sync --extra eval
uv run pre-commit install

uv run ruff check .              # lint
uv run pytest                    # unit tests (integration and live skipped)
uv run pytest -m integration     # adds the Julia-requiring tests
uv run jutul-agent eval canary   # bench canary
```

Developed at [SINTEF](https://www.sintef.no/en/).
