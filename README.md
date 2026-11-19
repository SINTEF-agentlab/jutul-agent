# Makker

[![CI](https://github.com/SINTEF-agentlab/makker/actions/workflows/ci.yml/badge.svg)](https://github.com/SINTEF-agentlab/makker/actions/workflows/ci.yml)
[![Simulators](https://github.com/SINTEF-agentlab/makker/actions/workflows/simulators.yml/badge.svg)](https://github.com/SINTEF-agentlab/makker/actions/workflows/simulators.yml)
[![Docs](https://img.shields.io/badge/docs-website-0f4c5c)](https://sintef-agentlab.github.io/makker/)
[![PyPI](https://img.shields.io/pypi/v/makker.svg)](https://pypi.org/project/makker/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An AI agent for computational science. Give Makker a question and it plans the
study, sets up and runs the simulations, analyses and plots the results, and
decides what to try next. It can carry an investigation on its own or work
through it with you, and it records every step so the result can be checked.

*Makker* (MAHK-er) is Norwegian for the partner you do a job with. The project
was formerly called jutul-agent.

Makker works across simulators and fields of science. The ones below come ready
to use, and you can
[add your own](https://sintef-agentlab.github.io/makker/extending-for-your-application/).

<p align="center">
  <a href="https://github.com/sintefmath/JutulDarcy.jl"><img src="https://raw.githubusercontent.com/sintefmath/JutulDarcy.jl/main/docs/src/assets/logo_wide.png" alt="JutulDarcy" height="42"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/BattMoTeam/BattMo.jl"><img src="https://raw.githubusercontent.com/BattMoTeam/BattMo.jl/main/docs/src/assets/battmologo_text.png" alt="BattMo" height="74"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/sintefmath/Fimbul.jl"><img src="https://raw.githubusercontent.com/sintefmath/Fimbul.jl/main/docs/src/assets/logo_text_wide.png" alt="Fimbul" height="50"></a>&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/sintefmath/Mocca.jl"><img src="https://raw.githubusercontent.com/sintefmath/Mocca.jl/main/docs/src/assets/mocca_small_transparent.png" alt="Mocca" height="62"></a>
</p>

<p align="center">
  <img src="docs/images/web-fimbul.png" alt="Makker web UI: a Fimbul geothermal doublet with the 3D temperature field over time" width="47%">
  &nbsp;&nbsp;&nbsp;
  <img src="docs/images/web-battmo.png" alt="Makker web UI: a BattMo discharge-rate investigation with plots and a report" width="47%">
</p>
<p align="center"><sub>The web UI (<code>makker</code>): chat on the left; interactive plots and reports pinned in a canvas on the right. On the left, a Fimbul geothermal doublet showing the 3D temperature field around the wells over time. On the right, a BattMo C-rate study with voltage curves, an attempts map, and a written report.</sub></p>

## Quickstart

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Julia
(via [juliaup](https://github.com/JuliaLang/juliaup)). Then:

```sh
uv tool install makker
makker key openai            # save a key (Anthropic, Google, and OpenRouter work too)

mkdir my-study && cd my-study
makker init --sim battmo     # jutuldarcy | battmo | fimbul | mocca
makker                       # start the web UI (or: makker tui)
```

Local models through [Ollama](https://ollama.com) need no key. `init`
precompiles the Julia and plotting stacks once (a few minutes), so sessions
then start in seconds, and `makker doctor` checks the setup if anything
looks wrong. Keep it current with `makker upgrade`.

One folder is bound to one simulator and its Julia environment, so use a
separate folder for each. `makker` with no command serves the web UI at
<http://127.0.0.1:8742>, locally for a single trusted user; `makker tui`
is the terminal alternative. The full [installation and usage
guide](https://sintef-agentlab.github.io/makker/usage/) covers workspaces,
interfaces, and models; to work on Makker itself, clone the repo (see
[Development](#development)).

## Supported simulators

| `--sim` | Package | Domain |
|---|---|---|
| `jutuldarcy` | [JutulDarcy.jl](https://github.com/sintefmath/JutulDarcy.jl) | Reservoir / multi-phase flow |
| `battmo` | [BattMo.jl](https://github.com/BattMoTeam/BattMo.jl) | Lithium-ion (and other) battery cells |
| `fimbul` | [Fimbul.jl](https://github.com/sintefmath/Fimbul.jl) | Geothermal (ATES, BTES, doublet, EGS) |
| `mocca` | [Mocca.jl](https://github.com/sintefmath/Mocca.jl) | Adsorption-based CO₂ capture (PSA / VSA / TSA) |

## What makes it work for scientific computing

- A persistent Julia REPL per session. State, loaded packages, and compiled
  methods carry across turns, and a first solve is fast because each
  simulator ships a precompiled warm package.
- The agent reads real source. Every package in the environment is on disk at
  its real `pkgdir` path (read-only in the shared depot), so answers come from
  the installed version, not from memory.
- Everything is recorded. Each session writes a trace of every message, tool
  call, and artifact. Transcripts and the benchmark grade against it.
- Models are interchangeable: OpenAI, Anthropic, Google, OpenRouter, or local models via
  Ollama, switchable mid-session.
- Where the packages are fixed, such as a demo or a shared machine,
  `init --sysimage` bakes the whole Julia environment into a [system
  image](https://sintef-agentlab.github.io/makker/sysimage/) and sessions
  start with it already loaded. It is checked against the environment before
  every session, so an image that has fallen behind refuses to run instead of
  quietly using the old code.

## Documentation

The full documentation lives at
[sintef-agentlab.github.io/makker](https://sintef-agentlab.github.io/makker/)
(source in [docs/](docs/)): using the agent, how it works (architecture,
the Julia kernel, the trace), and extending it (adding a simulator,
improving the agent, evaluation).

## Development

Work from a clone instead of a tool install. `uv run` resolves the
`makker` command from the checkout, so run every command through it
(`uv run makker ...`); upgrade with `git pull && uv sync`.

```sh
git clone https://github.com/SINTEF-agentlab/makker
cd makker
uv sync --extra eval
uv run pre-commit install

uv run ruff check .              # lint
uv run pytest                    # unit tests (integration and live skipped)
uv run pytest -m integration     # adds the Julia-requiring tests
uv run makker eval canary   # bench canary
```

Developed at [SINTEF](https://www.sintef.no/en/).
