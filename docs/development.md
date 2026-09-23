# Development

This page is the practical reference: setup, layout, tests, CI, policies.
The reasoning behind much of it, a codebase built with coding agents from
the start, is its own page: [agent-first development](agent-first.md).

## Setup

```sh
git clone <this-repo> && cd jutul-agent
uv sync --extra eval
uv run pre-commit install
```

`uv sync` creates `.venv/` from `pyproject.toml` + `uv.lock`. Re-run it
when dependencies change. The `eval` extra adds Inspect AI for the bench.
`uv run` also refreshes the checkout's installed version when its Git commit or
tags change; fetching a new release tag does not require a manual reinstall.

## Repository layout

```
src/jutul_agent/
  paths.py         install / workspace / state-home anchors, path-resolution policy
  workspace.py     config loader, simulator auto-detect, env bootstrap helpers
  session.py       Session: the unit of one invocation
  models.py        provider metadata, model catalog
  credentials.py   user-global API-key storage
  display.py       display detection, managed Xvfb for headless plotting
  agent/           deepagents wiring: builder, prompts, tools, turns, memory
  simulators/      adapter base + registry, one data folder per simulator
  julia/           JuliaSession protocol, Julia toolchain checks
  juliakernel/     the supervised Julia runtime (kernel.py + server.jl)
  julia_runtime/   the shared JutulAgent Julia package, synced into envs
  trace/           append-only SQLite event log + recorder middleware
  transcript/      HTML / markdown / report renderers
  eval/            jutul-bench: solver, scorers, runconfig, task suites
  interfaces/      cli/ and tui/
tests/             unit suite (integration/ and live/ are opt-in)
docs/              this documentation
```

## Tests

```sh
uv run pytest                    # unit tests (integration and live deselected)
uv run pytest -m integration     # adds Julia-requiring tests
uv run pytest tests/live/        # one real-LLM smoke (needs a provider key)
uv run pytest --snapshot-update  # accept changed syrupy snapshots, deliberately
```

How the tiers, gating, fakes, snapshots, and TUI pilot tests fit together
is its own page: [testing](testing.md).

## CI

Two test workflows:

- `ci.yml`, on every PR and push to main: lint (ruff check + format) and
  the full non-integration suite on Linux/macOS/Windows. Two pytest workers
  run test files in parallel; the workflow installs no Julia.
- `simulators.yml`: runs the live JuliaKernel and all four simulator jobs
  on every PR and push. The JutulDarcy job also runs GLMakie plot integration
  tests, sharing its setup and precompile. Weekly and manually dispatched
  runs check the same environments against the latest compatible upstream
  releases. Every run resolves from `Project.toml` and `[sources]`, just like
  a new user workspace; no CI lockfiles are maintained or cached.

Each simulator job runs an explicit `Pkg.precompile()`, which throws if
a direct dependency fails to precompile. A bare `Pkg.instantiate()` only
auto-precompiles best-effort and exits 0, which can leave a lane green
while every env on the runner is broken.

Simulator jobs restore the latest Julia depot for their simulator, OS,
architecture, and exact Julia version, then instantiate and precompile normally.
`JULIA_CPU_TARGET=generic` makes compiled package images portable across runner
CPUs. After successful precompilation, the depot is saved under a key that hashes the
resolved Manifest and bundled Julia source. If that key was restored, nothing is
uploaded. Ordinary Python/docs changes therefore reuse the same archive; a new
upstream dependency or changed Julia source saves a new one. Main's caches are
available to PRs through GitHub's normal cache scoping.

The cache contains installed packages and compiled code, not the env's Manifest.
An upstream release can still trigger expensive precompilation, and an evicted
cache needs rebuilding. This preserves testing against current upstream without
lockfile updates, cache-specific dependency policies, or a cleanup workflow.

## The bench in the dev loop

Before and after a change to the prompt, a skill, or a tool, run the cheap
suites on the default model:

```sh
uv run jutul-agent eval canary guardrails
```

See [evaluation](evaluation.md) for scorers, RunConfig attribution, and
adding tasks.

## Dependency policy

Dependencies are locked (`uv.lock`) and resolution is pinned to a snapshot
date (`tool.uv.exclude-newer` in `pyproject.toml`). To move the whole stack
forward: bump that date, run `uv lock --upgrade`, run the suite, commit. Plain
`uv lock` keeps the versions already resolved, so it will not move anything.
To move one package, `uv lock --upgrade-package <name>`.

deepagents needs more care than the rest, because it owns the agent's tool
surface and the format of what those tools print. Stay on the stable line (not
the `aN` pre-releases), and check two things by hand on a bump, since neither
shows up as a failing import:

- The tool surface. The framework decides which file tools exist, so a bump can
  add one or stop installing one. `test_agent_tool_surface_is_pinned` pins the
  exact set and requires every mutating tool to be approval-gated.
- Tool output formats. Anything that parses what a tool prints is a contract the
  framework can change silently, so those tests build their input with the
  framework's own formatter rather than a hand-written sample.

Then run the live smoke (`/compact`, a streamed tool, a HITL approve) and a TUI
pilot pass.

## Releasing

The package version is derived from git tags by hatch-vcs, so a release is a
tag rather than a manual version bump. A tag `vX.Y.Z` builds as version
`X.Y.Z`; commits past it build as the next patch's `.devN` pre-release
(for example, two commits after `v0.2.0` build as `0.2.1.dev2+g<hash>`). The
runtime reads the version back through `importlib.metadata`
(`jutul_agent.__version__`), and the update checker compares it against the
latest release published on PyPI.

Publishing is automated by `.github/workflows/release.yml`, which runs when a
GitHub Release is published: it builds the sdist and wheel, verifies the built
version equals the release tag, and uploads to PyPI with trusted publishing
(OIDC), so no API token is stored in the repository. Trusted publishing is
configured once on PyPI (a publisher for owner `SINTEF-agentlab`, repository
`jutul-agent`, workflow `release.yml`, environment `pypi`) against a GitHub
Environment of the same name.

Cutting a release:

1. Make sure `main` is at the commit to ship and CI is green.
2. Create a GitHub Release with the tag `vX.Y.Z` targeting `main` (the UI
   creates the tag), and write the release notes.
3. Publishing the release triggers the workflow, which builds and uploads to
   PyPI.
4. Verify with `uv tool install --reinstall jutul-agent`, then check that
   `jutul-agent --version` reports `X.Y.Z` and the new release shows on the
   PyPI project page.

## The docs site

The documentation in `docs/` doubles as an MkDocs Material site
(`mkdocs.yml` at the repo root):

```sh
uv sync --group docs
uv run mkdocs serve     # live-preview at http://127.0.0.1:8000
uv run mkdocs build --strict
```

Install the `docs` group (it pulls in `mkdocs-material`), not bare `mkdocs`:
a plain `uv pip install mkdocs` lacks the Material theme and fails with
`cannot find module 'material.extensions.emoji'`. `uv run --group docs
mkdocs serve` also works without a prior `uv sync` if you prefer.

The `Docs` workflow checks the strict build on docs-touching pull requests
and deploys the site to GitHub Pages when docs change on main.

The architecture diagram is TikZ source (`docs/assets/architecture.tex`)
compiled offline to `architecture-light.svg` and `architecture-dark.svg`.
To change it, edit the `.tex`, regenerate, preview on the built site in
both palettes (`uv run mkdocs serve`), then commit the `.tex` and both SVGs.
Requires `latex` and `dvisvgm` (Debian: `texlive-latex-base`, `dvisvgm`):

```sh
docs/assets/render-architecture.sh
```
