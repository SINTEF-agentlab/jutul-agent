# jutul-agent

## Purpose

jutul-agent is an assistant for Julia scientific simulation work on the Jutul
ecosystem, built on Deep Agents.

## Working rules

- Prefer the live Julia session and simulator source code over model memory.
- Use the persistent Julia REPL incrementally. Run the smallest useful probe
  first, inspect the result, then extend.
- Treat the installed simulator packages as the source of truth for APIs and
  examples.
- Keep Julia execution serialized unless the runtime explicitly supports
  concurrency.
- Prefer Deep Agents memory and skills for reusable guidance; keep custom
  Python focused on Julia execution, simulator inspection, and trace capture.

## Validation

- Default suite (no Julia): `uv run pytest -q -m "not integration"`
- Integration suite (requires Julia + an instantiated env): `uv run pytest -q -m integration`.
  Envs instantiate from `Project.toml` + `[sources]`.
  `tests/test_simulators_smoke.py` loads each simulator with `using <Sim>`.
- Live LLM (humans only, needs API key): `uv run pytest tests/live/`
- Update transcript / report snapshots after renderer changes:
  `uv run pytest --snapshot-update tests/test_transcript_html.py tests/test_transcript.py tests/test_report_renderer.py`
- Python style (matches CI): `uv run ruff check .` then `uv run ruff format .`
- After editing Python, run `uv run ruff format` on touched paths (or rely on
  format-on-save / pre-commit if installed).
- Web UI (React + TypeScript in `interfaces/server/webapp`): `npm install`, then
  `npm test` (vitest) and `npm run typecheck`. It ships **pre-built** — after any
  change under `webapp/src`, run `npm run build` and commit the regenerated
  `interfaces/server/web_dist`. That committed bundle is what the server serves,
  so end users install with `uv`/`pip` and never need Node.

## Continuous integration

GitHub Actions workflows:

- `ci.yml`: every push/PR. `lint` (ruff); `test` across Linux/Windows/macOS
  (`pytest -m "not integration"`, no Julia); `julia-integration` (Linux only:
  runs `test_juliakernel.py` against base Julia, which needs no env).
- `simulators.yml`: per-simulator smoke matrix (`jutuldarcy`, `battmo`,
  `fimbul`, `mocca`; Linux). On PR + push to `main`, a weekly Monday run
  (catches upstream breakage), and manual dispatch. Each job instantiates the
  env from `Project.toml` and runs `test_simulators_smoke.py`.
- `deps-canary.yml`: weekly and manual only. Resolves the newest stack,
  ignoring the pin, and reports to the run summary without ever marking a
  commit red. It is the early warning for when the pin can move.
- `docs.yml`: builds the MkDocs site on changes under `docs/`, publishing on
  `main`.
- `release.yml`: on a published GitHub release tagged `vX.Y.Z`, builds and
  uploads to PyPI. The tag is the only source of the version.

## Local git hooks

One-time per clone: `uv run pre-commit install`. Commits then run Ruff format
and check on staged `.py` files. To verify the whole tree like CI:
`uv run pre-commit run --all-files`.

## Important paths

- `src/jutul_agent/paths.py`: the three runtime anchors (`PACKAGE_ROOT`
  (install), `workspace_root()` (CWD), `state_home()` (sessions)) and the path
  policy (`resolve_in_workspace`, `is_host_path`), the one place that maps
  agent-visible paths to real files.
- `src/jutul_agent/workspace.py`: workspace config
  (`.jutul-agent/config.toml`), simulator auto-detect, Julia-env bootstrap.
- `src/jutul_agent/session.py`: `Session`, the unit of work for one
  invocation.
- `src/jutul_agent/models.py`: provider metadata and the model catalog
  discovery. `src/jutul_agent/display.py`: display detection and Xvfb
  management for headless plotting.
- `src/jutul_agent/agent/builder.py`: deepagents wiring (composite backend,
  HarnessProfile registration, `build_agent` entry point). Custom tools sit
  alongside (`tools.py`, `plot_julia.py`, `memory.py`, `approval.py`,
  `turns.py`, and `added_dirs.py`, which records `/add-dir` folders);
  `tool_labels.py` holds the one friendly name per tool shared by the TUI,
  approval prompt, and transcript. The Julia-side plot capture lives in the
  `julia_runtime/JutulAgent` package. The full system prompt is assembled in
  `prompts.py` (each rule stated once).
- `src/jutul_agent/simulators/`: adapter dataclass, registry, env bootstrap +
  launch-time preparation (`env_setup.prepare_workspace_env`), shared skills,
  and one folder per simulator (see below).
- `src/jutul_agent/julia/`: the `JuliaSession` Protocol (`session.py`) and the
  Julia toolchain checks (`requirements.py`).
- `src/jutul_agent/juliakernel/`: the backend, a self-contained, supervised
  Julia runtime (`kernel.py` + stdlib-only `server.jl`) with live output and
  SIGINT interrupt. Splittable into a standalone package.
- `src/jutul_agent/trace/`: SQLite event log and `TraceRecorder` middleware.
- `src/jutul_agent/transcript/`: renderers that consume a trace
  (HTML transcript, markdown transcript, investigation report).
- `src/jutul_agent/approval_preview.py`: the approval card body every surface
  renders for a pending tool call.
- `src/jutul_agent/session_host.py`: the one session bootstrap and turn-policy
  loop behind every front end.
- `src/jutul_agent/interfaces/`: `cli`, the Textual `tui`, and the `server`
  (FastAPI REST + per-session WebSocket). The browser UI is a React + TypeScript
  app in `server/webapp/` (see its README), built into `server/web_dist/`
  (committed and shipped, so an install needs no Node).
- `tests/`: end-to-end, chat, CLI, and tool coverage. `tests/integration/`
  needs Julia; `tests/live/` needs an LLM provider key.

## Per-simulator layout

Each simulator lives under `src/jutul_agent/simulators/<name>/` and owns
exactly three things:

- `adapter.py`: constructs the `SimulatorAdapter`. Set
  `module_dir = Path(__file__).resolve().parent` so the base class can
  derive `julia_env_template_path` and `skills_dir`.
- `julia_env/`: `Project.toml` declaring the deps the agent can `using`, plus a
  per-simulator `JutulAgent<Sim>/` warm package (a local `[sources]` path dep whose
  `@recompile_invalidations` + `@compile_workload` bakes that simulator's
  GLMakie-aware solve/plot into the precompile cache, so the first solve is fast).
  The shared, sim-agnostic `JutulAgent` package (figure capture, ensemble helpers,
  generic-Makie warm-up) has a single source in `src/jutul_agent/julia_runtime/` and
  is copied into the env at bootstrap (also a relative `[sources]` dep). The
  `Manifest.toml` is generated on instantiate (gitignored). The whole `julia_env/`
  folder is copied into a workspace on bootstrap, so the relative `[sources]` paths
  keep resolving.
- `skills/<skill-name>/SKILL.md`: markdown skills surfaced via the
  deep-agents skill system.

Add a simulator by creating that folder and one entry in
`simulators/registry.py`. Custom subagent factories, when they land, go on
the adapter's `subagent_factories` tuple.

## deepagents / langgraph private-surface contacts

We pin deepagents (see `[tool.uv] exclude-newer` in `pyproject.toml`) and
touch a few non-public surfaces. `tests/test_deepagents_canaries.py` asserts
each one still exists, so a bump fails at test time rather than degrading
silently at runtime. The contacts:

- `agent/tools.py`: reads `langgraph.pregel._tools._tool_call_writer`
  (ContextVar) to stream live Julia output as tool-output deltas.
  Import-guarded: if it moves, streaming silently disables.
- `agent/turns.py` + `jutul_agent/lab/fakes.py`: consume the v3
  `astream_events` typed projections (`run.messages` / `run.tool_calls` /
  `run.interrupts` / `run.output`); the fakes mirror that shape, so a
  projection change shows up as test failures.
- `agent/summarization.py`: manual `/compact` composes the summarization
  middleware's private engine methods. Guarded by a `hasattr` sweep that
  degrades to a clear "unavailable" message.
- `agent/backend.py`: subclasses `CompositeBackend` (overriding `grep`/`glob`
  so patterns recurse) and `LocalShellBackend` (overriding `write`/`edit`/
  `delete`/`execute` to refuse depot writes, reading `cwd` to resolve relative
  paths). A subclass breaks loudly if a base method's signature moves, but a
  narrowed override degrades quietly, which is why the search overrides
  forward `**kwargs`. After the real-paths refactor nothing routes through a
  mount, so `added_dirs.py` only records the added folders; every tool already
  reaches them by absolute path.
- `tests/test_builder.py`: imports
  `deepagents.profiles.harness.harness_profiles._harness_profile_for_model`
  to assert our profile resolves for built model instances. It also pins the
  exact tool set the framework installs, since a bump can add or drop one.
