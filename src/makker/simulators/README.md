# simulators/

One folder per simulator that makker knows about. Each folder is
self-contained:

```
simulators/<name>/
  adapter.py          # the SimulatorAdapter instance, exported from __init__.py
  julia_env/          # Project.toml + JutulAgent<Sim>/ warm package
  skills/             # one sub-folder per skill, each with a SKILL.md
```

`shared_skills/` holds skill markdown loaded for every session regardless
of the active simulator.

## Adding a new simulator

1. Create `simulators/<name>/`.
2. Write `adapter.py` exporting a `SimulatorAdapter`. Set
   `module_dir = Path(__file__).resolve().parent` — the base class derives
   `julia_env_template_path` and `skills_dir` from that.
3. Add `julia_env/Project.toml` with the deps the agent should be able to
   `using`. The kernel needs no special dep (its server is stdlib-only).
   Declare the shared `JutulAgent` package as a relative `[sources]` path dep
   (copy the entry from an existing env) — its single source lives in
   `src/makker/julia_runtime/` and is copied into the env at bootstrap.
   Do not commit a `Manifest.toml`; it is generated on instantiate.
4. Add a `julia_env/JutulAgent<Sim>/` warm package (start from an existing
   simulator's) and set the adapter's `warm_package`. Its
   `@recompile_invalidations` + `@compile_workload` bake the simulator's
   GLMakie-aware solve/plot into the precompile cache, so the agent's first
   solve is seconds instead of half a minute.
5. Add at least one skill at `skills/<name>-overview/SKILL.md`.
6. Register the adapter in `simulators/registry.py` and add the simulator to
   the `simulators.yml` CI matrix.

## Bootstrapping a workspace

`makker` copies `julia_env/` into `<workspace>/.makker/julia-env/`
on first use. To force a refresh after updating the template:

```sh
uv run makker init --sim <name> --force
```

To dev-link the simulator package against a local checkout:

```sh
uv run makker init --sim <name> --source-path /path/to/<Package>.jl
```
