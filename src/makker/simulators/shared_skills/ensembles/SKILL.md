---
name: ensembles
description: Run parallel ensembles (UQ, parameter sweeps) across warm Distributed workers from the Julia REPL
---

# Ensembles and parameter sweeps

## When to use

Use this when a task runs the **same simulation many times** with different inputs —
uncertainty quantification, parameter sweeps, sensitivity studies — and the runs are
independent. For a single run, just use `run_julia`.

When the task asks for a parallel ensemble, run it in parallel — don't quietly
fall back to a serial loop. If the parallel path fails, fix it or say so.

## How

The session loads the `JutulAgent` package, which exports `run_ensemble` (if it's
not yet defined, run `using JutulAgent` first). Call it from `run_julia`
(`MySim` below stands for whatever packages the case function needs):

```julia
run_ensemble(run_case, cases; nworkers=4, setup=:(using MySim))
```

- `run_case` — a function applied to each case; it runs **on a worker process**.
- `cases` — the inputs to sweep (a vector). Results come back in this order.
- `nworkers` — how many worker processes to spawn (capped at `length(cases)`).
- `setup` — an expression run on every worker first, to load whatever `run_case`
  needs. Required whenever `run_case` uses a package.

It spawns **warm** workers (they inherit this session's project and system image, so
no second precompile), runs the cases in parallel, returns the results in order, and
removes the workers it spawned afterwards.

While it runs, `run_ensemble` shows a live progress bar (`run_ensemble  60%|…|
ETA …`) that advances as each case finishes. Worker output only reaches this
session interleaved and prefixed (`From worker N:`), and a solve under
`info_level = -1` is silent — so that bar is the clean signal a parallel run is
progressing. Don't mistake the quiet for a hang, and tell the user the same if they ask.

## Rules (Distributed gotchas)

- **Anything defined only in this session does not exist on the workers.** Workers
  are spawned fresh inside `run_ensemble`; a *named* function from this session is
  serialised by name only, so the run fails with `UndefVarError(#run_case)`. The
  same applies to session globals (`GRID`, `SYS`, ...) referenced inside the case
  function — and to a `do` block that merely *wraps* a session-defined function:
  the wrapper serialises, the call inside it still fails.
- Two patterns that work (pick by size):
  1. **Inline `do` block, fully self-contained**: every binding it uses is either
     built inside it, passed through `cases`, or comes from a package loaded via
     `setup`.
  2. **Define the case function in a workspace file and `include` it via `setup`**,
     so every worker has the definition; then the named function can be passed
     directly.
- `setup` must be a quoted expression like `:(using MySim)` (`nothing`,
  not `:(nothing)`, to skip it).
- Keep what each case returns small (a scalar, a NamedTuple of metrics) — large return
  objects are serialised back to this process.

## Example (inline do-block)

```julia
# Sweep a parameter; each worker builds its own case and returns a scalar metric.
results = run_ensemble(
    [0.10, 0.15, 0.20, 0.25];
    nworkers = 4, setup = :(using MySim),
) do p
    # build the case from `p`, run it, extract the metric — everything in here
    return metric   # small return value
end
```

## Example (case function in a file)

```julia
# sweep_case.jl in the workspace defines `run_case(case)` at top level.
results = run_ensemble(
    run_case, cases;
    nworkers = 4,
    setup = :(begin using MySim; include("sweep_case.jl") end),
)
```

The `include` runs on every worker (their working directory is the workspace), so
`run_case` resolves there. This is the better pattern once the case function is
more than a few lines: it stays editable as a file and runs serially too
(`map(run_case, cases)`) for debugging one case first.

For very heavy sweeps, set `nworkers` to the cores you want to use; for a one-off
parallel block you can also drop to plain `Distributed` (`addprocs`, `@everywhere`,
`pmap`) inside `run_julia` — `run_ensemble` is just the ergonomic path.
