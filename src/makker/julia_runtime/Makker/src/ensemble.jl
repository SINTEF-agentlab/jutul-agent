# Ensemble helpers for makker: warm Distributed workers + pmap, for the
# parallel runs an uncertainty quantification or parameter sweep needs. Loaded into
# the session on demand (like julia_plot.jl), so it adds no environment dependency.
#
# Design: the agent's eval runs in the *master* process, so `addprocs` + `pmap`
# here are the standard Julia pattern — the master has every package loaded, so the
# "packages must be on the master" class of failure (the old worker-based backend's
# pain) cannot occur. Workers inherit the master's project and system image, so they
# boot warm.

module MakkerEnsemble

using Distributed
using ProgressMeter: Progress, next!

export run_ensemble, warm_addprocs

"""
    warm_addprocs(n; threads=nothing) -> Vector{Int}

Add `n` worker processes that inherit this (master) process's project and system
image, so they start warm — no second precompile. Returns the new worker ids.
"""
function warm_addprocs(n::Integer; threads::Union{Nothing,Integer}=nothing)
    n <= 0 && return Int[]
    flags = String[]
    proj = Base.active_project()
    proj === nothing || push!(flags, "--project=$(dirname(proj))")
    # Forward a *custom* system image so workers skip a cold precompile; the default
    # system image (under the Julia install) is found automatically, so leave it off.
    img = unsafe_string(Base.JLOptions().image_file)
    if !isempty(img) && isfile(img) && !startswith(img, Sys.BINDIR)
        push!(flags, "--sysimage=$img")
    end
    threads === nothing || push!(flags, "--threads=$(threads)")
    return addprocs(Int(n); exeflags=`$flags`)
end

"""
    run_ensemble(f, cases; nworkers=length(cases), setup=nothing,
                 threads=nothing, keep_workers=false) -> Vector

Run `f(case)` for each `case` in `cases` in parallel across warm workers and return
the results in order. Spawns workers as needed (up to `nworkers`, capped at the
number of cases), optionally evaluates `setup` (an expression, e.g.
`:(using JutulDarcy)`) on every worker first, `pmap`s the work, then removes the
workers it spawned unless `keep_workers`.

`f` runs on the workers, so anything it needs must be available there: load packages
via `setup`, and prefer a named top-level function over a closure capturing local
state.
"""
function run_ensemble(
    f,
    cases;
    nworkers::Integer=length(cases),
    setup::Union{Nothing,Expr}=nothing,
    threads::Union{Nothing,Integer}=nothing,
    keep_workers::Bool=false,
)
    cases = collect(cases)
    isempty(cases) && return Any[]
    existing = filter(!=(1), Distributed.workers())  # workers() is [1] when none
    target = min(Int(nworkers), length(cases))
    spawned = length(existing) < target ? warm_addprocs(target - length(existing); threads) : Int[]
    try
        setup === nothing || Distributed.remotecall_eval(Main, Distributed.workers(), setup)
        return _pmap_with_progress(f, cases)
    finally
        (keep_workers || isempty(spawned)) || rmprocs(spawned)
    end
end

"""
    _pmap_with_progress(f, cases) -> Vector

`pmap` with a live `ProgressMeter` bar driven from the master. A worker's stdout
reaches the master only interleaved and prefixed (`From worker N:`), and a solve
under `info_level = -1` is silent, so a plain parallel run gives no clean progress
signal, looks hung, and may be cancelled while fine. A single ordered bar fixes that.

`ProgressMeter.progress_pmap` won't do: it serialises a `ProgressMeter` closure to
the workers, which our fresh warm workers don't have loaded. Instead the bar lives
on the master and each case is dispatched with `remotecall_fetch(f, pool, case)`, so
only `f` and the case cross to a worker (exactly `pmap`'s serialisation, no
worker-side dependency) while `next!` ticks here as each returns. `asyncmap` keeps
one in-flight call per worker and preserves input order, like `pmap`.
"""
function _pmap_with_progress(f, cases)
    n = length(cases)
    workers = Distributed.workers()  # [1] (master) when none were spawned
    nw = max(length(workers), 1)
    pool = Distributed.WorkerPool(workers)
    println(stderr, "run_ensemble: $n case$(n == 1 ? "" : "s") across $nw worker$(nw == 1 ? "" : "s")…")
    flush(stderr)
    prog = Progress(n; desc = "run_ensemble ", output = stderr)
    return asyncmap(cases; ntasks = nw) do case
        r = Distributed.remotecall_fetch(f, pool, case)
        next!(prog)
        return r
    end
end

end # module
