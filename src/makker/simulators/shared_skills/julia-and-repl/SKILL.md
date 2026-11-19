---
name: julia-and-repl
description: Persistent Julia REPL workflow and live runtime introspection for Jutul-based work
---

# Julia REPL workflow

## When to use

Use this skill whenever a task depends on Julia execution, package APIs, or REPL state.

You drive a single persistent Julia process. Loaded modules, defined values,
and compiled methods persist across turns — pay the load cost once and re-use
the same bindings.

The REPL's working directory is the **workspace**, the same place the file tools
work, with the same paths. Reference workspace files by bare relative path
(`include("script.jl")` for a file written as `script.jl`); a file's absolute
path works too. A bare leading slash (`/script.jl`) is the machine root, not the
workspace.

**Always run Julia code through `run_julia` (or `plot_julia` for figures).**
Never reach for `execute` to spawn `julia`, `julia --project`, `julia -e ...`,
or a shell pipeline that runs Julia: every such call starts a brand-new
process with no shared state, pays the full precompile cost, and the user
has to approve a shell command. Results that feed your conclusions
(simulation outputs, quantities the user asked for) come from this session's
REPL, not recomputed through some other interpreter; `execute` covers
ordinary shell work (grep, find, ls, git, …).

## Build incrementally

Construct the smallest piece first, evaluate it, look at the result, then
extend. Do not dump a full script into the REPL and hope.

## Reuse expensive results

Run a simulation (or any expensive call) as the last statement of its own
`run_julia` call and bind it to a name, so the result persists in the REPL.
Read or post-process it in a separate call. If that later step errors, fix the
read and reuse the saved result; do not re-run a simulation that already
succeeded.

## Live introspection beats memory

- `@doc f` - docstring.
- `methods(f)` - all method signatures.
- `methodswith(T)` - methods that dispatch on `T`.
- `fieldnames(typeof(x))` - fields of a value.
- `pkgdir(Module)` - the package's real source path on disk.

The installed package is the source of truth. If your training prior says the
API has a function but `methods` finds nothing, trust `methods`. To read
examples or source, get the path with `pkgdir(<Package>)` and browse it with the
file tools (`glob`/`grep`/`read_file`); it's read-only (the shared depot).

## Code in your reply

Wrap Julia in fenced blocks. Do not paste REPL prompts. Do not claim a script
works unless you have actually run it.

## When Julia fails

Read the full stack trace. Common fixes:

- **File not found**: check the path. A workspace file is a relative path
  (`experiments/...`) or its absolute path; `pwd()` / `isfile("...")` in the
  REPL confirm. A bare leading slash (`/experiments/...`) is the machine root,
  not the workspace, so drop it or use the absolute path.
- **Package not found**: confirm with `using Pkg; Pkg.status()`. If a Julia
  standard library already covers the need, prefer it over adding a dependency.
  Otherwise `Pkg.add` it when the task needs it; the added package's source is
  then at `pkgdir(<Package>)`.
- **Method/API error**: probe with `@doc`, `methods`, and a smaller snippet
  before retrying the full script.
- **Old version installed, or a just-added package won't precompile**: a
  long-lived env holds shared dependencies at their pinned versions, so a plain
  `Pkg.add` may only fit an old one. Inspect with `Pkg.status(outdated = true)`
  (`⌃`/`⌅` flag held-back packages) and `Pkg.why("Dep")`, then `Pkg.update()` to
  re-resolve the whole env and retry. Only a package still held back after a full
  update is a genuine compatibility limit.
- **A loaded module won't resolve (`PkgId(...) not found`)**: the session's
  loaded modules are out of sync with the env, usually after a mid-session
  package change. Julia can't reload a module from inside the REPL, so this needs
  `reset_julia` (see below).

`reset_julia` restarts Julia with an empty session. It is the right fix when a
module must be reloaded, but it clears **all** state (loaded packages, values,
and results) and recompiles on the next run. Use it deliberately, not as a
default; install the packages you need *before* building up expensive state so a
reset stays cheap.

Do not repeat the same failing expression unchanged.