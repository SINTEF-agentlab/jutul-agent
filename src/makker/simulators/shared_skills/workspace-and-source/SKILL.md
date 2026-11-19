---
name: workspace-and-source
description: Workspace file layout, when to write a file vs. evaluate in the REPL, and how to search installed simulator source
---

# Workspace and source files

## Where you are working

You are running in the user's *workspace* (their current working directory).
Treat it like a project: read, write, and execute files here freely. The
workspace owns its Julia environment under `.makker/julia-env/` (or the
user's own `Project.toml` at the workspace root).

Sessions, traces, and artifacts live *outside* the workspace under the
user's state home (`$XDG_DATA_HOME/makker/workspaces/<hash>/`). Don't
write transcripts or logs into the workspace.

## Paths

The file tools (`read_file`, `write_file`, `edit_file`, `glob`, `grep`, `ls`),
`execute`, and the Julia REPL share one working directory, the workspace, and
use the same paths. Name a workspace file by a relative path (`model.jl`,
`experiments/data.csv`) or its absolute path; both work everywhere:

```julia
CSV.read("experiments/observations/data.csv", DataFrame)
include("candidate.jl")
```

So the file you `write_file` as `candidate.jl` is the file you `include`, and an
absolute path from a Julia stack trace opens directly with `read_file`. List
workspace files with `glob("**/*")` or `ls(".")` / `ls("experiments")`.

Every path is a real filesystem path: your files, installed package source,
memory notes, and added folders all open the same in the file tools, `execute`,
and `run_julia`. A bare leading slash (`/model.jl`) is the machine root, not the
workspace.

## When to write a file vs. evaluate in the REPL

- **Real implementation → real file.** If the user is asking you to build
  something (a simulation script, a setup function, a case definition),
  use `write_file` / `edit_file` to create a `.jl` (or `.py`, `.md`, …)
  file in the workspace. The user can then open it in their editor,
  inspect it, edit it, and run it. Iterate via `edit_file` diffs, not
  by regenerating the whole thing in chat.

- **Quick probe → REPL.** Tiny one-offs (`@doc`, `methods`, "what fields
  does this struct have?", "what does this return for a small input?")
  belong in `run_julia`. Don't litter the workspace with one-liners.

- **Pattern for running a file you just wrote:** write the file, then
  load it into the REPL with `run_julia('include("solve.jl")')`. The
  REPL keeps state across calls, so you can iterate on the file and
  re-`include` it without paying the package-load cost again.

## Reading installed package source and examples

The active simulator's package source path is given to you up front (in the
system prompt). Read and grep it directly with the file tools — don't run
`using <Sim>` / `pkgdir(<Sim>)` to find it, since that loads the package (slow
the first time) just to learn a path you already have:

```text
glob("/.../JutulDarcy/examples/**/*.jl")          # discover examples
read_file("/.../JutulDarcy/examples/.../example.jl")
grep("setup_well", path="/.../JutulDarcy/src")    # find API uses
```

Every *other* package the environment resolves (the Jutul-stack packages the
simulator builds on, their dependencies, and anything you `Pkg.add`) also has
its source on disk; `pkgdir(<Package>)` in the REPL returns that path:

```julia
# run_julia
pkgdir(SomeDependency)         # -> /.../.julia/packages/SomeDependency/<hash>
```

Installed source is **read-only**: it lives in the shared Julia depot, so the
file tools refuse to `write_file`/`edit_file` there (editing it would break
other projects). Read and grep it freely. To change a package itself,
`Pkg.develop` it (see below); the checkout lives outside the depot and is
writable.

For exact signatures and docstrings, stay in the REPL — these read the
installed version directly and are always current:

```julia
# run_julia
@doc some_function             # docstring
methods(solve)                 # available methods
names(SimulatorPackage)        # exported names of the active simulator's package
```

Rule of thumb: **read/grep the simulator's source at the path you were given (and
`pkgdir(<Package>)` for any other package); `@doc` / `methods` / `names` in
`run_julia` for precise API.**

If `.makker/config.toml` sets a `source_path` for the simulator, its
primary package is `Pkg.develop`-ed there, so the source path you were given is
that checkout. It is **writable**, so you can `edit_file` it to modify the
library and re-`include` to test.

## Workspace already has its own Project.toml?

If the user's workspace has a `Project.toml` at the root, that's the
Julia env you use — not `.makker/julia-env/`. The user owns it.
`Pkg.add`/`Pkg.develop` modifies *their* env; don't do that without
asking.
