# The agent's filesystem

The agent touches files through the file tools (`read_file`, `write_file`,
`edit_file`, `ls`, `glob`, `grep`) and `execute`, and they all operate on the
real filesystem. The workspace backend
(`agent/backend.py:WorkspaceShellBackend`, assembled in
`agent/builder.py:build_backend`) runs in real-path mode: a relative path
resolves against the workspace (the launch directory) and an absolute path is
used as-is. `run_julia` and `execute` resolve paths the same way, since their
working directory is the workspace too, so one string names one file in every
tool and the user can click any path the agent reports to open it.

| What | Where | Access |
|---|---|---|
| workspace files | relative to the launch dir, or their absolute path | read/write + shell |
| installed package source | the real depot path (`pkgdir(<Pkg>)`) | read-only |
| a `Pkg.develop` checkout | its real checkout path | read/write |
| agent memory | real files under the workspace state dir | read/write |
| skills | their real package-data directories | read |
| added folders (`--add-dir`) | their real absolute path | read/write |

Everything is a real path; the only special rule is that the file tools refuse
to **write** into the shared Julia depot (below). A bare leading slash
(`/model.jl`) is the machine root, not the workspace, in every tool. One path
model across the file tools, the shell, and the REPL means a path the agent
reads or writes is the same path it can hand to `run_julia`. The `filesystem`
eval suite checks this.

`include("file.jl")` inside `run_julia` resolves from the workspace because the
kernel rewrites top-level relative includes against its working directory (see
[the Julia kernel](julia-kernel.md)).

## Package source is read-only

Installed packages live in the shared Julia depot
(`~/.julia/packages/<Pkg>/<hash>/`), which other projects share, so editing them
would corrupt those projects. The workspace backend is given the depot's
read-only roots (the registry `package_sources`, keyed by location) and refuses
`write_file`/`edit_file` there, while reads and greps are unrestricted, which is
how the agent studies installed source. A `Pkg.develop` checkout lives outside
the depot, so it stays writable with no special-casing: location alone decides.

The agent finds a package's source with `pkgdir(<Pkg>)` in the REPL and reads or
greps that real path (its `examples/`, `docs/`, `src/`); a package added
mid-session with `Pkg.add` is at its `pkgdir` path immediately and is covered by
the write-guard automatically.

## Memory and skills

Agent memory is real markdown under the workspace state dir
(`$XDG_DATA_HOME/makker/workspaces/<hash>/memory/`); the agent reads and
edits it at that real path, the `remember` tool writes there directly, and the
user can open the files. Ephemeral memory (used by the eval) is the same, in a
real temp directory that is discarded afterward.

Skills are read from their real directories. The bundled ones ship as package
data, so they resolve at a real site-packages path even from a `pip install`
with no repo checkout. `skill_sources` returns those real dirs; the seam for
user- and project-level skills is to append more real dirs there (last wins).
