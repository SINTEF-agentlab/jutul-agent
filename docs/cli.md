# CLI reference

With a tool install (`uv tool install`), every command is just `makker
…`. From a dev checkout, run them through `uv run makker …` instead. Two
flags are accepted by every command:

| Flag | Meaning |
|---|---|
| `--workspace <path>` | Workspace directory (default: current working directory) |
| `--state-home <path>` | Where sessions and traces live (default: `$XDG_DATA_HOME/makker`, falling back to `~/.local/share/makker`) |

## Interfaces

`makker` has three interfaces. With no command it starts the web UI:

| Command | Interface |
|---|---|
| `makker [options]` | Web UI (HTTP + WebSocket server + bundled web app); `makker web [options]` is the same |
| `makker tui [options]` | Terminal UI |
| `makker run "<prompt>" [options]` | One headless turn, then exit |

`makker --help` lists the commands, and `makker --version` prints the
version. A word that is not a command is an error, so a typo or a bare prompt
never starts a server.

### Makker (web UI)

```sh
makker [--sim <name>] [--approval-mode <mode>] [--model <m>] [--host <addr>] [--port <n>]
```

Serves the web UI (default <http://127.0.0.1:8742>). `makker web` is the
same command with the same options. One folder is bound to
one simulator — from `--sim`, the workspace config, or auto-detection — and the
choice is remembered. Runs locally for a single trusted user; the protocol and
embedding are covered in [the server interface](server-interface.md).

The session knobs below set the default for every session this server creates
(one folder, one Julia environment); the model and approval policy can still be
changed per session in the UI.

| Option | Meaning |
|---|---|
| `--sim <name>` | Simulator for this folder's sessions (persisted to the workspace config) |
| `--approval-mode ask\|workspace\|auto` | Default human-in-the-loop policy for new sessions (change per session in the UI with `/approval-mode`) |
| `--model <provider:model>` | Default model for new sessions (override per session with the UI model picker) |
| `--julia-project <path>` | Use a prepared Julia project at this path; skip workspace env setup |
| `--threads <N\|auto>` | Julia compute threads (default: physical cores minus one) |
| `--add-dir <path>` | Add an extra folder the agent can read and edit (repeatable; also runtime `/add-dir`) |
| `--ephemeral-memory` | Use a throwaway memory directory; nothing persists to workspace memory |
| `--sysimage` / `--no-sysimage` | Start sessions from the folder's [system image](sysimage.md), or skip it, overriding the folder's setting |
| `--host <addr>` | Address to bind (default `127.0.0.1`, localhost only) |
| `--port <n>` | Port to bind (default `8742`) |
| `--allowed-origin <origin>` | Let pages from another origin use the server (repeatable). Only its own pages may by default; see [which pages may use it](server-interface.md#which-pages-may-use-it) |

### makker tui

```sh
makker tui [options]
```

Launches the interactive terminal UI.

| Option | Meaning |
|---|---|
| `--sim <name>` | Active simulator. Required only if not in workspace config and not auto-detectable from a `Project.toml` |
| `--model <provider:model>` | Model for this session. Precedence: this flag, workspace config, user config, `$MAKKER_MODEL`, default |
| `--julia-project <path>` | Use a prepared Julia project at this path; skip workspace env setup |
| `--threads <N\|auto>` | Julia compute threads (default: physical cores minus one) |
| `--add-dir <path>` | Add an extra folder for the agent (repeatable) |
| `--continue` | Continue the most recent session in this workspace |
| `--resume [id]` | Resume a session by id or unique prefix; with no value, pick from a list |
| `--approval-mode ask\|workspace\|auto` | Human-in-the-loop policy |
| `--ephemeral-memory` | Throwaway memory directory for this session |
| `--sysimage` / `--no-sysimage` | Start from the folder's [system image](sysimage.md), or skip it, overriding the folder's setting |

### makker run

```sh
makker run "<prompt>" [options]
```

Runs one headless turn and exits — same options as `tui` (headless runs need
`--approval-mode auto`). Exit codes: 0 on success, 3 when the turn needed an
approval that headless mode cannot ask for.

## makker init (alias: setup)

Bootstrap the workspace: write `.makker/config.toml`, copy the simulator's
Julia env template, and (by default) precompile the env, plotting backends
included, so the first session in any interface is fast.

| Option | Meaning |
|---|---|
| `--sim <name>` | Simulator to bootstrap (omit to auto-detect) |
| `--source-path <path>` | `Pkg.develop` a local checkout of the simulator package (persisted to the workspace config); the checkout's own `[compat]` replaces the template's bounds for the simulator and what it depends on |
| `--no-precompile` | Skip the bake — bootstrap config + env only; the first session builds the rest |
| `--force` | Replace an existing workspace env with a fresh template copy, discarding packages added to it (to update it in place instead, use `makker upgrade`) |
| `--sysimage` | Also build a [system image](sysimage.md) for this folder, and start from it from now on. Adds tens of minutes to `init` |

## makker sysimage

Manage this folder's Julia [system image](sysimage.md): every package the
environment uses, compiled into one file the session starts from, which takes
most of the package loading out of every launch.

```sh
makker sysimage           # status: what it contains, and whether it still fits (default)
makker sysimage build     # build one, and start from it from now on
makker sysimage clear     # remove it, and stop using it
```

A successful build turns the folder's setting on and `clear` turns it off, so
the setting and the file on disk cannot drift apart. A failed or interrupted
build leaves the previous image in place: the new one is verified under a
temporary name and only then moved in.

| Option | Meaning |
|---|---|
| `--sim <name>` | Simulator to build for (default: workspace config / auto-detect) |
| `--cpu-target <target>` | CPU the image is compiled for. The default, `native`, is fastest here and will not run on other hardware; pass `generic` for a portable image |

## makker doctor

Diagnose the setup: Julia version, provider key (or Ollama reachability),
which Julia project the workspace resolves to, whether the simulator package
is actually in the manifest, whether the env was built from the current
simulator template, whether a system image still matches the environment,
display/xvfb for plotting, and a boot check of the env. Each finding comes
with a fix.

| Option | Meaning |
|---|---|
| `--sim <name>` | Simulator to check against (default: workspace config / auto-detect) |

## makker key

View and set the provider API keys Makker saves (in the global `.env`).
Run it from anywhere; it needs no workspace or Julia. This is the way to add a
key on a pip-only install, or to replace a wrong or expired one.

```sh
makker key                # list which keys are set, and where each comes from
makker key openai         # prompt for the OpenAI key and save it (hidden input)
makker key --show         # list status only, never prompt
```

The status flags a key that the process environment or a project `.env`
overrides, since those take precedence over the saved one.

## makker upgrade

Bring everything up to date, in two steps:

1. **Makker itself.** If a newer release exists, upgrade it the right way
   for how it was installed: a tool install runs `uv tool upgrade makker`;
   a dev checkout is pointed at `git pull && uv sync`.
2. **This folder's Julia environment.** A workspace env keeps the simulator
   versions it resolved, so after an upgrade it can be older than the bundled
   skills expect. If it is, update it in place to what the (new) version
   supports, keeping packages added to it, and rebuild the folder's system
   image if it uses one (from a terminal, you choose between rebuilding it and
   no longer using one).

```sh
makker upgrade             # both steps
makker upgrade --check     # report both, change nothing
makker upgrade --env-only  # only this folder's Julia environment
```

| Option | Meaning |
|---|---|
| `--check` | Report the installed and latest versions and this folder's environment; change nothing |
| `--env-only` | Only update this folder's Julia environment |
| `--sysimage rebuild\|off` | If the folder uses a system image: rebuild it after the update, or stop using one (asked when run from a terminal) |

Use `makker init --force` instead to rebuild a workspace's environment
from scratch. A launch in a workspace whose simulator is older than the
installed version supports refuses to start, names both versions, and from a
terminal offers to run the update.

Makker also checks for updates when the web UI and the terminal UI start,
and prints a one-line notice when a newer version exists (cached for a day;
runs in the background so it never delays startup). The web UI shows the
Makker and simulator versions at the foot of the sidebar, with a quiet
"available" note when a newer release exists; clicking it runs the same update
from the browser (see [Upgrading](usage.md#upgrading)). Disable the check with
`MAKKER_NO_UPDATE_CHECK=1`.

## makker transcript

Render a recorded session.

```sh
makker transcript                  # last session in this workspace, HTML
makker transcript <id>             # a specific session
makker transcript --format markdown
makker transcript --bundle         # zip including artifacts
```

## makker sessions

List this workspace's sessions, newest first: start time, id, and the
title derived from each session's first prompt. Any id (or unique prefix)
can be passed to `--resume`.

```sh
makker sessions
makker sessions --limit 0    # show all
```

## makker eval

Run bench suites through Inspect AI (needs `uv sync --extra eval`). See
[evaluation](evaluation.md).

```sh
makker eval --list
makker eval <suite> [<suite>...] --model <provider/model>[,<provider/model>...]
```

| Option | Meaning |
|---|---|
| `--model` | Inspect model id(s), comma-separated for a matrix |
| `--epochs <n>` | Repetitions per sample (default 1) |
| `--max-samples <n>` | Concurrent samples (default 1: sessions only interleave, the workspace root is process-global) |
| `--limit <n>` | Only the first n samples of each task |
| `--log-dir <path>` | Where Inspect writes logs (default: state home `/eval-logs`) |
| `--list` | List available suites |

Exit code 1 when any task errored. Note the model id form differs from the
app's: Inspect uses `provider/model`, the agent uses `provider:model`.
