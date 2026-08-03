# CLI reference

With a tool install (`uv tool install`), every command is just `jutul-agent
…`. From a dev checkout, run them through `uv run jutul-agent …` instead. Two
flags are accepted by every command:

| Flag | Meaning |
|---|---|
| `--workspace <path>` | Workspace directory (default: current working directory) |
| `--state-home <path>` | Where sessions and traces live (default: `$XDG_DATA_HOME/jutul-agent`, falling back to `~/.local/share/jutul-agent`) |

## Interfaces

`jutul-agent` has three interfaces; pick one explicitly (bare `jutul-agent`
prints this list and exits, rather than launching one by default):

| Command | Interface |
|---|---|
| `jutul-agent web [options]` | Browser UI (HTTP + WebSocket server + bundled web app) |
| `jutul-agent tui [options]` | Terminal UI |
| `jutul-agent run "<prompt>" [options]` | One headless turn, then exit |

`jutul-agent --version` prints the version.

### jutul-agent web

```sh
jutul-agent web [--sim <name>] [--approval-mode <mode>] [--model <m>] [--host <addr>] [--port <n>]
```

Serves the browser UI (default <http://127.0.0.1:8742>). One folder is bound to
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
| `--julia-project <path>` | Override the resolved workspace Julia project |
| `--threads <N\|auto>` | Julia compute threads (default: physical cores minus one) |
| `--add-dir <path>` | Add an extra folder the agent can read and edit (repeatable; also runtime `/add-dir`) |
| `--ephemeral-memory` | Use a throwaway memory directory; nothing persists to workspace memory |
| `--host <addr>` | Address to bind (default `127.0.0.1`, localhost only) |
| `--port <n>` | Port to bind (default `8742`) |

### jutul-agent tui

```sh
jutul-agent tui [options]
```

Launches the interactive terminal UI.

| Option | Meaning |
|---|---|
| `--sim <name>` | Active simulator. Required only if not in workspace config and not auto-detectable from a `Project.toml` |
| `--model <provider:model>` | Model for this session. Precedence: this flag, workspace config, user config, `$JUTUL_AGENT_MODEL`, default |
| `--julia-project <path>` | Override the resolved workspace Julia project |
| `--threads <N\|auto>` | Julia compute threads (default: physical cores minus one) |
| `--add-dir <path>` | Add an extra folder for the agent (repeatable) |
| `--continue` | Continue the most recent session in this workspace |
| `--resume [id]` | Resume a session by id or unique prefix; with no value, pick from a list |
| `--approval-mode ask\|workspace\|auto` | Human-in-the-loop policy |
| `--ephemeral-memory` | Throwaway memory directory for this session |

### jutul-agent run

```sh
jutul-agent run "<prompt>" [options]
```

Runs one headless turn and exits — same options as `tui` (headless runs need
`--approval-mode auto`). Exit codes: 0 on success, 3 when the turn needed an
approval that headless mode cannot ask for.

## jutul-agent init (alias: setup)

Bootstrap the workspace: write `.jutul-agent/config.toml`, copy the simulator's
Julia env template, and (by default) precompile the env, plotting backends
included, so the first session in any interface is fast.

| Option | Meaning |
|---|---|
| `--sim <name>` | Simulator to bootstrap (omit to auto-detect) |
| `--source-path <path>` | `Pkg.develop` a local checkout of the simulator package (persisted to the workspace config) |
| `--no-precompile` | Skip the bake — bootstrap config + env only; the first session builds the rest |
| `--force` | Replace an existing workspace env with a fresh template copy (after upgrading jutul-agent) |

## jutul-agent doctor

Diagnose the setup: Julia version, provider key (or Ollama reachability),
which Julia project the workspace resolves to, whether the simulator package
is actually in the manifest, whether the env was built from the current
simulator template, display/xvfb for plotting, and a boot check of the env.
Each finding comes with a fix.

| Option | Meaning |
|---|---|
| `--sim <name>` | Simulator to check against (default: workspace config / auto-detect) |

## jutul-agent key

View and set the provider API keys jutul-agent saves (in the global `.env`).
Run it from anywhere; it needs no workspace or Julia. This is the way to add a
key on a pip-only install, or to replace a wrong or expired one.

```sh
jutul-agent key                # list which keys are set, and where each comes from
jutul-agent key openai         # prompt for the OpenAI key and save it (hidden input)
jutul-agent key --show         # list status only, never prompt
```

The status flags a key that the process environment or a project `.env`
overrides, since those take precedence over the saved one.

## jutul-agent upgrade

Upgrade jutul-agent to the latest version, doing the right thing for how it
was installed: a tool install runs `uv tool upgrade jutul-agent`; a dev
checkout is pointed at `git pull && uv sync`. After upgrading, rebuild any
workspace env that was set up with an older version
(`jutul-agent init --sim <name> --force`).

```sh
jutul-agent upgrade           # upgrade to the latest
jutul-agent upgrade --check   # report latest vs installed, change nothing
```

| Option | Meaning |
|---|---|
| `--check` | Only report the latest available version; don't upgrade |

jutul-agent also checks for updates at launch and prints a one-line notice
when a newer version exists (cached for a day; runs in the background so it
never delays startup). Disable it with `JUTUL_AGENT_NO_UPDATE_CHECK=1`.

## jutul-agent transcript

Render a recorded session.

```sh
jutul-agent transcript                  # last session in this workspace, HTML
jutul-agent transcript <id>             # a specific session
jutul-agent transcript --format markdown
jutul-agent transcript --bundle         # zip including artifacts
```

## jutul-agent sessions

List this workspace's sessions, newest first: start time, id, and the
title derived from each session's first prompt. Any id (or unique prefix)
can be passed to `--resume`.

```sh
jutul-agent sessions
jutul-agent sessions --limit 0    # show all
```

## jutul-agent eval

Run bench suites through Inspect AI (needs `uv sync --extra eval`). See
[evaluation](evaluation.md).

```sh
jutul-agent eval --list
jutul-agent eval <suite> [<suite>...] --model <provider/model>[,<provider/model>...]
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
