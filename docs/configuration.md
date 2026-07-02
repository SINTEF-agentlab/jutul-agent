# Configuration

Three layers, each optional: workspace config, user config, environment.
For the model, precedence is `--model` flag, then workspace, then user
config, then `$JUTUL_AGENT_MODEL`, then the built-in default.

## Workspace config

`.jutul-agent/config.toml` in the workspace, written by `init` and by the
TUI when you save a choice:

```toml
simulator = "jutuldarcy"           # active simulator for this workspace
model = "<provider:model>"         # optional, overrides the user default
approval_mode = "workspace"        # optional: ask | workspace | auto

[simulators.jutuldarcy]
source_path = "/path/to/JutulDarcy.jl"  # dev-link, set by init --source-path
```

This file is safe to commit: API keys never go here.

## User config

`<state home>/config.toml` (default `~/.local/share/jutul-agent/config.toml`)
holds user-wide defaults, currently the model:

```toml
model = "<provider:model>"
```

Set it from the TUI model selector with `Ctrl+A` (save for all workspaces).

## API keys

Keys are read from the process environment, a `.env` in the working
directory, and the user-global `<state home>/.env`, in that order of
precedence. The global file is what the interactive key prompts write to,
with owner-only permissions. Recognized:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`. Ollama needs none.

You never have to find or edit that file by hand. Set or change a key in
whichever interface you use:

- **Command line:** `jutul-agent key` lists which provider keys are set (and
  where each comes from); `jutul-agent key <provider>` (e.g. `jutul-agent key
  openai`) prompts for a key and saves it. This is the way to replace a wrong
  or expired key, and it needs no checkout, workspace, or Julia.
- **Terminal UI:** `/key` shows the same status; `/key <provider>` opens a
  prompt to set or replace one.
- **Web UI:** if the selected model has no key, a dialog asks for it before the
  session starts; the top-bar **Keys** button opens the same dialog any time.

Because the process environment and a project `.env` take precedence over the
saved file, a key set there shadows the saved one. The status views flag this
so a stale shell variable can't silently override a key you just changed.

## Environment variables

| Variable | Meaning |
|---|---|
| `JUTUL_AGENT_MODEL` | Default model when no flag or config sets one (useful in CI) |
| `JUTUL_AGENT_JULIA_THREADS` | Compute threads for the Julia kernel: an integer, or `auto` for all logical cores. Defaults to physical cores minus one; the `--threads` flag overrides it. Jutul's assembly and preconditioner are threaded, so this speeds up larger solves. The kernel adds one interactive thread on top, and pins OpenBLAS to one thread (unless `OPENBLAS_NUM_THREADS` is set) to avoid oversubscription. The eval/benchmark harness uses the same policy (production parity); pin it per run with this variable. |
| `JUTUL_AGENT_HYPRE_THREADS` | OpenMP threads for HYPRE's BoomerAMG (JutulDarcy's CPR pressure preconditioner). Defaults to physical cores minus one, capped at 8 (HYPRE's solver performance degrades with many threads). Independent of the Julia compute-thread count. |
| `JUTUL_AGENT_OLLAMA_NUM_CTX` | Memory cap for local models' context window (default 64K tokens, lower on tight hardware) |
| `JUTUL_AGENT_NO_XVFB` | Opt out of starting a virtual display on headless Linux (plotting then errors at use) |
| `JUTUL_AGENT_NO_OPEN` | Never open artifacts in the OS default application (CI, tests) |
| `JUTUL_AGENT_NO_UPDATE_CHECK` | Skip the once-a-day release check at launch |
| `JUTUL_AGENT_REVIEW` | Developer tool: set to `1` to auto-review every finished turn (see [session review](review.md)) |
| `JUTUL_AGENT_REVIEW_MODEL` | Model the reviewer uses (default `openai:gpt-5.4`) |
| `XDG_DATA_HOME` | Relocates the state home (`<value>/jutul-agent`) |

## What lives where

In the workspace:

```
<workspace>/
  .jutul-agent/
    config.toml          # workspace config (committable)
    julia-env/           # the simulator's Julia environment
  jutul-agent-output/
    sessions/<date>-<id>/
      artifacts/         # plots and reports the agent saved
```

In the state home (default `~/.local/share/jutul-agent/`):

```
config.toml              # user config
.env                     # saved API keys (owner-only)
workspaces/<hash>/
  memory/                # the workspace's agent-maintained memory
  sessions/<id>/
    trace.sqlite         # the session event log
    checkpoints.sqlite   # conversation state (resume, model switching)
    julia-startup.log    # kernel boot log, referenced by error messages
eval-logs/               # Inspect logs from jutul-agent eval
eval-envs/<sim>/         # golden envs the bench copies per sample
```

The state home is keyed by a hash of the workspace path, so the same folder
always maps to the same memory and session history.
