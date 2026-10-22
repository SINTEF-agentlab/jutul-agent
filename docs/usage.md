# Installation and usage

## Installation

You need two tools on PATH:

- [uv](https://docs.astral.sh/uv/getting-started/installation/), which
  installs Python and manages the tool
- Julia 1.10 or newer, via [juliaup](https://github.com/JuliaLang/juliaup)

On headless Linux, plotting also needs `xvfb`.

Install jutul-agent as a uv tool. This puts a `jutul-agent` command on your
PATH that runs from any folder:

```sh
uv tool install jutul-agent
```

`jutul-agent upgrade` keeps it current (see [Upgrading](#upgrading)).

API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`) can go in
your environment or a `.env`, but you don't have to manage that file: run
`jutul-agent key <provider>` to set or replace a key (or just `jutul-agent
key` to see what's set), and the web UI and TUI prompt for a missing key on
their own. Local models through [Ollama](https://ollama.com) need no key.
`jutul-agent doctor` verifies the whole setup and prints a fix per finding.

Prefer a clone for working on jutul-agent itself? See
[development](development.md), where you run commands as `uv run jutul-agent`.

## Workspaces

`jutul-agent` runs in the directory you invoke it from. That directory is the
workspace: the agent reads and writes files there, and the Julia environment
lives in `.jutul-agent/julia-env/` inside it. Start it from a project folder.
Initialise once per folder, then start the agent there:

```sh
mkdir my-reservoir-study && cd my-reservoir-study
jutul-agent init --sim jutuldarcy
jutul-agent                         # the web UI, at http://127.0.0.1:8742
```

`init` writes `.jutul-agent/config.toml`, copies the simulator's Julia env
template, and **precompiles by default**: it instantiates the env and warms the
precompile caches, including the terminal and web plotting backends. The first
time can take a while (Julia compiles the simulator and the plotting stacks),
after which the first session in any interface starts in seconds. Useful
variants:

```sh
jutul-agent init --sim jutuldarcy --no-precompile                      # quick bootstrap, bake later
jutul-agent init --sim jutuldarcy --source-path /path/to/JutulDarcy.jl
jutul-agent init --sim jutuldarcy --force                              # rebuild env from the template
```

`--no-precompile` skips the bake for a fast bootstrap — the first session then
builds what's missing (the ~minutes wait moves to first use). `--source-path`
dev-links the simulator to a local checkout the agent can read and edit at its
real path (a dev checkout, so it is writable, unlike a registry install in the
shared depot). In a managed env, the checkout's own `[compat]` then decides the
simulator and the packages it depends on, so a checkout ahead of the registry
can use a newer Jutul (see
[trying a new simulator version](adding-a-simulator.md#trying-a-new-simulator-version)).
`--force` rebuilds the env from the template, discarding packages added to it;
`jutul-agent upgrade` updates it in place instead (see [Upgrading](#upgrading)).
`setup` is an alias for `init`. If you skip `init`
entirely, the first run bootstraps the workspace and auto-detects the simulator
from a `Project.toml` when it can.

One simulator per workspace. Different simulators have incompatible Julia
dependencies, so use a separate folder for each. Pointing an existing
workspace at another simulator rebuilds the env from the new template on the
next run.

## Upgrading

jutul-agent is actively developed, so keep it current:

```sh
jutul-agent upgrade          # upgrade jutul-agent, then this folder's Julia environment
jutul-agent upgrade --check  # just report what's installed, what's latest, and the folder's env
```

`upgrade` does the right thing for how you installed: a tool install runs
`uv tool upgrade jutul-agent`; a dev checkout is told to `git pull && uv sync`.
Then, run in a workspace, it brings that folder's Julia environment up to the
simulator versions the installed jutul-agent supports (`--env-only` does just
this part).
On Windows the running executable can't replace itself, so the upgrade runs in
a new console window and jutul-agent exits to release the file; reopen it when
that window finishes.
At launch, jutul-agent also prints a one-line notice when a newer version is
available (a background check, cached for a day; it never delays startup), and
the web UI shows the jutul-agent and simulator versions at the foot of the
sidebar. Turn the check off with `JUTUL_AGENT_NO_UPDATE_CHECK=1`.

The web UI can do the same update without a terminal. When a release is out,
the foot of the sidebar says so; click it to update. jutul-agent upgrades
itself, updates the folder's Julia environment, restarts, and the page reloads
into the new version. Open chats are closed for it (their conversations are
kept), and it waits while a session is starting or working. It only does this
when jutul-agent runs on your own computer and was installed with `uv tool
install`. On Windows, where a running program can't replace its own files,
jutul-agent closes and the update runs in a new console window. jutul-agent
then keeps running in that window (Ctrl+C there stops it), and the window's
output is also saved in `~/.local/share/jutul-agent/update/update.log`. If an
update fails, the page shows its output, also after it reloads into the
restarted jutul-agent. An outdated Julia environment gets an "Update
environment" link in the same place, on every system.

An upgrade ships new simulator env templates, but existing workspace envs keep
the packages they resolved. When a workspace's simulator is older than the
installed jutul-agent supports, launch refuses to start and, from a terminal,
offers to update the env in place (the web UI offers it too). To do it yourself:

```sh
jutul-agent upgrade --env-only        # in place, keeping packages added to the env
jutul-agent init --sim <name> --force  # from scratch
```

If the folder starts from a [system image](sysimage.md), the update also has to
rebuild it (about 20 minutes); from a terminal you can choose to stop using an
image instead.

A simulator newer than the installed jutul-agent was made for is not refused: it
runs with a note at launch, since the bundled skills may not cover what changed.
That is what a simulator developer trying an unreleased version gets, and an
update never moves it back.

## Adding folders

Add folders outside the workspace so the agent can use them with the same
file tools:

```sh
jutul-agent --add-dir ../shared-data --add-dir ~/datasets/spe10
```

`jutul-agent tui` takes the same flag. During a session, `/add-dir <path>`
adds one immediately, in the web UI and the TUI; in the TUI, `/add-dir` alone
lists the folders added so far. The agent uses each at its real absolute path in
every tool: the file tools, `run_julia`, and `execute`. Added folders last
for the session and are not written to config.

## Interfaces

With no command, `jutul-agent` starts the web UI (`jutul-agent web` is the
same). The terminal UI and headless runs have their own commands:

```sh
jutul-agent                  # web UI: chat with interactive plots and reports
jutul-agent tui              # terminal UI
jutul-agent run "<prompt>"   # one headless turn, then exit
```

All three are the same agent and session core, so a session started in one
resumes in another. The web UI is covered in [the web UI](web-ui.md) and
[the server interface](server-interface.md); the terminal UI and headless runs
are below.

## The TUI

```sh
jutul-agent tui
```

| Command | Effect |
|---|---|
| `/model` | Open the model selector (or `/model <provider:model>`) |
| `/key` | Show provider key status (or `/key <provider>` to set or replace one) |
| `/approval-mode` | Set approval policy: `ask`, `workspace`, `auto` |
| `/add-dir <path>` | Add an extra folder |
| `/context` | Show context usage: tokens held vs the model's window |
| `/compact` | Summarize older turns to free context space |
| `/memory` | View workspace memory (`/memory edit` opens it in `$EDITOR`) |
| `/transcript` | Write the session transcript (add `md` for markdown) |
| `/copy` | Copy the last assistant message |
| `/clear` | Clear the visible log |
| `/help` | List commands |
| `/quit` | Exit |

| Key | Effect |
|---|---|
| `Ctrl+C` | Interrupt the running turn. With text selected, copy. Twice when idle, exit |
| `Ctrl+G` | Cancel the in-flight turn (resets Julia if needed) |
| `Ctrl+O` | Toggle tool and reasoning cards between preview and full output |
| `Ctrl+L` | Clear the visible log |
| `Shift+Tab` | Cycle approval mode |
| `Ctrl+P` / `↑` | Previous history entry |

Approval modes: `ask` (default) prompts before shell commands and file edits,
`workspace` auto-allows file writes inside the workspace, `auto` allows all
side-effecting tools.

## Resuming a session

The conversation survives the process: every session checkpoints its
thread, so you can pick an earlier one back up. The web UI lists past
sessions in its sidebar and reopens one where you left off. In the terminal:

```sh
jutul-agent tui --continue          # reopen the most recent session
jutul-agent tui --resume            # pick from a list of recent sessions
jutul-agent tui --resume 2026-06-12 # by id, or any unique prefix
jutul-agent sessions                # list what's resumable
```

A resumed TUI replays the prior exchanges and the model continues with the
full conversation. Sessions are named by start time plus a short suffix
(`2026-06-12-2315-3f2a`), so listings sort chronologically; the first
prompt also titles the session, and its output folder under
`jutul-agent-output/sessions/` carries that title as a slug.

One thing does not survive: the Julia REPL restarts with the process, so
variables and loaded packages from earlier turns are gone (files and
artifacts on disk remain). The agent is told this and re-runs setup it
needs.

## Headless turns

`jutul-agent run` takes a prompt, runs one turn, and exits:

```sh
jutul-agent run --approval-mode auto "Plot the voltage curve for the chen_2020 cell."
```

Headless mode cannot pause for approval, so use `--approval-mode auto` (the
run exits with an error if it would have needed to ask). `--ephemeral-memory`
gives the run a throwaway memory directory.

## Transcripts

Every session appends an event log to
`$XDG_DATA_HOME/jutul-agent/workspaces/<hash>/sessions/<id>/trace.sqlite`:
prompts, responses, every tool call with arguments and output, artifacts,
token usage. Render one:

```sh
jutul-agent transcript                   # last session, HTML
jutul-agent transcript <id>              # specific session
jutul-agent transcript --format markdown
jutul-agent transcript --bundle          # zip with artifacts included
```

## Models

Model ids are `provider:model` strings. Resolution precedence: `--model`
flag, workspace config, user config (`Ctrl+A` in the selector), the
`JUTUL_AGENT_MODEL` environment variable, then the default.

`/model` opens the selector: bundled OpenAI, Anthropic, Google, OpenRouter, and Ollama
models, plus anything `init_chat_model` supports typed as `provider:model`.
Switching mid-session keeps the conversation. Missing API keys are prompted
for and saved to a user-global `.env` (never to config files).

### Local models (Ollama)

Pick an `ollama:` model in the selector. No key is needed. The list shows
recommended models, your installed ones, and Ollama Cloud models, and pulls
anything missing. Requirements and caveats:

- The model must support tool calling (`ollama show <model>` lists `tools`
  under Capabilities). Keep Ollama itself current: an outdated Ollama can
  fail to parse a new model's template and silently lose tool support.
- The agent's prompt is large. Local models load with a context window sized
  to what the model supports, capped by a memory budget (64K by default,
  lowered with `JUTUL_AGENT_OLLAMA_NUM_CTX` on tight hardware).

### Other providers

Add the provider's LangChain package to the tool install, then use its
`provider:model` id (replace the package placeholder below):

```sh
uv tool upgrade jutul-agent --with 'langchain-<provider>'
```

(From a dev checkout, use `uv add 'langchain-<provider>'` instead.) If the
package is missing, jutul-agent names the exact one to install.
OpenAI, Anthropic, Google, OpenRouter, and Ollama are already bundled.

## Troubleshooting

Start with the built-in check:

```sh
jutul-agent doctor
```

It verifies Julia is on PATH and 1.10+, the active model's key is set (or
Ollama is running for a local model), which Julia project the workspace
resolves to, that the simulator package is actually resolved in the env
manifest, that a display (or xvfb) is available for plotting, and that Julia
boots cleanly in the env. Each failure comes with a one-line fix.

Common cases:

- Workspace versus launch directory: the workspace is where you launch from,
  not where you ran `init`. `cd` there or pass `--workspace <path>`.
- A `Project.toml` at the workspace root takes precedence over
  `.jutul-agent/julia-env/`. If that root project lacks the simulator
  packages, `using <Sim>` fails even though Julia starts. `doctor` flags
  this.
- "Julia failed to start before the kernel was ready": the Julia subprocess
  crashed during startup. The message includes Julia's own error and the
  path to `julia-startup.log`. This is usually a stale env, rebuilt with
  `init --sim <name> --force`.
- Headless Linux without `xvfb`: simulation works, plotting errors. Install
  it (`sudo apt-get install -y xvfb`).
