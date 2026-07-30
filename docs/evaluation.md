# Evaluation

Evaluation answers one question with numbers: did this change to the
harness, a prompt, a skill, or the model help or hurt? It runs the real
agent, end to end, and grades what actually happened.

The code in `src/jutul_agent/eval` is the harness: the solver, the scorers,
RunConfig, and the `jutul-agent eval` command. jutul-bench is the benchmark
built on that harness. Its public suites live in this repository under
`eval/tasks/`. A private holdout suite lives in a separate repository so
its answers stay out of training data.

The harness is built on [Inspect AI](https://inspect.aisi.org.uk). Inspect
provides the runner: model providers (OpenAI, Anthropic, Google, local
Ollama), repeated epochs, per-sample token and time limits, logs, and a
viewer. Ours is the part Inspect cannot know: a solver that runs one full
jutul-agent session per sample, and scorers that read the session trace.

## Running

```sh
uv sync --extra eval

uv run jutul-agent eval --list
uv run jutul-agent eval canary
uv run jutul-agent eval canary guardrails --model <provider/model>,<provider/model>
uv run jutul-agent eval canary --epochs 5 --epochs-reducer mean,pass_at_2
```

`jutul-agent eval` loads provider keys the way the app does (the user-global
`.env` plus the working directory's) before Inspect resolves the model, and
runs samples one at a time. Logs land in the jutul-agent home under
`eval-logs/`. Inspect them with `uv run inspect view --log-dir <dir>`.

`--model` takes Inspect's `provider/model` ids and defaults to the agent's
default model. A comma-separated list runs a matrix that can mix cloud and
local models in one run.

## How a sample runs

The solver (`eval/solver.py`) builds the same stack a real session uses: a
fresh workspace (with any task fixtures written into it), a Julia kernel, a
`Session`, `build_agent`, `TurnRunner`. It runs inside Inspect's agent
bridge, which intercepts the model client, so the eval's `--model` decides
which provider answers while the agent's tools, skills, prompt, and trace
run unchanged. Memory is ephemeral per sample, so runs cannot learn from
each other.

Two sample metadata keys opt into heavier setup: `needs_env` instantiates
the simulator's Julia environment in the workspace (slow on a cold depot),
and `needs_display` starts a virtual display for plotting.

## Scoring: the trace, not the story

A model can claim anything in its final message, so every task pairs an
answer check with at least one trace check (`eval/scorers.py`):

- `used_tools([...])`: the required tools appear as `tool_call` events.
- `no_interpreters_via_execute()`: the agent did not spawn `julia` through
  the shell (the kernel exists for it, and the workspace backend blocks it
  outright). Tasks can widen the check to other interpreters.
- `artifact_produced(".png")`: the trace records a plot artifact and the
  file on disk is non-empty.
- `investigation_recorded(min_attempts, metric)`: the `attempt` events
  written by `record_attempt` form a tree of at least `min_attempts`
  rationaled steps, each carrying the named metric. This is the scorer
  for investigation tasks, where the recorded process is the deliverable.
- `includes()` (Inspect built-in): the target value appears in the answer.

### Goldens: pin the physics, not the path

The agent takes a different path every run: different probes, different
intermediate code, sometimes a different API route entirely. Goldens work
anyway because they never check the path. They check a physical result,
and a well-posed task makes that result deterministic. The same cell with
the same protocol ends at the same voltage no matter how the agent got the
simulation to run.

That puts a requirement on the task, not the grader: the prompt must pin
every physics-determining choice (which cell, which protocol, which case),
leaving the agent freedom only in *how*. A prompt like "a small two-phase
case" is not golden-able, because the agent legitimately picks grid sizes.
Until the prompt is pinned, it gets structural checks instead
(`numeric_answer`: values in a physical range, ordered the way physics
demands).

Capture is always from a trusted, agent-free run: execute the canonical
case directly in Julia in the instantiated env, read the number, commit it
in the task with a tolerance (`numeric_close(expected, tol)`) and a
provenance comment. The tolerance absorbs solver and version noise, not
model creativity, so measure the noise before trusting one: a result that
depends on which timestep crosses a threshold jitters between runs, a
tolerance tighter than that jitter flakes, and "find the input that hits
value X exactly" can be ill-posed even when X itself is a good golden.
The battery discharge task is the worked example: its golden voltages came
from running the canonical example directly, and the live agent run
independently reproduced them. When a package upgrade legitimately moves a
result, the golden is updated deliberately, with the same direct-run
procedure, never by re-running the agent until it agrees.

## Attribution: the RunConfig

Each sample stores a RunConfig (`eval/runconfig.py`): hashes of the
assembled system prompt, every active skill file, and the instantiated Julia
manifest, plus the jutul-agent commit (and whether the tree was dirty) and
dependency versions. Two runs that differ in exactly one hash answer "did
that change help" mechanically. Treat dirty-tree runs as exploratory, never
as baselines.

This is also the improvement loop: when real use surfaces a failure, add it
as a task first, fix the harness, and let the score movement prove the fix.
The suites then hold the behavior in place. See
[improving the agent](improving-the-agent.md) for a worked example.

## Suites

| Suite | Needs | What it checks |
|---|---|---|
| `canary` | nothing | read a file, eval through the kernel, answer (run first on any change) |
| `guardrails` | nothing | trajectory rules under temptation (no shell julia) |
| `plotting` | env + display | a real, non-empty PNG artifact exists |
| `jutuldarcy` | env | gravity segregation run, structural saturation checks |
| `battmo` | env | chen_2020 CC discharge against captured golden voltages |
| `fimbul` | env | geothermal doublet, production temperatures in °C, cooling |
| `mocca` | env | the canonical cyclic VSA run against captured goldens |
| `calibration` | nothing | iterative two-parameter fit, graded on the recorded attempt tree |

The first `needs_env` run per simulator builds a golden environment under
the jutul-agent home (`eval-envs/<sim>`). Later samples start from a copy
of it, so env preparation takes seconds instead of minutes. A cached env
would otherwise freeze whatever upstream served when it was built, so each
run re-resolves it against the registry on first use: a new upstream
release is picked up automatically, and a run never silently grades
against stale versions.

## Adding a task

A task is data plus a scorer list. Add a `Sample` to an existing suite, or a
new module under `eval/tasks/`:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import includes
from jutul_agent.eval.scorers import used_tools
from jutul_agent.eval.solver import jutul_agent_solver, load_eval_credentials

load_eval_credentials()


@task
def my_suite() -> Task:
    sample = Sample(
        id="short-stable-id",
        input="The user prompt.",
        target="expected substring",
        metadata={
            "fixtures": {"data.csv": "x,y\n1,2\n"},  # written into the workspace
            "needs_env": False,
            "simulator": "jutuldarcy",
        },
    )
    return Task(
        dataset=[sample],
        solver=jutul_agent_solver(),
        scorer=[includes(), used_tools(["run_julia"])],
        time_limit=600,
        token_limit=200_000,
        message_limit=50,
    )
```

Set the limits: they are the cost cap on a runaway loop. One gotcha:
Inspect's `token_limit` counts cache-read tokens, which on multi-turn
sessions run ~20x the billed input. Size sim-task budgets in the millions
and let `time_limit`/`message_limit` carry the real cost guard. A suite
module exposes one task named like the module, or several through a
module-level `TASKS` list of task factories. New suite modules are picked
up by `jutul-agent eval --list` automatically, and a unit test in
`tests/test_eval_bench.py` imports every suite so API drift fails fast.

## Public and private task sets

The bench is designed for a two-tier dataset, the same pattern reference
benchmarks use:

- The public set is the in-repo suites: the contributor example, the
  development loop, and what anyone can reproduce. Assume anything public
  on GitHub eventually enters model training data.
- The private set is a holdout for honest measurement: a separate private
  repository with the same layout plus an agent-free reference solution
  per task, run by passing paths (`jutul-agent eval path/to/task.py`),
  since nothing in the harness assumes the bundled location. Every task
  file carries a canary GUID. Results from the holdout are the reportable
  numbers. If it is ever published, it stops being a holdout, and a new
  one is rotated in.

The placement rule: a task goes public when leaking it does not
invalidate what it measures, private when it does. Behavior rules
(guardrails), structural checks, and one documented worked example per
pattern survive leaking, because they pin behavior rather than claim
capability: a model that memorizes them and behaves accordingly is doing
exactly what they ask. Trap phrasings (honesty tasks), capability goldens
meant for reportable numbers, and any task whose value is that the model
has not seen it go private. When a public and a private task share a
workflow, vary the physics knob so the private golden discriminates: the
private answer must sit outside the public golden's tolerance band.

## Running with Docker

The repo ships a container setup for isolated runs:

```sh
docker compose build
docker compose run --rm eval canary
docker compose run --rm eval battmo --model <provider/model>
```

Provider keys come from `./.env`, and `network_mode: host` lets
`ollama/<model>` reach the host's Ollama. Two named volumes do the heavy
lifting: the Julia depot and the jutul-agent state home (golden envs, eval
logs). Only the first run per simulator pays the instantiate-and-precompile
cost. Everything after reuses the caches. Logs land on the `jutul-state`
volume. Copy them out, or point `--log-dir` at a bind mount to keep them on
the host.

## Parallelism and timing

How runs scale, and the costs that dominate:

- Within one container or process, samples run serially. The agent's
  workspace root is process-global, and one Julia kernel per sample is
  already CPU-hungry. `--max-samples 1` is the default and the solver
  enforces it with a lock.
- Across containers, runs are embarrassingly parallel. The natural unit is
  one container per (suite, model) pair: `docker compose run -d` several,
  or let a CI matrix do it (one job per pair). Containers can share the
  depot volume, since Julia's package manager handles concurrent access
  with its own locks. Give each container a few CPUs (the compose file caps at 4):
  the kernel runs the solve, and Xvfb plus the Python side need little.
- Time limits are per-sample cost caps, not stopwatches. Every task sets
  `time_limit` (wall clock) and `token_limit`. Budget them for the worst
  legitimate path: a task whose point is installing a package must absorb
  a `Pkg.add` plus precompile, so it gets its own generous limit rather
  than inflating every other task's.
- Keep the common path warm. The golden-env cache exists so ordinary
  `needs_env` samples never pay instantiate time. Only tasks that
  deliberately change the environment should ever compile, and with the
  depot volume even those pay once.
- Timing is a metric as well as a limit. Inspect records wall-clock per
  sample, and the trace's timestamps plus `model_usage` events separate
  model time from Julia time. Compare timings only within one machine (the
  RunConfig records the platform): cross-machine wall-clock is noise.

## Running safely

Bench samples run with `--approval-mode auto`: nobody reviews the agent's
shell commands, and `run_julia` is never gated (see
[approval and safety](approval.md)). For the bundled suites (our own
prompts, major-provider models) the practical risk is accidental damage,
and each sample already runs in a throwaway workspace with ephemeral
memory. Two graduations when the trust level drops:

- Scheduled or bulk runs belong on CI runners, which are ephemeral VMs:
  isolation for free.
- Adversarial tiers and third-party task packs should run inside a
  container or VM. That boundary, not approval, is what contains a
  destructive command.

## Current limits

- OpenAI, Anthropic, Google, and Ollama targets are verified working.
  Gemini needs a local workaround for an Inspect bug, where thought
  signatures cross the bridge as urlsafe base64 but are decoded as
  standard. `eval/_gemini_compat.py` patches the decode and should be
  removed once fixed upstream.
- Samples run serially. The agent's workspace root is process-global, so
  parallelism needs per-sample isolation first.
- The first `needs_env` run per simulator pays a full env instantiate on a
  cold Julia depot, which can take many minutes on a fresh machine.
