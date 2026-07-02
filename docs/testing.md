# Testing

How the test infrastructure works: what runs where, what gates what, and
the conventions that keep an agent system testable at all.

## Three tiers

| Tier | Selected by | Needs | Time |
|---|---|---|---|
| Unit | `uv run pytest` (default) | nothing beyond the venv | ~35 s |
| Integration | `-m integration` | Julia (some tests an instantiated env) | seconds to minutes |
| Live | `pytest tests/live/` | a provider API key | one real agent turn |

The default run must work on a fresh clone with no Julia, no keys, and no
network. Anything that cannot promise that is marked `integration` or
lives in `tests/live/`.

## Gating: tests skip themselves

Integration tests check their own prerequisites and skip with a reason
rather than fail: Julia on PATH, an env template with a generated
`Manifest.toml`, a display or Xvfb for GL tests. This is what makes one
test suite serve three environments: a laptop without Julia, a dev box
with everything, and CI lanes that instantiate exactly what their job
needs. The simulator smoke tests are the pattern: parametrized over every
registered adapter, each instance skipping unless that simulator's env is
instantiated.

The live smoke test gates on a provider key, runs one real turn (read a
file, evaluate through the kernel, answer), asserts on the answer *and* on
the trace, and retries once, a deliberate compromise with model
nondeterminism. Proper repetition statistics belong to the bench, not the
test suite.

## Fakes, not mocks

`jutul_agent.lab.fakes` provides hand-written fakes (a fake simulator
adapter, a fake Julia session that returns scripted results, scripted
agents) instead of patch-based mocks. Tests build real objects (`Session`,
backends, tools) around a fake edge, so they exercise real wiring and
survive refactors that would break patch paths. They live in the package
rather than under `tests/` because the same fakes drive the shipped lab
harness and the example apps. The eval scorer tests follow the same idea:
they write a real `trace.sqlite` with synthetic events and run the actual
scorer against it.

## Snapshots

Prompt assembly and other rendered text are snapshot-tested with syrupy.
A change to the system prompt shows up as a snapshot diff in review:
deliberate changes are accepted with `--snapshot-update`, accidental ones
get caught. Treat a snapshot update in a PR as a prompt change to review,
not noise.

## TUI tests

Every widget that renders model-controlled text gets a Textual pilot test
(drive the real app headless, press keys, assert on the screen). The
failure class this guards is real: model output containing markup-like
text can crash or corrupt a widget that renders it as markup. New TUI
surfaces ship with a pilot test that feeds them hostile text.

## Kernel tests

The Julia kernel has two layers. Unit tests drive `connection.py` over an
in-process socketpair, playing the Julia side byte-for-byte: framing,
routing, and failure paths with no Julia at all. Integration tests run the
real kernel: eval round-trips, fd-level output capture (C `printf`),
interrupt-under-load, and cancel recovery. Two of them are the protocol's
safety net and intentionally stress race conditions
(`test_interrupt_during_heavy_printing_repeatedly`, the bounded cancel
test). Treat a flake there as a real bug until proven otherwise.

## Eval tests

`tests/test_eval_bench.py` runs entirely offline: scorers against
synthetic traces, RunConfig hashing stability, and an import-and-build of
every task suite so API drift in the entrypoint files fails fast. The
whole module skips when the `eval` extra is not installed. Live model
calls never run under pytest. They are bench runs
([evaluation](evaluation.md)).

## What CI runs

- `ci.yml`: lint, the unit suite on Linux/macOS/Windows, the kernel
  integration job, and the plot job (instantiates the JutulDarcy env under
  xvfb, renders a real GLMakie figure).
- `simulators.yml`: per-simulator env instantiate + smoke, on PRs and
  weekly. The weekly schedule is the upstream-breakage canary, since envs
  carry no version pins. Both instantiate steps use an explicit
  `Pkg.precompile()` because best-effort auto-precompile exits 0 on
  failure (see [development](development.md)).

## Conventions for new tests

- Unit by default. Mark `integration` the moment a test needs Julia.
- Gate on prerequisites with a skip and a reason, never a hard fail.
- Test through public seams (`JuliaSession`, backends, `TurnRunner`), not
  internals, so implementation rewrites keep their tests.
- When a real failure surfaces in use, pin it: a regression test if it is
  mechanical, a bench task if it is behavioral.
