# The lab: testing and improving the app without a live session

The agent's UI and runtime are normally driven by a real model and a real Julia
process, which makes iterating on them slow: you have to start a session and try
each thing by hand. The lab (`makker.lab`) drives the same stack
headlessly with test doubles, so an agent or a developer can render every UI state,
look at it, change something, and re-render, and can profile cold start, all without
a live session.

It is built on the shared doubles in `makker.lab.fakes`: a scripted model
and agent, a fake Julia session, and a fake simulator adapter. The test suite uses
the same doubles.

## The TUI lab

A scenario is a named recipe of an agent plus a short sequence of interactions that
puts the TUI into a state worth looking at. The lab drives one through the real
Textual app and captures what it renders.

```bash
python -m makker.lab.tui list
python -m makker.lab.tui run tool_call --png
python -m makker.lab.tui all --png -o ./out
```

Each scenario writes three artifacts under `<out>/<name>/`:

- `screen.svg`: the screenshot Textual exports.
- `screen.txt`: a plain-text view of the screen, recovered from the SVG, for grep
  and diffs.
- `transcript.md`: the session transcript.

With `--png` it also rasterises the SVG to a PNG with a headless browser (Chrome,
Chromium, or Edge), so an agent can view the UI directly. If no browser is found the
SVG and text are still written.

### The self-improvement loop

This is how an agent improves the UI on its own:

1. `python -m makker.lab.tui run <scenario> --png` and open the PNG.
2. Spot a problem (clutter, a leaked internal value, bad wrapping, a confusing
   state).
3. Edit the widget or stylesheet under `src/makker/interfaces/tui/`.
4. Re-render and compare. Run `pytest tests/test_lab.py` to confirm nothing
   crashes.

The approval card, for example, used to print an internal interrupt id; rendering the
`approval` scenario made that obvious, and the fix was one line.

### Adding a scenario

Add a call to `scenario(...)` in `makker/lab/scenarios.py` with a builder
that returns a scripted agent and a tuple of steps. A step is a prompt string to
submit, or an async callable taking the Textual pilot for key presses. Keep scenarios
small and deterministic.

## Live mode (real model, real Julia)

The scripted scenarios are fast and deterministic but cannot show what a live model
and a live solve actually render. Live mode runs one prompt through the full stack
and captures the screen:

```bash
python -m makker.lab.live "compute the mean of [1.0, 2.0, 3.0] in Julia" \
    --workspace testbed/jutuldarcy --model openai:gpt-5.4 --png
```

It costs API and needs a workspace with an instantiated Julia env. A cold first turn
pays Julia compilation, so it can be slow; raise `--timeout` (and the settle window)
for heavier prompts, or warm the env first. Everything is wrapped in an overall
timeout so a hung kernel cannot stall a run.

## Robustness

The scenarios tagged `robustness` are the fault and edge cases: a long output, a
unicode-heavy result, a tool error, a tiny terminal, a wide unbroken line, an empty
answer, and approving a pending request. `tests/test_lab.py` renders every
scenario, so it is a standing guarantee that the TUI does not crash on any of them.
Add a scenario for any new state you want held to that guarantee.

## Profiling

Two profilers, for the two kinds of slowness.

Cold start (imports and bringing the app up):

```bash
python -m makker.lab.profile_startup
python -m makker.lab.profile_startup --module makker.interfaces.tui
```

It runs a fresh interpreter under `-X importtime` (heaviest imports, rolled up per
top-level package) and times the post-import phases (creating a session, building the
agent graph) with the fakes. The cold-start cost is dominated by third-party imports,
chiefly the Anthropic SDK (pulled eagerly by deepagents), then Textual and LangSmith.
A guard test (`tests/test_cli_startup.py`) keeps the CLI entry from importing the
agent stack, so light commands like `--version` stay fast.

The hot path (our own Python during a turn):

```bash
python -m makker.lab.profile_turn --scenario long_output --repeat 5
```

It cProfiles rendering a scenario through the full agent runtime and TUI with the
fakes, and reports the Makker functions by self-time. Use it to find a slow
render or middleware path, and to catch a regression that makes turns sluggish.

Run any of these before and after a change to measure the effect.
