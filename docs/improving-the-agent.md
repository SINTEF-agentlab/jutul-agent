# Improving the agent

The harness has a small number of levers. Knowing which one to pull, and how
to verify the change did anything, is most of the work.

## Choose the right lever

| You want to change | Lever |
|---|---|
| A rule that must always hold | System prompt ground rules (`agent/prompts.py`) |
| Domain knowledge for a kind of task | A skill (`simulators/*/skills/`, `shared_skills/`) |
| Behavior that must be guaranteed, not requested | A tool that does the right thing (`agent/tools.py`) |
| A separate context for a sub-task | A subagent (adapter `subagent_factories`) |
| Facts about one user's setup or preferences | Workspace memory (the agent maintains it) |
| The simulator's own ergonomics | Upstream change to the package |

## System prompt versus skills

Skills are progressively disclosed: the model always sees every skill's name
and description, but reads a skill's body only when it decides to. That makes
the body the right place for workflows and domain detail, and the wrong place
for rules that must hold on every turn. A model that never opens the skill
never sees them.

Always-on rules go in `prompts.py`. Everything else can be a skill.

Writing skills: keep the description specific enough that the model knows
when to open it, point at source to read (a package's `pkgdir`, its
`examples/`) and probes to run (`@doc`, `methods`) rather than restating documentation,
and prefer one worked example over prose. Keep frontmatter valid YAML.

## Tools over instructions

If a behavior matters, encode it in a tool instead of asking the model
nicely. `plot_julia` rejects empty figures rather than instructing the model
to check. `record_attempt` makes investigation structure a tool contract
rather than a convention. Include rewriting in the kernel makes relative
paths just work instead of documenting a workaround. Prompt text asking for
discipline is the weakest version of every one of these.

## Subagents

An adapter can contribute subagents via `subagent_factories`: each factory
takes the session and returns a deepagents subagent spec. Use one when a
sub-task benefits from its own context window and toolset (a long literature
lookup, a parameter sweep with noisy output). None of the bundled simulators
ship one yet, but the seam exists.

## Memory

Each workspace has agent-maintained memory: an index (`MEMORY.md`) that is
always in the prompt, and one markdown file per fact, read on demand. The
agent writes it through the `remember` tool and ordinary file edits. If the
agent keeps re-learning something about your setup, ask it to remember the
fact, and check what it wrote.

## Making a simulator agent-friendly

The biggest improvements often live upstream, in the simulator package
itself:

- Discoverable APIs: meaningful docstrings on the entry points, a small
  number of canonical setup functions, consistent keyword names.
- Runnable examples in the repository. The agent reads a package's
  `examples/` (at its `pkgdir` path) and imitates them, so a missing example
  is a missing capability.
- Structured returns rather than printed tables. A `Dict` of vectors can be
  checked and plotted. A pretty-printed summary cannot.
- Errors that say what to do: "expected a `ScalarField`, got `Matrix`" beats
  a deep stack trace.
- A PrecompileTools workload, so the agent's warm package has something to
  build on and first solves are fast.

These changes help every user of the package. The agent is just an
unusually systematic one.

## Verify with the bench

Every lever above changes an input that makker-bench hashes (prompt, skills,
manifest, code version). The workflow:

1. Run the relevant suite before the change.
2. Change one thing.
3. Run it again and compare. The RunConfig hash that differs tells you the
   comparison is the one you meant to make.

When real use surfaces a failure, add it as a task first, then fix it, so
the fix is proven by a score that moves and pinned against regression. See
[evaluation](evaluation.md).
