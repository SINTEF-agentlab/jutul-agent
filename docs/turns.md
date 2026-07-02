# Turns

A turn is the unit of interaction: one user prompt in, one assistant
answer out, with any number of model calls and tool calls in between. The
agent loop keeps going (model proposes tool calls, tools run, results
feed back) until the model answers without requesting tools, or a gated
tool needs approval.

Everything that runs the agent funnels through one class,
`TurnRunner` (`agent/turns.py`): the TUI, the headless CLI, the web
server, the live smoke test, and the bench solver. That is deliberate:
there is exactly one place where a turn's streaming, interrupts, and
trace bookkeeping are defined, so every interface and the bench exercise
the same behavior. One level up, `SessionHost.drive_turn` wraps the
runner with the shared policy loop: it auto-resumes past interrupts the
approval mode or the session allowlist already permits, and runs the
end-of-turn duties (report transcript refresh) when the turn settles, so
those cannot drift between front ends either.

## Anatomy of a turn

```
run_prompt(prompt)
  └─ agent.astream_events(...)        the deepagents/langgraph loop
       ├─ model call ── tool calls ── tool results ── model call ── ...
       └─ final assistant message (or an approval interrupt)
  returns TurnRunResult(messages, interrupts)
```

The runner consumes deepagents' typed projection streams:

- `run.messages` yields one message stream per graph node. Only the
  `model` node's stream is the assistant's visible output (text and
  reasoning deltas, forwarded to the UI as they arrive). Every other
  node's stream (tool results, middleware) is drained but not rendered;
  rendering them would paste raw tool output into the chat as if the
  assistant had typed it.
- `run.tool_calls` yields tool lifecycle events (started, output delta,
  finished). This is how `run_julia`'s live output streams into the TUI
  while a solve runs: the kernel's output chunks are forwarded as
  tool-output deltas.
- Interrupts are collected at the end: if a gated tool paused the graph,
  the result carries the pending requests instead of a final answer.

## Approval round-trips

A turn that hits a gated tool returns with `interrupts` set. The caller
shows the request and resumes with the decision:

```python
result = await host.drive_turn(lambda: runner.run_prompt(prompt), approval_mode=mode)
if result.interrupts:  # only interrupts the policy could not answer itself
    decision = ask_the_user(result.interrupts)
    result = await runner.resume(decision)
```

Resume re-enters the same graph thread, which is why the agent is built
with a checkpointer: conversation state persists across the pause. The
TUI drives this loop interactively. Headless mode refuses (exit code 3)
under `ask` because there is nobody to ask, and the bench treats an
unexpected interrupt as a failed sample. See
[approval and safety](approval.md).

## The trace contract

The runner owns the turn's bookkeeping so call sites cannot forget it: it
records `message_user` when a prompt enters and `hitl_response` when a
resume carries a decision. Everything in between (assistant messages,
reasoning, tool calls and results, token usage) is recorded by the trace
middleware as it happens (see [the trace database](trace.md)). One turn
therefore reads back from the trace as: `message_user`, then alternating
`tool_call`/`tool_result` and `message_*` events, ending in the final
`message_assistant`.

## Boundaries

State that outlives a turn lives elsewhere by design: conversation history
in the checkpointer, knowledge in [memory](memory.md), results in the
workspace and the trace. A turn itself is stateless plumbing, which is
what makes it equally usable from a chat UI, a one-shot CLI call, and an
eval harness.
