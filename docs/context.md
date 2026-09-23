# Context handling

What the model actually sees on a turn, where it comes from, and how growth
is managed.

## What enters the context

Every turn, the model receives:

- The system prompt, assembled per session (`agent/prompts.py`): the
  harness ground rules, the active simulator's description and domain
  hints, and runtime context (paths, plotting availability).
- The skills index: every skill's name and description. Skill bodies are
  not in the context until the agent reads one. That is the
  progressive-disclosure contract, and it is why always-on rules live in
  the prompt, not in skill bodies (see
  [improving the agent](improving-the-agent.md)).
- The memory index `MEMORY.md` (see [memory](memory.md)). Individual notes
  enter only when read.
- The conversation so far: messages, tool calls, and tool results.

Tool results enter as real content, not summaries, but they are not kept
forever at full size: a large single result is offloaded and old ones are
cleared as the window fills (see [Growth and its limits](#growth-and-its-limits)
below). The Julia kernel's streamed output is rendered through a terminal
emulator (so progress bars collapse to their final state instead of thousands of
carriage-return frames) and tail-capped at 256 KB; result values and error
messages are capped at 64 KiB on the kernel side.

The TUI's collapsed tool and reasoning cards are display only. `Ctrl+O`
toggles the full output for you, and the model's context is unaffected
either way.

## Persistence

Conversation state lives in `checkpoints.sqlite` per session (langgraph's
checkpointer). That is what makes mid-session model switching work: the
agent is rebuilt with the new model on the same checkpoint thread, and the
conversation carries over. The trace (`trace.sqlite`) is a separate,
append-only record for humans and scorers. It is never fed back to the
model.

## Growth and its limits

A session's context grows with the conversation, so growth is managed in
layers, cheapest first, each running ahead of the next so the expensive ones
fire only when the cheaper ones are not enough:

- **Eviction of a large single result.** Any tool result over ~20k tokens is
  written to `large_tool_results/<id>` under the session state dir and replaced
  inline with a head/tail preview plus a pointer the agent can `read_file`. The
  full result stays recoverable; the context holds a stub. This is deepagents'
  `FilesystemMiddleware`.
- **Clearing of old tool results.** Once the conversation passes ~60% of the
  model's window, the older tool results (source reads, REPL output) are
  replaced by a `[cleared]` placeholder while the most recent ones stay
  verbatim, via langchain's `ContextEditingMiddleware` (wired in
  `agent/context_editing.py`). It is
  transparent (the raw log is untouched; only the model's per-call view is
  clipped) and cleared results are re-derivable: the files are still on disk and
  REPL commands can be re-run. The attempt log is never cleared, since the agent
  refers back to it by value.
- **Summarization.** When clearing is not enough and the conversation reaches
  ~85% of the window, the older turns are replaced by a structured summary
  (session intent, key decisions, artifacts, next steps) while the newest turns
  stay verbatim. The summarized turns are offloaded to
  `conversation_history/<internal-session-id>.md` first, so they remain
  recoverable, and the summary embeds that path. This is deepagents' stock
  backend-recoverable `SummarizationMiddleware`, installed by
  `create_deep_agent`, sized from the model profile (which
  `builder._set_profile_window` points at the real loaded window), and
  non-mutating; `TraceRecorder` records each compaction. We lean on the stock
  middleware so upstream improvements arrive without porting.

`/context` shows measured usage by category (the last call's `usage_metadata`
is exactly what the conversation costs to send) plus both the clearing and the
summarization thresholds, so it is clear what will happen as the window fills.
The status bar keeps a `ctx` percentage in view (yellow at 70%, red at 90%).

The window the thresholds are measured against is the model's real loaded size:
the provider package's profile data for cloud models, and for a local model the
`num_ctx` it was actually loaded with (its reported maximum capped at the memory
budget), not the daemon's theoretical maximum, which the model was never loaded
with. Sizing the trigger to the loaded window is what lets compaction fire
before a local model overflows.

- `/compact` runs a summarization pass on demand against the checkpointed
  thread. Every compaction (automatic or manual) is recorded as a
  `context_compaction` trace event, and the full conversation remains in the
  trace; compaction is non-mutating, so nothing is lost from the record.
- Keep sessions task-shaped regardless: clearing and the summary preserve the
  working set and the conclusions, not every byte. Durable knowledge belongs in
  memory, which survives the session boundary by design.
- Token usage per model turn is recorded in the trace (`model_usage` events), so
  the cost of a workflow is measurable, and the bench records it per sample.

Subagents are the structural answer for context-heavy sub-tasks: a subagent runs
in its own context window and returns a result, so the parent's context pays for
the conclusion, not the exploration. The seam exists per simulator
(`subagent_factories`); a general source-exploration subagent (so that browsing
installed package source never enters the main context at all) is the planned
next step.
