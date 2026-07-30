# The trace database

Every session appends its events to `trace.sqlite` in the session state
directory. The trace is the system's source of truth for what happened:
transcripts are renderings of it, bench scorers grade against it, and any
future learn-from-usage loop mines it. The model never sees it.

## Schema

One table, append-only, WAL mode (`trace/log.py`):

```sql
CREATE TABLE events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    TEXT NOT NULL,      -- UTC ISO-8601
    kind         TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
```

Payloads are JSON, so new event kinds and new fields are additive: no
migrations, and old traces stay readable. The `TraceLog` class is the
writer and reader API, though plain `sqlite3` works too. Kind names live
in `trace/schema.py`; writers and readers import the constants rather
than restating the strings, and the table below is the payload contract.

## Event kinds

| Kind | Written by | Payload |
|---|---|---|
| `session_start` | `Session.create` | `session_id`, `simulator` |
| `session_resume` | `Session.resume` | `session_id`, `simulator` |
| `session_title` | `Session.adopt_title` | `session_id`, `title` (from the first prompt) |
| `session_end` | `Session.finalize` | none |
| `message_user` | `TurnRunner` | `content` |
| `message_reasoning` | recorder middleware | `content` (the model's reasoning text) |
| `message_assistant` | recorder middleware | `content` |
| `model_usage` | recorder middleware | `input_tokens`, `output_tokens`, `total_tokens`, provider detail fields |
| `context_compaction` | summarization middleware / `/compact` | `summarized`, `kept` (message counts, null when unknown), `offloaded`, `manual` |
| `tool_call` | recorder middleware | `id`, `name`, `args` |
| `tool_result` | recorder middleware | `tool_call_id`, `name`, `content`, `status` |
| `hitl_request` | `TurnRunner` | the pending tool call awaiting approval |
| `hitl_response` | `TurnRunner` | `interrupt_id`, the decision payload |
| `artifact` | `plot_julia` / `recapture_plot` / `write_report` | `path` (relative to the session output dir), `mime`, `caption`, `tool_call_id`, `format`, `kind` (`plot`/`report`, routes the browser viz), `size_px`, `dpi`, `slot`, `source_code`, `poster` (thumbnail for an interactive plot), `live_url` |
| `attempt` | `record_attempt` | `id`, `parent_id`, `rationale`, `parameters_changed`, `metrics`, `candidate_path`, `plot_artifact_path`, `notes` |
| `upload` | web server | `path` of a file the user uploaded into the workspace |
| `ui` | a capability's user-interface tool | `action` and `payload`, forwarded to the front end as a `ui` message |
| `ui_event` | web server | `payload` of an interaction the front end reported (recorded for later analysis) |
| `eval_target` | bench solver | `expected` golden answer for an eval session |
| `eval_result` | `jutul-agent review eval-link` | `passed`, `task`, `scores` (written onto the session after scoring) |

The recorder is an agent middleware (`trace/recorder.py`), so it observes
the same stream the model produces: every model turn and every tool
round-trip, including tool errors (a raised tool exception is recorded as
an error result, not lost).

Two kinds carry the domain structure that makes the trace more than a
chat log. `artifact` ties every figure to the exact code that produced it.
`attempt` records one step of a parameter investigation, with a
`parent_id` so calibration runs form a tree. The investigation report and
the bench's process scorers both read that structure.

## Consumers

- `jutul-agent transcript` renders the trace as HTML or markdown, with
  `--bundle` zipping the referenced artifacts alongside.
- Bench scorers read tool calls, arguments, artifacts, and attempts to
  verify the agent did the work it claims
  ([evaluation](evaluation.md)).
- `model_usage` events make token cost per turn and per workflow
  measurable in real sessions, not just bench runs.

## Reading one

```python
from jutul_agent.trace import TraceLog

with TraceLog.open_readonly(path_to_trace) as log:
    for event in log.iter_events():
        print(event.timestamp, event.kind, event.payload)
```

`open_readonly` inspects a trace without touching it (no schema creation,
no journal-mode change), which is the right way to read another session's
file, possibly one still being written.

Traces live under the state home, at
`workspaces/<hash>/sessions/<id>/trace.sqlite` (see
[configuration](configuration.md)), and are plain files: copy one next to a
bug report and the whole session comes with it.
