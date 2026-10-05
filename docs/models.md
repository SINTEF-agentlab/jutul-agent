# Models

jutul-agent treats models as opaque `provider:model` strings resolved by
LangChain's `init_chat_model`. Adding a new model is configuration, not a
code change.

## Selection

Precedence, highest first: the `--model` flag, the workspace config, the
user config, `$JUTUL_AGENT_MODEL`, the built-in default.

In the web UI, `/model` opens the model picker with OpenAI, Anthropic, Google,
OpenRouter, and Ollama entries. You can also type `/model provider:model` to
use another model ID. Switching mid-session keeps the conversation (see
[context handling](context.md)).

Providers beyond the bundled five work once their LangChain package is
installed, and jutul-agent names the exact package when it is missing.
API keys are prompted for on first use and stored in the user-global `.env`
([configuration](configuration.md)).

## OpenRouter

OpenRouter is included. Save an API key and start the web UI:

```sh
jutul-agent key openrouter
jutul-agent web
```

Then use `/model` to choose an OpenRouter model.

## What the harness asks of a model

The agent is tool-driven: every Julia evaluation, file edit, and plot is a
tool call, often ten or more per task. Models therefore need solid tool
calling, and weaker models fail in characteristic ways: skipping tools and
answering from memory, or calling a tool with malformed arguments. The
bench's trajectory scorers exist to catch exactly this, and the canary
suite is the quick way to qualify a new model
([evaluation](evaluation.md)).

## Reasoning

Models that can reason get it requested explicitly, because the provider
defaults hide it or skip it: recent OpenAI reasoning models default to
effort *none* (they do not reason at all), and Anthropic's extended
thinking is off without a budget. The builder asks the model's bundled
profile (`BaseChatModel.profile`) whether it reasons and then enables, per
provider:

- OpenAI: `reasoning={"effort": "medium", "summary": "auto"}`. The model
  reasons when a task warrants it and the summary streams into the TUI's
  reasoning card while it thinks. The `-chat` hybrids, which reject the
  effort parameter, are excluded by the same profile data (they keep
  temperature support; true reasoning models do not).
- Anthropic: extended thinking with a 10k-token budget (output capped at
  24k so thinking leaves room for the answer). The thinking text streams
  into the same card.
- Gemini: `include_thoughts=True`. These models think by default (level
  `high` on Gemini 3+), so the only thing to add is visibility. A Gemini
  model with no profile entry is treated as thinking, since every model
  newer than the bundled data thinks; the legacy non-thinking ones are
  marked explicitly in the data.
- OpenRouter: models whose bundled profile supports reasoning receive
  `reasoning={"effort": "medium", "summary": "auto"}` through the dedicated
  integration, which preserves reasoning across tool calls.
- Ollama: think mode is requested explicitly for models the daemon
  reports as thinking-capable. Left at the daemon default, the thinking
  segment is dropped on the client side, and a turn the model spends
  entirely on thinking then surfaces as an empty reply with no tool
  calls, and the agent falls silent. Requested, the thinking is
  separated and visible.

Because the capability check is profile data shipped with the provider
packages, new models are covered by upgrading the package, and there is no
model list to maintain here. A model whose profile is missing (or a
session without the provider key at build time) falls back to the plain
spec string and simply runs without visible reasoning. Reasoning cards
appear only when the model actually reasons: trivial prompts legitimately
produce none.

The same profile data supplies the context window for `/context` and the
auto-compaction trigger; Gemini models missing from the data ask the
Gemini API for their limit instead.

## Local models (Ollama)

Local models run through Ollama with no key. Two things the harness handles
for you:

- Capability gating: the selector checks that a model supports tool
  calling before offering it, and can pull a model that is not installed
  yet. An outdated Ollama can mis-parse a new model's template and lose
  tool support, so keep Ollama itself current.
- Context sizing: the agent's system prompt is large, so local models are
  loaded with a context window sized to what the model reports, capped by
  a memory budget (64K tokens by default, lowered with
  `JUTUL_AGENT_OLLAMA_NUM_CTX` on tight hardware).
- Malformed tool calls: a weaker model occasionally serializes a tool
  call with unparseable arguments (e.g. two parallel calls merged into
  one). Such a reply would otherwise end the turn silently; instead the
  parse error is fed back and the model retries (see
  [turns](turns.md)). Parallel calls themselves are normal and kept.

Expect local models to be noticeably weaker at multi-step tool use than the
hosted ones. They are best for cheap iteration and offline work, with the
bench as the honest comparison.

## Models in the bench

`jutul-agent eval` uses Inspect's model layer, so model ids there take the
`provider/model` form, preserving any slashes or tags in the model ID, and
one run can matrix several models. The agent under test runs against the
bridge regardless of the target provider, which is how the same session
code serves every model. See [evaluation](evaluation.md).
