# Benchmark results

Snapshot generated 2026-06-24 from runs 2026-06-23T23-07-15 … 2026-06-24T01-40-50 (Makker 06251868c). Every sample runs the real agent end to end in a fresh workspace and is graded on the session trace as well as the answer. See [how evaluation works](evaluation.md). Each model ran the suite **3 times**, and cells aggregate across runs, so a fraction like 2/3 means the sample passed two of three runs.

## Overview

Pass rate is passing runs over runs that completed (infrastructure errors excluded). Tool calls and tokens are the per-run totals across the suite, the harness-efficiency signals: at equal pass rate, fewer means the harness got the agent there in less work. Input tokens note how many were served from the prompt cache (a cheap fraction of the input price); a model that caches aggressively processes a large input cheaply, which is why cost doesn't track raw token counts and is shown alongside them. Cost and wall time are for **one** pass over the suite (the per-run average), measured on a single machine. Within a model samples run one at a time, but wall time still depends on that machine and on how many models shared it during the run, so read it as indicative and comparable only within this snapshot; pass rate and cost are unaffected by either. Dollar costs use provider prices as of 2026-06-15 (see `eval/report.py`) and include prompt-cache reads/writes; the self-hosted model is priced against a hosted reference.

| Model | Pass rate | Tool calls / run | Input tokens / run | Output tokens / run | Cost / run | Wall / run |
|---|---|---|---|---|---|---|
| claude-haiku-4-5 | <span class="bench-partial">121/123</span> | 267 | 6.8M (6.3M cached, 93%) | 70k | $1.61 | 0.6 h |
| gemini-3.1-flash-lite | <span class="bench-partial">108/123</span> | 283 | 7.9M (6.2M cached, 79%) | 62k | $0.67 | 0.7 h |
| gpt-5.4-mini | <span class="bench-partial">111/123</span> | 307 | 3.8M (3.3M cached, 87%) | 25k | $0.72 | 0.4 h |
| qwen3.6:27b | <span class="bench-partial">114/123</span> | 266 | 4.4M | 58k | $1.46 | 0.9 h |

## By suite

| Suite | claude-haiku-4-5 | gemini-3.1-flash-lite | gpt-5.4-mini | qwen3.6:27b |
|---|---|---|---|---|
| api | <span class="bench-pass">6/6</span> | <span class="bench-pass">6/6</span> | <span class="bench-pass">6/6</span> | <span class="bench-partial">5/6</span> |
| battmo | <span class="bench-pass">6/6</span> | <span class="bench-pass">6/6</span> | <span class="bench-pass">6/6</span> | <span class="bench-pass">6/6</span> |
| calibration | <span class="bench-pass">3/3</span> | <span class="bench-pass">3/3</span> | <span class="bench-pass">3/3</span> | <span class="bench-pass">3/3</span> |
| canary | <span class="bench-pass">3/3</span> | <span class="bench-pass">3/3</span> | <span class="bench-pass">3/3</span> | <span class="bench-pass">3/3</span> |
| ensembles | <span class="bench-pass">12/12</span> | <span class="bench-partial">9/12</span> | <span class="bench-partial">10/12</span> | <span class="bench-partial">9/12</span> |
| filesystem | <span class="bench-pass">27/27</span> | <span class="bench-pass">27/27</span> | <span class="bench-partial">26/27</span> | <span class="bench-partial">26/27</span> |
| fimbul | <span class="bench-pass">3/3</span> | <span class="bench-pass">3/3</span> | <span class="bench-pass">3/3</span> | <span class="bench-partial">2/3</span> |
| guardrails | <span class="bench-pass">3/3</span> | <span class="bench-pass">3/3</span> | <span class="bench-partial">1/3</span> | <span class="bench-pass">3/3</span> |
| jutuldarcy | <span class="bench-partial">8/9</span> | <span class="bench-partial">4/9</span> | <span class="bench-pass">9/9</span> | <span class="bench-partial">6/9</span> |
| mocca | <span class="bench-pass">6/6</span> | <span class="bench-partial">5/6</span> | <span class="bench-partial">3/6</span> | <span class="bench-pass">6/6</span> |
| plotting | <span class="bench-pass">6/6</span> | <span class="bench-partial">4/6</span> | <span class="bench-pass">6/6</span> | <span class="bench-pass">6/6</span> |
| search | <span class="bench-partial">26/27</span> | <span class="bench-partial">23/27</span> | <span class="bench-partial">26/27</span> | <span class="bench-pass">27/27</span> |
| usage | <span class="bench-pass">12/12</span> | <span class="bench-pass">12/12</span> | <span class="bench-partial">9/12</span> | <span class="bench-pass">12/12</span> |
| **all** | <span class="bench-partial">121/123</span> | <span class="bench-partial">108/123</span> | <span class="bench-partial">111/123</span> | <span class="bench-partial">114/123</span> |

## By simulator

Cross-cut of the same samples by the simulator they exercise (`general` = sim-agnostic tasks like canary, calibration, plotting).

| Simulator | claude-haiku-4-5 | gemini-3.1-flash-lite | gpt-5.4-mini | qwen3.6:27b |
|---|---|---|---|---|
| battmo | <span class="bench-pass">12/12</span> | <span class="bench-partial">10/12</span> | <span class="bench-partial">11/12</span> | <span class="bench-partial">11/12</span> |
| fimbul | <span class="bench-pass">9/9</span> | <span class="bench-partial">8/9</span> | <span class="bench-pass">9/9</span> | <span class="bench-partial">6/9</span> |
| general | <span class="bench-partial">82/84</span> | <span class="bench-partial">74/84</span> | <span class="bench-partial">74/84</span> | <span class="bench-partial">79/84</span> |
| jutuldarcy | <span class="bench-pass">6/6</span> | <span class="bench-pass">6/6</span> | <span class="bench-pass">6/6</span> | <span class="bench-pass">6/6</span> |
| mocca | <span class="bench-pass">12/12</span> | <span class="bench-partial">10/12</span> | <span class="bench-partial">11/12</span> | <span class="bench-pass">12/12</span> |
| **all** | <span class="bench-partial">121/123</span> | <span class="bench-partial">108/123</span> | <span class="bench-partial">111/123</span> | <span class="bench-partial">114/123</span> |

<details markdown="1">
<summary>All samples (pass count, tool calls, tokens, cost, wall time)</summary>

| Suite | Sample | Sim | Model | Passed | Failures | Tool calls | Input | Output | Cost | Wall |
|---|---|---|---|---|---|---|---|---|---|---|
| api | `api1-newton-residual` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 13 | 135k | 1k | $0.03 | 0 min |
| api | `api1-newton-residual` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 7 | 93k | 555 | $0.01 | 0 min |
| api | `api1-newton-residual` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 11 | 83k | 708 | $0.02 | 0 min |
| api | `api1-newton-residual` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 13 | 107k | 2k | $0.04 | 1 min |
| api | `api2-internal-darcy` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 7 | 91k | 875 | $0.02 | 0 min |
| api | `api2-internal-darcy` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 6 | 73k | 343 | $0.01 | 0 min |
| api | `api2-internal-darcy` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 7 | 63k | 462 | $0.01 | 0 min |
| api | `api2-internal-darcy` | general | qwen3.6:27b | <span class="bench-partial">2/3</span> | wrong answer | 4 | 53k | 660 | $0.02 | 0 min |
| battmo | `bm1-chen-cc-discharge` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 8 | 129k | 1k | $0.03 | 1 min |
| battmo | `bm1-chen-cc-discharge` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 18 | 424k | 2k | $0.03 | 1 min |
| battmo | `bm1-chen-cc-discharge` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 10 | 101k | 736 | $0.02 | 1 min |
| battmo | `bm1-chen-cc-discharge` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 5 | 71k | 981 | $0.02 | 1 min |
| battmo | `bm3-crate-sweep` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 14 | 312k | 3k | $0.08 | 1 min |
| battmo | `bm3-crate-sweep` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 22 | 377k | 3k | $0.03 | 1 min |
| battmo | `bm3-crate-sweep` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 12 | 130k | 1k | $0.03 | 1 min |
| battmo | `bm3-crate-sweep` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 13 | 183k | 2k | $0.06 | 2 min |
| calibration | `cal1-exp-decay-fit` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 4 | 70k | 1k | $0.02 | 1 min |
| calibration | `cal1-exp-decay-fit` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 4 | 59k | 520 | $0.01 | 1 min |
| calibration | `cal1-exp-decay-fit` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 7 | 67k | 543 | $0.01 | 0 min |
| calibration | `cal1-exp-decay-fit` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 7 | 95k | 2k | $0.03 | 1 min |
| canary | `x0-sum-from-file` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 3 | 54k | 300 | $0.02 | 0 min |
| canary | `x0-sum-from-file` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 4 | 53k | 139 | $0.01 | 0 min |
| canary | `x0-sum-from-file` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 4 | 41k | 213 | $0.01 | 0 min |
| canary | `x0-sum-from-file` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 2 | 37k | 299 | $0.01 | 0 min |
| ensembles | `ens-bm-crate-sweep` | battmo | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 14 | 290k | 3k | $0.07 | 2 min |
| ensembles | `ens-bm-crate-sweep` | battmo | gemini-3.1-flash-lite | <span class="bench-partial">2/3</span> | serial / mechanism | 23 | 633k | 4k | $0.05 | 1 min |
| ensembles | `ens-bm-crate-sweep` | battmo | gpt-5.4-mini | <span class="bench-partial">2/3</span> | wrong answer | 18 | 221k | 2k | $0.04 | 1 min |
| ensembles | `ens-bm-crate-sweep` | battmo | qwen3.6:27b | <span class="bench-partial">2/3</span> | wrong answer | 21 | 417k | 4k | $0.13 | 4 min |
| ensembles | `ens-fb-injtemp-sweep` | fimbul | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 21 | 491k | 4k | $0.10 | 9 min |
| ensembles | `ens-fb-injtemp-sweep` | fimbul | gemini-3.1-flash-lite | <span class="bench-partial">2/3</span> | hit budget | 9 | 668k | 3k | $0.06 | 6 min |
| ensembles | `ens-fb-injtemp-sweep` | fimbul | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 17 | 269k | 1k | $0.04 | 4 min |
| ensembles | `ens-fb-injtemp-sweep` | fimbul | qwen3.6:27b | <span class="bench-partial">1/3</span> | serial / mechanism, wrong answer | 21 | 486k | 6k | $0.16 | 9 min |
| ensembles | `ens-jd-porosity-sweep` | jutuldarcy | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 5 | 89k | 997 | $0.03 | 1 min |
| ensembles | `ens-jd-porosity-sweep` | jutuldarcy | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 7 | 101k | 569 | $0.01 | 1 min |
| ensembles | `ens-jd-porosity-sweep` | jutuldarcy | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 10 | 123k | 752 | $0.02 | 1 min |
| ensembles | `ens-jd-porosity-sweep` | jutuldarcy | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 10 | 180k | 3k | $0.06 | 3 min |
| ensembles | `ens-mc-cycles-sweep` | mocca | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 17 | 390k | 5k | $0.09 | 4 min |
| ensembles | `ens-mc-cycles-sweep` | mocca | gemini-3.1-flash-lite | <span class="bench-partial">2/3</span> | serial / mechanism | 14 | 280k | 4k | $0.03 | 2 min |
| ensembles | `ens-mc-cycles-sweep` | mocca | gpt-5.4-mini | <span class="bench-partial">2/3</span> | wrong answer | 19 | 288k | 2k | $0.05 | 2 min |
| ensembles | `ens-mc-cycles-sweep` | mocca | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 22 | 533k | 6k | $0.17 | 6 min |
| filesystem | `fs1-write-and-include` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 2 | 33k | 202 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 2 | 31k | 92 | $0.00 | 0 min |
| filesystem | `fs1-write-and-include` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 3 | 31k | 128 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 2 | 33k | 199 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include-battmo` | battmo | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 2 | 28k | 188 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include-battmo` | battmo | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 2 | 31k | 101 | $0.00 | 0 min |
| filesystem | `fs1-write-and-include-battmo` | battmo | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 2 | 28k | 82 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include-battmo` | battmo | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 2 | 33k | 236 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include-fimbul` | fimbul | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 2 | 24k | 182 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include-fimbul` | fimbul | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 2 | 35k | 105 | $0.00 | 0 min |
| filesystem | `fs1-write-and-include-fimbul` | fimbul | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 3 | 31k | 128 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include-fimbul` | fimbul | qwen3.6:27b | <span class="bench-partial">2/3</span> | wrong answer | 2 | 29k | 165 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include-mocca` | mocca | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 2 | 29k | 193 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include-mocca` | mocca | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 2 | 31k | 95 | $0.00 | 0 min |
| filesystem | `fs1-write-and-include-mocca` | mocca | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 4 | 38k | 190 | $0.01 | 0 min |
| filesystem | `fs1-write-and-include-mocca` | mocca | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 2 | 33k | 234 | $0.01 | 0 min |
| filesystem | `fs2-nested-write-and-include` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 2 | 33k | 240 | $0.01 | 0 min |
| filesystem | `fs2-nested-write-and-include` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 3 | 42k | 121 | $0.00 | 0 min |
| filesystem | `fs2-nested-write-and-include` | general | gpt-5.4-mini | <span class="bench-partial">2/3</span> | wrong answer | 6 | 67k | 337 | $0.01 | 0 min |
| filesystem | `fs2-nested-write-and-include` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 2 | 33k | 242 | $0.01 | 0 min |
| filesystem | `fs3-edit-and-rerun` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 3 | 50k | 350 | $0.01 | 0 min |
| filesystem | `fs3-edit-and-rerun` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 4 | 53k | 190 | $0.01 | 0 min |
| filesystem | `fs3-edit-and-rerun` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 6 | 68k | 371 | $0.01 | 0 min |
| filesystem | `fs3-edit-and-rerun` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 3 | 44k | 284 | $0.01 | 0 min |
| filesystem | `fs4-save-output-file` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 2 | 41k | 244 | $0.01 | 0 min |
| filesystem | `fs4-save-output-file` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 2 | 35k | 118 | $0.00 | 0 min |
| filesystem | `fs4-save-output-file` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 6 | 58k | 349 | $0.01 | 0 min |
| filesystem | `fs4-save-output-file` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 3 | 33k | 319 | $0.01 | 0 min |
| filesystem | `fs5-multi-file-project` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 4 | 42k | 431 | $0.01 | 0 min |
| filesystem | `fs5-multi-file-project` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 5 | 65k | 284 | $0.01 | 0 min |
| filesystem | `fs5-multi-file-project` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 8 | 78k | 470 | $0.01 | 0 min |
| filesystem | `fs5-multi-file-project` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 4 | 41k | 370 | $0.01 | 0 min |
| filesystem | `fs6-read-transform-write` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 2 | 37k | 243 | $0.01 | 0 min |
| filesystem | `fs6-read-transform-write` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 6 | 77k | 498 | $0.01 | 0 min |
| filesystem | `fs6-read-transform-write` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 3 | 28k | 135 | $0.01 | 0 min |
| filesystem | `fs6-read-transform-write` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 2 | 37k | 706 | $0.01 | 0 min |
| fimbul | `fb1-doublet-cooldown` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 12 | 367k | 3k | $0.08 | 3 min |
| fimbul | `fb1-doublet-cooldown` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 9 | 164k | 1k | $0.02 | 3 min |
| fimbul | `fb1-doublet-cooldown` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 11 | 130k | 788 | $0.03 | 3 min |
| fimbul | `fb1-doublet-cooldown` | general | qwen3.6:27b | <span class="bench-partial">2/3</span> | wrong answer | 8 | 115k | 2k | $0.04 | 3 min |
| guardrails | `x1-no-shell-julia` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 1 | 24k | 84 | $0.01 | 0 min |
| guardrails | `x1-no-shell-julia` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 1 | 21k | 35 | $0.00 | 0 min |
| guardrails | `x1-no-shell-julia` | general | gpt-5.4-mini | <span class="bench-partial">1/3</span> | wrong answer | 2 | 25k | 100 | $0.01 | 0 min |
| guardrails | `x1-no-shell-julia` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 1 | 26k | 240 | $0.01 | 0 min |
| jutuldarcy | `jd-millidarcy-conversion` | general | claude-haiku-4-5 | <span class="bench-partial">2/3</span> | hit budget | 17 | 1.2M | 9k | $0.22 | 2 min |
| jutuldarcy | `jd-millidarcy-conversion` | general | gemini-3.1-flash-lite | <span class="bench-fail">0/3</span> | hit budget, wrong answer | 22 | 1.3M | 14k | $0.09 | 2 min |
| jutuldarcy | `jd-millidarcy-conversion` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 14 | 199k | 2k | $0.04 | 1 min |
| jutuldarcy | `jd-millidarcy-conversion` | general | qwen3.6:27b | <span class="bench-partial">2/3</span> | wrong answer | 12 | 241k | 5k | $0.08 | 3 min |
| jutuldarcy | `jd1-gravity-segregation` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 16 | 299k | 5k | $0.08 | 1 min |
| jutuldarcy | `jd1-gravity-segregation` | general | gemini-3.1-flash-lite | <span class="bench-partial">2/3</span> | hit budget | 13 | 278k | 3k | $0.02 | 14 min |
| jutuldarcy | `jd1-gravity-segregation` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 11 | 157k | 2k | $0.03 | 1 min |
| jutuldarcy | `jd1-gravity-segregation` | general | qwen3.6:27b | <span class="bench-partial">2/3</span> | wrong answer | 13 | 239k | 5k | $0.08 | 3 min |
| jutuldarcy | `jd3-halved-injection` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 22 | 449k | 7k | $0.11 | 2 min |
| jutuldarcy | `jd3-halved-injection` | general | gemini-3.1-flash-lite | <span class="bench-partial">2/3</span> | hit budget | 15 | 799k | 13k | $0.06 | 2 min |
| jutuldarcy | `jd3-halved-injection` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 16 | 278k | 2k | $0.04 | 1 min |
| jutuldarcy | `jd3-halved-injection` | general | qwen3.6:27b | <span class="bench-partial">2/3</span> | wrong answer | 10 | 175k | 3k | $0.06 | 2 min |
| mocca | `mc1-vsa-cyclic-golden` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 11 | 199k | 3k | $0.05 | 2 min |
| mocca | `mc1-vsa-cyclic-golden` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 10 | 243k | 2k | $0.02 | 1 min |
| mocca | `mc1-vsa-cyclic-golden` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 12 | 143k | 1k | $0.03 | 1 min |
| mocca | `mc1-vsa-cyclic-golden` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 8 | 116k | 2k | $0.04 | 2 min |
| mocca | `mc4-tsa-toth-honesty` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 0 | 1.0M | 11k | $0.24 | 4 min |
| mocca | `mc4-tsa-toth-honesty` | general | gemini-3.1-flash-lite | <span class="bench-partial">2/3</span> | wrong answer | 13 | 992k | 6k | $0.07 | 5 min |
| mocca | `mc4-tsa-toth-honesty` | general | gpt-5.4-mini | <span class="bench-fail">0/3</span> | wrong answer | 25 | 477k | 2k | $0.06 | 2 min |
| mocca | `mc4-tsa-toth-honesty` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 23 | 287k | 3k | $0.09 | 2 min |
| plotting | `x5-headless-plot` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 2 | 34k | 350 | $0.01 | 0 min |
| plotting | `x5-headless-plot` | general | gemini-3.1-flash-lite | <span class="bench-partial">1/3</span> | wrong answer | 4 | 108k | 301 | $0.01 | 1 min |
| plotting | `x5-headless-plot` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 3 | 35k | 251 | $0.01 | 0 min |
| plotting | `x5-headless-plot` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 5 | 70k | 836 | $0.02 | 1 min |
| plotting | `x6-read-the-bar` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 2 | 43k | 499 | $0.01 | 1 min |
| plotting | `x6-read-the-bar` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 3 | 44k | 197 | $0.01 | 1 min |
| plotting | `x6-read-the-bar` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 8 | 75k | 788 | $0.02 | 1 min |
| plotting | `x6-read-the-bar` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 6 | 82k | 1k | $0.03 | 1 min |
| search | `se1-locate-definition` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 3 | 50k | 339 | $0.01 | 0 min |
| search | `se1-locate-definition` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 4 | 54k | 183 | $0.01 | 0 min |
| search | `se1-locate-definition` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 2 | 19k | 106 | $0.01 | 0 min |
| search | `se1-locate-definition` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 1 | 22k | 164 | $0.01 | 0 min |
| search | `se1-locate-definition-battmo` | battmo | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 3 | 50k | 316 | $0.01 | 0 min |
| search | `se1-locate-definition-battmo` | battmo | gemini-3.1-flash-lite | <span class="bench-partial">2/3</span> | wrong answer | 2 | 35k | 106 | $0.00 | 0 min |
| search | `se1-locate-definition-battmo` | battmo | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 2 | 19k | 107 | $0.01 | 0 min |
| search | `se1-locate-definition-battmo` | battmo | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 2 | 37k | 273 | $0.01 | 0 min |
| search | `se1-locate-definition-fimbul` | fimbul | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 3 | 46k | 280 | $0.01 | 0 min |
| search | `se1-locate-definition-fimbul` | fimbul | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 4 | 49k | 157 | $0.01 | 0 min |
| search | `se1-locate-definition-fimbul` | fimbul | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 2 | 19k | 107 | $0.01 | 0 min |
| search | `se1-locate-definition-fimbul` | fimbul | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 1 | 26k | 169 | $0.01 | 0 min |
| search | `se1-locate-definition-mocca` | mocca | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 4 | 59k | 379 | $0.01 | 0 min |
| search | `se1-locate-definition-mocca` | mocca | gemini-3.1-flash-lite | <span class="bench-partial">2/3</span> | wrong answer | 3 | 43k | 130 | $0.00 | 0 min |
| search | `se1-locate-definition-mocca` | mocca | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 3 | 32k | 175 | $0.01 | 0 min |
| search | `se1-locate-definition-mocca` | mocca | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 1 | 22k | 163 | $0.01 | 0 min |
| search | `se2-locate-example` | general | claude-haiku-4-5 | <span class="bench-partial">2/3</span> | wrong answer | 3 | 41k | 252 | $0.01 | 0 min |
| search | `se2-locate-example` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 2 | 35k | 123 | $0.00 | 0 min |
| search | `se2-locate-example` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 3 | 25k | 134 | $0.01 | 0 min |
| search | `se2-locate-example` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 1 | 22k | 128 | $0.01 | 0 min |
| search | `se3-find-call-sites` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 4 | 42k | 544 | $0.01 | 0 min |
| search | `se3-find-call-sites` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 5 | 66k | 345 | $0.01 | 0 min |
| search | `se3-find-call-sites` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 3 | 29k | 232 | $0.01 | 0 min |
| search | `se3-find-call-sites` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 2 | 22k | 346 | $0.01 | 0 min |
| search | `se4-count-jl-files` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 4 | 68k | 401 | $0.02 | 0 min |
| search | `se4-count-jl-files` | general | gemini-3.1-flash-lite | <span class="bench-partial">1/3</span> | wrong answer | 1 | 24k | 48 | $0.00 | 0 min |
| search | `se4-count-jl-files` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 1 | 19k | 73 | $0.01 | 0 min |
| search | `se4-count-jl-files` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 1 | 22k | 110 | $0.01 | 0 min |
| search | `se5-find-constant` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 4 | 59k | 451 | $0.01 | 0 min |
| search | `se5-find-constant` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 5 | 61k | 246 | $0.01 | 0 min |
| search | `se5-find-constant` | general | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 3 | 29k | 165 | $0.01 | 0 min |
| search | `se5-find-constant` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 2 | 26k | 230 | $0.01 | 0 min |
| search | `se6-call-chain` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 10 | 144k | 1k | $0.03 | 0 min |
| search | `se6-call-chain` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 7 | 131k | 372 | $0.01 | 0 min |
| search | `se6-call-chain` | general | gpt-5.4-mini | <span class="bench-partial">2/3</span> | wrong answer | 11 | 62k | 588 | $0.01 | 0 min |
| search | `se6-call-chain` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 6 | 65k | 611 | $0.02 | 0 min |
| usage | `use-bm-cell-capacity` | battmo | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 4 | 68k | 761 | $0.02 | 0 min |
| usage | `use-bm-cell-capacity` | battmo | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 7 | 93k | 603 | $0.01 | 0 min |
| usage | `use-bm-cell-capacity` | battmo | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 10 | 110k | 746 | $0.02 | 0 min |
| usage | `use-bm-cell-capacity` | battmo | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 12 | 170k | 3k | $0.06 | 2 min |
| usage | `use-csv-mean` | general | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 4 | 60k | 561 | $0.02 | 0 min |
| usage | `use-csv-mean` | general | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 4 | 55k | 255 | $0.01 | 0 min |
| usage | `use-csv-mean` | general | gpt-5.4-mini | <span class="bench-fail">0/3</span> | wrong answer | 5 | 45k | 214 | $0.01 | 0 min |
| usage | `use-csv-mean` | general | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 6 | 79k | 685 | $0.03 | 1 min |
| usage | `use-jd-well-api` | jutuldarcy | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 8 | 126k | 1k | $0.03 | 0 min |
| usage | `use-jd-well-api` | jutuldarcy | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 5 | 75k | 377 | $0.01 | 0 min |
| usage | `use-jd-well-api` | jutuldarcy | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 4 | 29k | 335 | $0.01 | 0 min |
| usage | `use-jd-well-api` | jutuldarcy | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 3 | 40k | 748 | $0.01 | 1 min |
| usage | `use-mc-list-examples` | mocca | claude-haiku-4-5 | <span class="bench-pass">3/3</span> | — | 2 | 37k | 270 | $0.01 | 0 min |
| usage | `use-mc-list-examples` | mocca | gemini-3.1-flash-lite | <span class="bench-pass">3/3</span> | — | 2 | 35k | 171 | $0.00 | 0 min |
| usage | `use-mc-list-examples` | mocca | gpt-5.4-mini | <span class="bench-pass">3/3</span> | — | 2 | 19k | 167 | $0.01 | 0 min |
| usage | `use-mc-list-examples` | mocca | qwen3.6:27b | <span class="bench-pass">3/3</span> | — | 1 | 22k | 254 | $0.01 | 0 min |

</details>

## Reading the results

A sample passes only when every scorer passes: the answer checks *and* the trace checks (the required mechanism appears in code the agent actually ran). Failures fall into:

- **wrong answer**: the reported values failed the golden or structural check.
- **serial / mechanism**: the answer may be right, but a required mechanism is missing from the trace (e.g. a sweep that ran serially when the prompt asked for a parallel ensemble).
- **hit budget**: the sample reached its message or time cap before finishing.
- **infra error**: the run failed before the agent could work (provider or harness error); excluded from pass rates, not a model result.

Composite tasks are noisy at a single epoch, so each model runs the suite a few times and the cells aggregate the runs. Regenerate this page with:

```sh
uv run makker eval <suite> --model <provider/model> --epochs 3
uv run python -m makker.eval.report <log-prefix> -o docs/benchmark.md
```

To add a model without re-running the others, merge the committed snapshot instead: pass `--records docs/benchmark-records.jsonl` and write it back with `--json docs/benchmark-records.jsonl`.
