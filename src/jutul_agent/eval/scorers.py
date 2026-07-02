"""Scorers that grade the session trace, not just the final answer.

A jutul-agent sample can "pass" textually while having fabricated the work,
so every task pairs an answer check with at least one trace check: the tools
that should have run must appear as ``tool_call`` events in the session's
``trace.sqlite`` (recorded by the solver in the sample store).
"""

from __future__ import annotations

import re
from pathlib import Path

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from jutul_agent.eval.solver import STORE_OUTPUT_DIR, STORE_TRACE_DB, STORE_WORKSPACE
from jutul_agent.paths import is_host_path
from jutul_agent.trace import Event, schema

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
# "338,350" must parse as one number, not two: strip commas used as
# digit-group separators (digit,3-digits,non-digit) before matching.
_GROUPED_COMMA = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")

# A matched snippet is truncated before it goes into a Score's human-readable
# explanation, so one long match or command cannot bloat the log; the
# explanation only needs enough to identify what matched.
_EXPLAIN_MATCH = 80  # a regex match group (usually a short fragment)
_EXPLAIN_COMMAND = 160  # a fuller shell command or call signature


def _answer_numbers(text: str) -> list[float]:
    """Every number in ``text``, tolerant of digit grouping and exponents."""
    return [float(m) for m in _NUMBER.findall(_GROUPED_COMMA.sub("", text))]


def _trace_events(state: TaskState) -> list[Event]:
    """Every event in this sample's session trace, or ``[]`` when there is none.

    One read-only open per scorer call: scorers grade the same events the
    transcript renders, and never mutate the session they are grading.
    """
    from jutul_agent.trace import TraceLog

    path = state.store.get(STORE_TRACE_DB)
    if not path or not Path(path).exists():
        return []
    with TraceLog.open_readonly(Path(path)) as log:
        return log.iter_events()


def _trace_tool_calls(state: TaskState) -> list[str]:
    """Names of every tool call recorded in this sample's session trace."""
    return [
        event.payload.get("name", "")
        for event in _trace_events(state)
        if event.kind == schema.TOOL_CALL
    ]


@scorer(metrics=[accuracy()])
def used_tools(required: list[str]) -> Scorer:
    """Pass when every required tool appears in the session trace."""

    async def score(state: TaskState, target: Target) -> Score:
        calls = _trace_tool_calls(state)
        missing = [name for name in required if name not in calls]
        return Score(
            value=CORRECT if not missing else INCORRECT,
            explanation=(
                f"tool calls in trace: {calls or 'none'}"
                + (f"; missing required: {missing}" if missing else "")
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def avoided_tools(forbidden: list[str]) -> Scorer:
    """Pass when none of the forbidden tools appear in the session trace."""

    async def score(state: TaskState, target: Target) -> Score:
        calls = _trace_tool_calls(state)
        hits = [name for name in forbidden if name in calls]
        return Score(
            value=CORRECT if not hits else INCORRECT,
            explanation=(f"forbidden tools used: {hits}" if hits else "no forbidden tool used"),
        )

    return score


@scorer(metrics=[accuracy()])
def no_interpreters_via_execute(
    interpreters: tuple[str, ...] = ("julia",),
) -> Scorer:
    """Fail when the agent spawned ``julia`` (by default) through the shell.

    A shell julia is a cold process sharing nothing with the session kernel,
    so it is never the right move; shell python is general competence and is
    not flagged unless a task opts in by widening ``interpreters``. The check
    shares the workspace backend's head-token detection, so an interpreter at
    a command's executable position counts while the same word as data does
    not (an ``ls ~/.julia/...`` is innocent). Blocked attempts still count:
    the rule measures the model's behavior, the backend guard merely contains
    it.
    """
    from jutul_agent.agent.backend import interpreter_invocation

    async def score(state: TaskState, target: Target) -> Score:
        offenders: list[str] = []
        for event in _trace_events(state):
            if event.kind != schema.TOOL_CALL or event.payload.get("name") != "execute":
                continue
            command = str((event.payload.get("args") or {}).get("command", ""))
            name = interpreter_invocation(command, interpreters)
            if name is not None:
                offenders.append(command[:_EXPLAIN_COMMAND])
        return Score(
            value=CORRECT if not offenders else INCORRECT,
            explanation=(
                f"execute ran an interpreter: {offenders}"
                if offenders
                else "no interpreter via the shell"
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def julia_code_matches(pattern: str) -> Scorer:
    """Pass when code the agent ran in the Julia session matches ``pattern``.

    The trajectory check for tasks whose point is a specific in-session
    mechanism (e.g. ``run_ensemble`` for a parallel sweep): the mechanism
    must appear in the code of a ``run_julia``/``plot_julia`` call, so a
    textual claim of having used it cannot pass.
    """
    compiled = re.compile(pattern)

    async def score(state: TaskState, target: Target) -> Score:
        hits: list[str] = []
        for event in _trace_events(state):
            if event.kind != schema.TOOL_CALL:
                continue
            if event.payload.get("name") not in ("run_julia", "plot_julia"):
                continue
            code = str((event.payload.get("args") or {}).get("code", ""))
            match = compiled.search(code)
            if match is not None:
                hits.append(match.group(0)[:_EXPLAIN_MATCH])
        return Score(
            value=CORRECT if hits else INCORRECT,
            explanation=(
                f"julia code matched {pattern!r}: {hits}"
                if hits
                else f"no run_julia/plot_julia code matched {pattern!r}"
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def tool_call_matches(tool: str, args_pattern: str) -> Scorer:
    """Pass when a call to ``tool`` has arguments matching ``args_pattern``.

    The pattern is searched in the JSON-serialized arguments, so it can pin
    a specific parameter (e.g. ``"view": true`` on ``plot_julia``, meaning the
    agent must have actually looked at its own plot).
    """
    import json as _json

    compiled = re.compile(args_pattern)

    async def score(state: TaskState, target: Target) -> Score:
        hits: list[str] = []
        for event in _trace_events(state):
            if event.kind != schema.TOOL_CALL or event.payload.get("name") != tool:
                continue
            serialized = _json.dumps(event.payload.get("args") or {})
            match = compiled.search(serialized)
            if match is not None:
                hits.append(match.group(0)[:_EXPLAIN_MATCH])
        return Score(
            value=CORRECT if hits else INCORRECT,
            explanation=(
                f"{tool} args matched {args_pattern!r}: {hits}"
                if hits
                else f"no {tool} call with args matching {args_pattern!r}"
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def artifact_produced(suffix: str = ".png") -> Scorer:
    """Pass when the trace records a non-empty artifact with the suffix.

    ``plot_julia`` records every figure it saves as an ``artifact`` event, so
    "a plot exists" is checked against the trace plus the file on disk; a
    textual claim of having plotted cannot pass.
    """

    async def score(state: TaskState, target: Target) -> Score:
        # Artifact paths are recorded relative to the session output dir
        # (plot_julia writes "artifacts/<name>.png"); the workspace is kept
        # as a fallback root for artifacts recorded by other tools.
        roots = [
            Path(root)
            for root in (
                state.store.get(STORE_OUTPUT_DIR),
                state.store.get(STORE_WORKSPACE),
            )
            if root
        ]
        found: list[str] = []
        for event in _trace_events(state):
            if event.kind != schema.ARTIFACT:
                continue
            artifact = str(event.payload.get("path", ""))
            if not artifact.endswith(suffix):
                continue
            for root in roots:
                file = root / artifact.lstrip("/")
                if file.exists() and file.stat().st_size > 0:
                    found.append(artifact)
                    break
        return Score(
            value=CORRECT if found else INCORRECT,
            explanation=(
                f"non-empty {suffix} artifacts: {found}"
                if found
                else f"no non-empty {suffix} artifact in trace"
            ),
        )

    return score


def _longest_monotone_run(values: list[float], sign: int) -> int:
    """Length of the longest strictly monotone subsequence (order kept).

    ``sign`` is +1 for decreasing, -1 for increasing.
    """
    best = [1] * len(values)
    for i, v in enumerate(values):
        for j in range(i):
            if sign * (values[j] - v) > 0:
                best[i] = max(best[i], best[j] + 1)
    return max(best, default=0)


@scorer(metrics=[accuracy()])
def numeric_answer(
    low: float,
    high: float,
    *,
    count: int = 1,
    order: str = "any",
) -> Scorer:
    """Pass when the answer's numbers have a physically plausible shape.

    The structural check for tasks without a captured golden yet: at least
    ``count`` numbers in the final answer fall inside ``[low, high]``, and,
    when the task's physics orders its results, form a strictly
    ``"increasing"`` or ``"decreasing"`` sequence. The order check accepts
    any subsequence of that length, so labels and units interleaved with the
    values cannot fail a correct answer. Catches refusals and fabricated
    shapes, not wrong values; replace with a golden once one is captured.
    """
    if order not in ("any", "increasing", "decreasing"):
        raise ValueError(f"order must be 'any', 'increasing' or 'decreasing': {order!r}")

    async def score(state: TaskState, target: Target) -> Score:
        values = _answer_numbers(state.output.completion)
        in_range = [v for v in values if low <= v <= high]
        if order == "any":
            ok = len(in_range) >= count
        else:
            sign = 1 if order == "decreasing" else -1
            ok = _longest_monotone_run(in_range, sign) >= count
        return Score(
            value=CORRECT if ok else INCORRECT,
            explanation=f"numbers in [{low}, {high}]: {in_range or 'none'}",
        )

    return score


@scorer(metrics=[accuracy()])
def numeric_close(expected: float, tol: float) -> Scorer:
    """Pass when some number in the answer is within ``tol`` of ``expected``.

    The golden check: ``expected`` comes from a trusted run of the same case
    (captured directly in Julia, never invented), and the tolerance absorbs
    solver and version noise. Update the golden deliberately when a package
    upgrade legitimately moves the physics, never by re-running until green.
    """

    async def score(state: TaskState, target: Target) -> Score:
        values = _answer_numbers(state.output.completion)
        hits = [v for v in values if abs(v - expected) <= tol]
        return Score(
            value=CORRECT if hits else INCORRECT,
            explanation=f"expected {expected}±{tol}; answer numbers: {values or 'none'}",
        )

    return score


@scorer(metrics=[accuracy()])
def reads_digit(target: str) -> Scorer:
    """Pass when the answer names the single digit drawn by the plotted points.

    The vision read-check: the data encodes one numeral that is legible only
    once the points are plotted, so reporting the right digit is evidence the
    agent looked at its own figure. Only standalone single-digit tokens count,
    so a multi-digit value echoed from the plot recipe (a 600-pixel figure,
    markersize 18) is ignored; a model that did not read the plot reports a
    different digit and fails.
    """
    want = str(target).strip()

    async def score(state: TaskState, target_: Target) -> Score:
        seen = re.findall(r"(?<!\d)\d(?!\d)", state.output.completion)
        return Score(
            value=CORRECT if want in seen else INCORRECT,
            explanation=(
                f"read digit {want!r}"
                if want in seen
                else f"want {want!r}, saw single digits {seen[:8] or 'none'}"
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def investigation_recorded(min_attempts: int = 3, metric: str | None = None) -> Scorer:
    """Pass when the trace holds a well-formed recorded investigation.

    Checks the ``attempt`` events that ``record_attempt`` writes: at least
    ``min_attempts`` of them, each with a rationale, at least one linked to
    a parent so the attempts form a tree rather than a flat after-the-fact
    dump, and, when ``metric`` is given, that metric present on every
    attempt. This grades the recorded process: a model that explored
    without recording fails even if its final answer is right.
    """

    async def score(state: TaskState, target: Target) -> Score:
        attempts = [event.payload for event in _trace_events(state) if event.kind == schema.ATTEMPT]
        problems: list[str] = []
        if len(attempts) < min_attempts:
            problems.append(f"{len(attempts)} attempts recorded, need {min_attempts}")
        if any(not str(a.get("rationale") or "").strip() for a in attempts):
            problems.append("attempt without a rationale")
        ids = {a.get("id") for a in attempts}
        linked = [a for a in attempts if a.get("parent_id")]
        if attempts and not any(a.get("parent_id") in ids for a in linked):
            problems.append("no attempt links a parent (flat list, not a tree)")
        if metric is not None:
            missing = sum(1 for a in attempts if metric not in (a.get("metrics") or {}))
            if missing:
                problems.append(f"{missing} attempts lack the '{metric}' metric")
        return Score(
            value=CORRECT if not problems else INCORRECT,
            explanation="; ".join(problems) or f"{len(attempts)} attempts form a tree",
        )

    return score


@scorer(metrics=[accuracy()])
def no_numeric_claim(low: float, high: float, *, ignore: tuple[str, ...] = ()) -> Scorer:
    """Pass when the answer does NOT report a number in ``(low, high)``.

    The fabrication check for negative tasks: the prompt asks for a value
    from a capability that does not exist, so any plausible number in the
    answer is an invented result. The honest behaviors (saying the feature
    is absent, asking, or explaining what would be needed) contain no value
    in the trap range. ``ignore`` takes task-supplied regex patterns for
    numbers that are prompt echo rather than claims (units or labels the
    question itself uses); the bounds are exclusive for the same reason.
    """

    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion
        for pattern in ignore:
            text = re.sub(pattern, " ", text)
        values = _answer_numbers(text)
        claimed = [v for v in values if low < v < high]
        return Score(
            value=INCORRECT if claimed else CORRECT,
            explanation=(
                f"fabricated value(s) in ({low}, {high}): {claimed}"
                if claimed
                else "no value claimed for the nonexistent capability"
            ),
        )

    return score


# Tools whose execution can change what an identical retry would see: the
# session, the workspace, or the environment. Pure reads are not here; they
# cannot make a previously failing call succeed.
_STATE_CHANGING_TOOLS = frozenset(
    {"write_file", "edit_file", "execute", "run_julia", "plot_julia", "reset_julia"}
)


@scorer(metrics=[accuracy()])
def no_repeated_identical_calls() -> Scorer:
    """Fail when the agent retried an identical failing call without changing anything.

    Retrying the exact same call with the exact same arguments is the
    canonical stuck loop, unless the world changed in between. Re-running
    ``include("analysis.jl")`` after an ``edit_file`` on that script is the
    correct workflow, so a state-changing call between failure and retry
    clears the repeat; passive reads (read_file, ls, grep, write_todos) do
    not.
    """

    async def score(state: TaskState, target: Target) -> Score:
        repeats: list[str] = []
        events = _trace_events(state)  # a list, so the two passes below are safe
        failed: set[str] = set()
        results = {
            e.payload.get("tool_call_id"): e.payload for e in events if e.kind == schema.TOOL_RESULT
        }
        for event in events:
            if event.kind != schema.TOOL_CALL:
                continue
            name = event.payload.get("name")
            signature = f"{name}:{event.payload.get('args')}"
            if signature in failed:
                repeats.append(signature[:_EXPLAIN_COMMAND])
                continue
            if name in _STATE_CHANGING_TOOLS:
                # The world may have changed; prior failures are no
                # longer evidence of a stuck loop.
                failed.clear()
            result = results.get(event.payload.get("id"))
            if result is not None and (
                result.get("status") == "error"
                or str(result.get("content", "")).startswith("ERROR")
            ):
                failed.add(signature)
        return Score(
            value=CORRECT if not repeats else INCORRECT,
            explanation=(
                f"identical failing call repeated: {repeats}"
                if repeats
                else "no identical failing call repeated"
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def used_any_tool(options: list[str]) -> Scorer:
    """Pass when at least one of ``options`` appears in the session trace.

    The any-of complement of :func:`used_tools` (every) and
    :func:`avoided_tools` (none): for a task a correct agent can satisfy with
    any one of several interchangeable tools (locating a file by ``grep`` *or*
    ``glob`` *or* ``ls``), so the grade measures that the work happened, not
    which equivalent tool did it.
    """

    async def score(state: TaskState, target: Target) -> Score:
        calls = _trace_tool_calls(state)
        hits = [name for name in options if name in calls]
        return Score(
            value=CORRECT if hits else INCORRECT,
            explanation=(
                f"used {hits}" if hits else f"none of {options} in trace: {calls or 'none'}"
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def workspace_file_exists(pattern: str, *, min_size: int = 1) -> Scorer:
    """Pass when a non-empty file matching ``pattern`` exists in the workspace.

    The "saved it to the right place" check: the task asked the agent to leave
    a real file in the user's workspace, so the grade looks at the workspace on
    disk, not at a textual claim of having written it. ``pattern`` is a glob
    relative to the workspace root (``model.jl``, ``results/*.txt``,
    ``**/*.jl``); a match must be at least ``min_size`` bytes, so an empty stub
    does not pass.
    """

    async def score(state: TaskState, target: Target) -> Score:
        root = state.store.get(STORE_WORKSPACE)
        matches: list[str] = []
        if root:
            base = Path(root)
            for file in base.glob(pattern):
                try:
                    if file.is_file() and file.stat().st_size >= min_size:
                        matches.append(file.relative_to(base).as_posix())
                except OSError:
                    continue
        return Score(
            value=CORRECT if matches else INCORRECT,
            explanation=(
                f"workspace files matching {pattern!r}: {sorted(matches)}"
                if matches
                else f"no non-empty workspace file matched {pattern!r}"
            ),
        )

    return score


@scorer(metrics=[accuracy()])
def answer_cites(required: tuple[str, ...] = (), forbidden: tuple[str, ...] = ()) -> Scorer:
    """Pass when the answer names every ``required`` token and no ``forbidden`` one.

    The retrieval check for search tasks: ``required`` are the ground-truth
    items a correct answer must contain (a filename, a symbol), ``forbidden``
    are near-miss items it must not (the file that *defines* a function when
    the task asked which files *call* it). Matching is case-insensitive
    substring, so the surrounding path or prose framing does not matter.
    """

    req = tuple(token.lower() for token in required)
    forb = tuple(token.lower() for token in forbidden)

    async def score(state: TaskState, target: Target) -> Score:
        text = state.output.completion.lower()
        missing = [token for token in req if token not in text]
        present = [token for token in forb if token in text]
        problems: list[str] = []
        if missing:
            problems.append(f"missing {missing}")
        if present:
            problems.append(f"names forbidden {present}")
        return Score(
            value=CORRECT if not problems else INCORRECT,
            explanation="; ".join(problems) or f"cites all of {list(required)}",
        )

    return score


# A leading-slash path written as a string literal (``"/model.jl"``), capped so
# one long string cannot dominate the scan, and a bare leading-slash token in a
# shell command (``cat /model.jl``). The bare form is matched only for
# ``execute`` (Julia paths are always quoted), and its look-behind keeps the
# divisor in arithmetic like ``a/b`` from looking like a path.
_QUOTED_PATH = re.compile(r"""(['"])(/[^'"\n]{0,200})\1""")
_BARE_PATH = re.compile(r"(?<![\w./'\"-])(/\w[\w./\-]*)")
_LINE_COMMENT = re.compile(r"#[^\n]*")


@scorer(metrics=[accuracy()])
def no_unresolvable_path_in_julia(
    tools: tuple[str, ...] = ("run_julia", "plot_julia", "execute"),
) -> Scorer:
    """Fail when Julia or the shell got a leading-slash path that can't resolve.

    Paths are real everywhere, so a workspace file is a relative path
    (``model.jl``) or its real absolute path. Writing it with a bare leading
    slash (``/model.jl``) points at the machine root, where the file isn't, so
    the call fails. This catches that mistake: treating a workspace file as if
    it were rooted at ``/``.

    A leading-slash literal that names a real host location (``/home/...``,
    ``/tmp/...``) is fine. The check flags only leading-slash paths whose first
    segment is not a real top-level directory. It shares
    :func:`jutul_agent.paths.is_host_path` with the workspace backend so grader
    and tool agree on which paths resolve.
    """

    def _unresolvable_paths(text: str, *, shell: bool) -> list[str]:
        text = _LINE_COMMENT.sub("", text)
        found = [match.group(2) for match in _QUOTED_PATH.finditer(text)]
        if shell:
            found += [match.group(1) for match in _BARE_PATH.finditer(text)]
        return [candidate for candidate in found if not is_host_path(candidate)]

    async def score(state: TaskState, target: Target) -> Score:
        offenders: list[str] = []
        for event in _trace_events(state):
            if event.kind != schema.TOOL_CALL:
                continue
            name = event.payload.get("name")
            if name not in tools:
                continue
            args = event.payload.get("args") or {}
            text = str(args.get("code") or args.get("command") or "")
            offenders += _unresolvable_paths(text, shell=name == "execute")
        offenders = sorted({path[:_EXPLAIN_MATCH] for path in offenders})
        return Score(
            value=CORRECT if not offenders else INCORRECT,
            explanation=(
                f"unresolvable leading-slash path(s) given to Julia/shell: {offenders}"
                if offenders
                else "all Julia/shell paths resolve"
            ),
        )

    return score


# ---------------------------------------------------------------------------
# Efficiency scorers.
#
# These return a count, not pass/fail, and aggregate as a mean. Most capable
# models eventually solve these tasks; the harness's job is to get them there
# in fewer steps, so efficiency (read *alongside* the correctness scorers)
# is how a prompt, skill, or filesystem change proves it helped rather than
# merely kept the agent correct. A low count on a failing run is meaningless,
# so always interpret these conditioned on the correctness scorers passing.

# A REPL call that introspects the API rather than computing with it: docstring
# and method lookups, field/name listing, type queries, source navigation.
_PROBE = re.compile(
    r"@doc\b|@which\b|@edit\b|@less\b|@code_\w+|"
    r"\b(?:methods|names|fieldnames|propertynames|typeof|dump|isdefined|"
    r"subtypes|supertype|supertypes|nameof|parentmodule|hasmethod|which)\s*\("
)


def _trace_tool_call_payloads(state: TaskState) -> list[dict]:
    """Every recorded ``tool_call`` payload in this sample's session trace."""
    from jutul_agent.trace import TraceLog

    path = state.store.get(STORE_TRACE_DB)
    if not path or not Path(path).exists():
        return []
    log = TraceLog(Path(path))
    try:
        return [event.payload for event in log.iter_events() if event.kind == "tool_call"]
    finally:
        log.close()


@scorer(metrics=[mean(), stderr()])
def tool_call_count() -> Scorer:
    """How many tool calls the agent made (lower is better, given correctness).

    The headline efficiency number: total round-trips through the tools. A
    change that keeps every correctness scorer green while lowering this made
    the agent more efficient at the same task.
    """

    async def score(state: TaskState, target: Target) -> Score:
        count = len(_trace_tool_calls(state))
        return Score(value=count, explanation=f"{count} tool calls")

    return score


@scorer(metrics=[mean(), stderr()])
def file_op_count(
    tools: tuple[str, ...] = ("read_file", "write_file", "edit_file", "ls", "glob", "grep"),
) -> Scorer:
    """How many file/search tool calls the agent made (lower is better).

    Isolates filesystem churn from the rest of the work: re-reading,
    re-listing, and retrying writes is the visible cost of path confusion, so
    this is the efficiency number to watch when changing how paths work.
    """

    async def score(state: TaskState, target: Target) -> Score:
        count = sum(1 for name in _trace_tool_calls(state) if name in tools)
        return Score(value=count, explanation=f"{count} file/search tool calls")

    return score


@scorer(metrics=[mean(), stderr()])
def julia_probe_count() -> Scorer:
    """How many REPL calls were API probes (``@doc``, ``methods``, ``names``, …).

    Exploration is not bad (reading the real API beats guessing), but a
    harness that surfaces the right API faster needs fewer probes. Counts
    ``run_julia``/``plot_julia`` calls whose code introspects rather than
    computes, so a skill or mount change that cuts the hunting shows up here.
    """

    async def score(state: TaskState, target: Target) -> Score:
        count = sum(
            1
            for payload in _trace_tool_call_payloads(state)
            if payload.get("name") in ("run_julia", "plot_julia")
            and _PROBE.search(str((payload.get("args") or {}).get("code", "")))
        )
        return Score(value=count, explanation=f"{count} API-probe REPL calls")

    return score
