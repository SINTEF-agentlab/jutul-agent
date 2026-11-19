"""The critic prompt for a session review.

Deliberately general: a domain-agnostic rubric for the universal failure modes
(silent errors, unconverted units, ignored validation, implausible results,
unverified claims, substituting a default for what was asked), plus the active
simulator's own ``review_hints`` injected from its adapter. Simulator-specific
checks (reservoir permeability ranges, battery voltage windows, …) live with that
simulator, not in this shared prompt.
"""

from __future__ import annotations

from typing import Any

SYSTEM = """\
You are a meticulous reviewer auditing one session of makker — an AI agent \
that drives Julia scientific simulators on a user's behalf from a persistent Julia \
REPL. You work for the agent's *developer*, not the user. Your job is to find what \
went wrong or was missed in this session, and turn it into concrete improvements. \
The active simulator and any domain-specific things to weigh for it are given with \
the transcript; reason from those rather than assuming a particular domain.

The failure that matters most is the SILENT one: the run completes, nothing throws, \
the answer looks plausible — yet an input was nonsense (a unit left unconverted, a \
sign error, an out-of-range value) or a result is physically impossible. These never \
surface as errors, so only careful reading catches them.

What to look for (examples, not an exhaustive list — reason from the actual session \
and from any simulator-specific notes given with the transcript):
- Physically suspect inputs: a value orders of magnitude outside a sensible range \
usually means a unit was left unconverted; a fraction outside 0..1; a time that looks \
like days/years where seconds are expected; a quantity negative that can't be.
- Validation: if the simulator offers input/case validation, did the agent run it and \
act on it? Running anyway after a warning, or skipping available validation, is a \
finding; validation passing while the case is still wrong is a *validation gap*.
- Results that ran but are implausible: off by orders of magnitude, a conserved \
quantity that isn't conserved, a "converged" run that clearly diverged.
- Agent mistakes: wrong/guessed API, misread output, gave up, claimed something it did \
not verify, did unnecessary/destructive work, or quietly substituted a simpler default \
for what was explicitly asked.
- A missing tool, skill, or check that would have prevented the issue.

Be specific and evidence-based: quote the exact line(s) from the session that show \
the problem. Do not invent issues — if the session is genuinely clean, return no \
findings. Prefer a few high-signal findings over many speculative ones. The summary \
should also note what the agent did *well*, not only what went wrong — a fair picture \
of the session is more useful than a list of complaints.

The category and fix_target labels below are a guide, not a checklist. They exist to \
group similar issues, not to force a real problem into a box. If a finding does not fit, \
use "other" and explain it plainly in the detail field; describing the actual issue \
precisely always beats picking a near-miss label. Use the detail field for any nuance \
the short evidence/suggestion can't hold (why it matters, scope, an alternative the \
agent missed).

For each finding:
- category (guide): validation-gap, agent-error, plausible-but-wrong, silent-failure, \
tooling-gap, or other
- severity: low, medium, or high (high = wrong scientific result or a crash the agent \
papered over)
- fix_target (guide): where a fix would most naturally go — case-validation (extend the \
active simulator's input/case validation, if it has one), skill (clarify agent guidance), \
prompt (system prompt), eval (add a regression case), code (makker change), or other
- detail: optional free-form context that doesn't fit evidence/suggestion

Reply with STRICT JSON and nothing else:
{
  "summary": "two or three sentences: overall quality, what went well, and the main gap",
  "findings": [
    {
      "category": "...",
      "severity": "...",
      "title": "short imperative title",
      "evidence": "quoted/located evidence from the session",
      "suggestion": "the concrete fix",
      "fix_target": "...",
      "detail": "optional: nuance, scope, why it matters"
    }
  ]
}
If nothing is wrong, return an empty "findings" list.
"""


def _simulator_review_hints(simulator: str | None) -> str:
    """The active simulator's reviewer hints, or ``''`` (unknown sim, or none declared).

    Discovered from the simulator's own adapter, so simulator-specific guidance stays
    with that simulator instead of leaking into the general rubric.
    """
    if not simulator:
        return ""
    try:
        from makker.simulators.registry import get

        return get(simulator).review_hints
    except (KeyError, ImportError):
        return ""


def build_user_message(
    transcript: str, *, simulator: str | None, ground_truth: str | None = None
) -> str:
    sim = f" The active simulator was {simulator}." if simulator else ""
    hints = _simulator_review_hints(simulator)
    sim_block = (
        f"\n\nDomain-specific things to weigh for {simulator} (in addition to the "
        f"general rubric above):\n{hints}"
        if hints
        else ""
    )
    truth = ""
    if ground_truth:
        truth = (
            "\n\nThis was an EVAL run with a known correct answer: "
            f"{ground_truth}. Judge whether the agent's result matches it and is "
            "physically sensible; a plausible-looking answer that misses this is a "
            "high-severity finding."
        )
    return (
        f"Review the following makker session and report findings as JSON.{sim}{truth}"
        f"{sim_block}\n\n"
        "=== SESSION TRANSCRIPT ===\n"
        f"{transcript}\n"
        "=== END SESSION TRANSCRIPT ==="
    )


def full_prompt(transcript: str, *, simulator: str | None, ground_truth: str | None = None) -> str:
    """The whole critic prompt as one document, for a coding agent to act on.

    ``makker review prompt <session>`` emits this; a tool like Claude Code
    reads it, produces the JSON, and feeds it back via ``review ingest`` — no API
    call for the expensive read.
    """
    user = build_user_message(transcript, simulator=simulator, ground_truth=ground_truth)
    return f"{SYSTEM}\n\n{user}"


# Where a fix of each kind usually lives, to point the coding agent at the right
# part of the repo. A hint, not a rule — the agent confirms against the actual code.
_FIX_TARGET_HINTS = {
    "case-validation": (
        "Extend the active simulator's input/case validation (e.g. JutulDarcy's "
        "`CaseValidation`); if it has none to extend, a skill or code check may fit better."
    ),
    "skill": (
        "Clarify or extend a skill under `src/makker/simulators/*/skills/` or "
        "`src/makker/simulators/shared_skills/`. Keep the change general (see the "
        "project rule that skills stay general; task-specific recipes go in demo/eval files)."
    ),
    "prompt": "Adjust the system prompt in `src/makker/agent/` (the prompt builder).",
    "eval": (
        "Add a regression case under `src/makker/eval/tasks/`. Prefer a golden-backed "
        "check over brittle trace matching; capture the golden from a trusted run."
    ),
    "code": "A makker code change under `src/makker/`. Add or update a test.",
}


def fix_prompt(issue: Any, *, transcript_paths: list[str] | None = None) -> str:
    """A self-contained brief for a coding agent to fix one curated issue.

    ``makker review fix <id>`` emits this. It states the problem, the evidence
    gathered across sessions, where the fix most likely belongs, and asks for a
    minimal, tested change plus marking the issue resolved.
    """
    hint = _FIX_TARGET_HINTS.get(issue.fix_target, "Use your judgement on where this belongs.")
    examples = "\n".join(f"  - {ex}" for ex in issue.examples) or "  (none recorded)"
    sessions = ", ".join(issue.sessions) or "(none)"
    transcripts = ""
    if transcript_paths:
        joined = "\n".join(f"  - {p}" for p in transcript_paths)
        transcripts = f"\nRendered transcripts you can read for full context:\n{joined}\n"
    return (
        "You are fixing one recurring issue found by the makker session reviewer. "
        "Work in the makker repo. Make the smallest correct change, add or update a "
        "test, and keep it general (don't hard-code for one session).\n\n"
        f"# Issue: {issue.title}\n"
        f"- id: {issue.id}\n"
        f"- category: {issue.category}\n"
        f"- severity: {issue.severity}\n"
        f"- seen: {issue.count}x across {len(issue.sessions)} session(s)"
        f" (last on version {issue.last_version or 'unknown'})\n"
        f"- where the fix likely belongs ({issue.fix_target}): {hint}\n\n"
        f"## Evidence from the sessions\n{examples}\n\n"
        f"## Sessions\n{sessions}\n{transcripts}\n"
        "## Do\n"
        "1. Confirm the problem against the current code (it may already be fixed — if so, "
        "say so and run `makker review resolve "
        f"{issue.id}` instead of changing code).\n"
        "2. Implement the minimal fix where it belongs; verify it.\n"
        "3. Add/adjust a test that would have caught this.\n"
        f"4. When done and verified, run `makker review resolve {issue.id}`.\n"
    )
