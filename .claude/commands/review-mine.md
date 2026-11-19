---
description: Mine makker sessions for issues with no API cost (you are the reviewer).
argument-hint: "[limit] (max sessions to review this run; default 10)"
allowed-tools: Bash(makker review:*), Read, Write
---

You are acting as the session reviewer for makker: an autonomous critic that
reads finished agent sessions and flags what the agent (or its in-Julia validation)
missed. This is the coding-agent path: you do the expensive reading yourself, so it
costs no API. Be skeptical and evidence-based.

Mine up to ${ARGUMENTS:-10} of the most recent unreviewed sessions, one at a time:

1. Load the current state so you can curate as you go:

   ```
   makker review issues --json     # existing curated issues (titles + ids)
   makker review pending --json --limit ${ARGUMENTS:-10}
   ```

   Curation is your job here, with no API: when a finding is the same root cause as
   an existing issue, reuse that issue's exact title in your findings JSON so `ingest`
   merges it instead of creating a near-duplicate. Only invent a new title for a
   genuinely new problem. Match on root cause, not wording.

   Each pending entry has a `session_id`. If the list is empty, stop and say so.

2. For each session id, in order:

   a. Get the critic prompt and full transcript:

      ```
      makker review prompt <session_id>
      ```

      The output begins with the reviewer rubric and then the session transcript.
      Read all of it, including any "known correct answer" line for eval runs.

   b. Review it exactly as the rubric says. The failure that matters most is the
      silent one: a run that completed and looks plausible but used a nonsensical
      input (a unit left unconverted, a sign error, an out-of-range value) or produced
      a physically impossible result. Also flag skipped case validation, wrong or
      guessed API, unverified claims, and missing tools or checks. Quote exact lines
      as evidence. If the session is genuinely clean, return an empty `findings` list;
      do not invent issues.

   c. Write the result as strict JSON to `.makker-review/<session_id>.json`:

      ```json
      {
        "summary": "2-3 sentences: overall quality, what went well, and the main gap",
        "findings": [
          {
            "category": "validation-gap|agent-error|plausible-but-wrong|silent-failure|tooling-gap|other",
            "severity": "low|medium|high",
            "title": "short imperative title (reuse an existing issue's title to merge)",
            "evidence": "quoted/located evidence from the session",
            "suggestion": "the concrete fix",
            "fix_target": "case-validation|skill|prompt|eval|code|other",
            "detail": "optional: nuance, scope, why it matters"
          }
        ]
      }
      ```

      The category and fix_target labels are a guide, not a box. If a finding does not
      fit, use `other` and explain in `detail`. A precise description beats a near-miss
      label.

   d. Ingest it (logs the findings and curates them into the issue store):

      ```
      makker review ingest <session_id> --from .makker-review/<session_id>.json
      ```

3. When every session is done, show the ranked result and report a short summary to
   me: how many sessions you reviewed, how many had findings, and the top recurring
   issues.

   ```
   makker review
   ```

   The user can open the interactive dashboard themselves with `makker review
   dashboard`.

Keep findings high-signal: a few well-evidenced issues beat many speculative ones.
Severity `high` means a wrong scientific result or a crash papered over.
