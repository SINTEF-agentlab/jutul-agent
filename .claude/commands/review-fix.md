---
description: Fix one issue found by the session reviewer (pass the issue id).
argument-hint: "<issue-id> (from `makker review`)"
allowed-tools: Bash(makker review:*), Read, Grep, Glob, Edit, Write
---

Fix the reviewer issue **${ARGUMENTS}**.

1. Get the brief (problem, evidence across sessions, where the fix belongs, and
   transcript pointers):

   ```
   makker review fix ${ARGUMENTS}
   ```

2. Confirm the problem against the current code first; it may already be resolved. If
   it is, say so and run `makker review resolve ${ARGUMENTS}` instead of changing
   code.

3. Otherwise make the smallest correct change where it belongs, keep it general (no
   hard-coding for one session), add or update a test, and verify.

4. When done and verified, run `makker review resolve ${ARGUMENTS}` and summarise
   what you changed.
