"""A lab for testing, profiling, and improving makker without a live session.

The agent runtime (LangGraph + deepagents + the Textual TUI) is normally driven by a
real model and a real Julia process. This package supplies the doubles and tooling to
drive it headlessly instead, so an agent or a test can:

- render the TUI through scripted interactions and capture what it looks like
  (:mod:`makker.lab.tui`, :mod:`makker.lab.scenarios`),
- exercise error and edge paths for robustness,
- drive one real prompt end to end and capture it (:mod:`makker.lab.live`),
- and profile cold start and a turn's hot path
  (:mod:`makker.lab.profile_startup`, :mod:`makker.lab.profile_turn`).

The :mod:`makker.lab.fakes` module holds the shared doubles (a scripted
model and agent, a fake Julia session, a fake simulator adapter) used by both the
test suite and the lab.
"""

from __future__ import annotations
