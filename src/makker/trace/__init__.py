"""Per-session event log: append-only store plus the middleware that feeds it.

Event kind names and shared payload constructors live in
``makker.trace.schema``.
"""

from makker.trace.log import Event, TraceLog
from makker.trace.recorder import TraceRecorder

__all__ = ["Event", "TraceLog", "TraceRecorder"]
