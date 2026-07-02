"""Attempt tree built from ``attempt`` trace events.

``record_attempt`` appends one ``attempt`` event per call. The report
renderer (and any other consumer) rebuilds the parent/child tree from
those events via :func:`build_attempt_tree`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jutul_agent.trace import Event, schema


@dataclass
class Attempt:
    """One step recorded by ``record_attempt`` in the session trace."""

    id: str
    parent_id: str | None
    rationale: str
    parameters_changed: dict
    metrics: dict[str, float]
    candidate_path: str | None = None
    plot_artifact_path: str | None = None
    notes: str | None = None
    children: list[Attempt] = field(default_factory=list)


def _attempt_from_event(event: Event) -> Attempt:
    payload = event.payload
    metrics = payload.get("metrics") or {}
    return Attempt(
        id=str(payload["id"]),
        parent_id=payload.get("parent_id"),
        rationale=str(payload.get("rationale") or ""),
        parameters_changed=dict(payload.get("parameters_changed") or {}),
        metrics={str(k): float(v) for k, v in metrics.items()},
        candidate_path=payload.get("candidate_path"),
        plot_artifact_path=payload.get("plot_artifact_path"),
        notes=payload.get("notes"),
    )


def build_attempt_tree(events: list[Event]) -> list[Attempt]:
    """Return root attempts with ``children`` populated from ``attempt`` events."""

    by_id: dict[str, Attempt] = {}
    order: list[str] = []

    for event in events:
        if event.kind != schema.ATTEMPT:
            continue
        attempt = _attempt_from_event(event)
        by_id[attempt.id] = attempt
        order.append(attempt.id)

    roots: list[Attempt] = []
    for attempt_id in order:
        attempt = by_id[attempt_id]
        parent_id = attempt.parent_id
        if parent_id and parent_id in by_id:
            by_id[parent_id].children.append(attempt)
        else:
            roots.append(attempt)

    return roots
