"""Keep OpenAI Responses-API reasoning items intact across replayed history.

The Responses API is item-addressed: a reasoning model returns ``reasoning``
items (``rs_...``) and each following ``function_call`` records which reasoning
item it belongs to. Every request replays the conversation, and the server
rejects it outright when a ``function_call`` arrives without its reasoning
item. Because the offending turn is checkpointed, the rejection then repeats
on every retry: the session cannot continue until the replay is fixed.

The defect is upstream: langchain-openai's ``_implode_reasoning_blocks`` folds
consecutive reasoning content blocks back into one Responses item, deciding
purely on adjacency without checking that the blocks share an item id. A
single response can carry two reasoning items back to back; the second is then
swallowed into the first one's ``summary`` and its id never reaches the
server, while the ``function_call`` citing it still does.

``enable_reasoning_item_grouping`` feeds upstream one same-id run at a time,
so its merge can only act within a single item. All conversion logic stays
upstream; the wrapper only chooses what upstream sees at once. Stored messages
are complete (the loss is outbound-only), so installing the fix also repairs
threads recorded before it. Drop the shim once upstream keys its merge on the
item id; ``tests/test_openai_responses.py`` pins the upstream behavior and
flags when that day comes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

__all__ = ["enable_reasoning_item_grouping"]

# Marks an already-wrapped callable so installation is idempotent.
_PATCH_FLAG = "_jutul_agent_reasoning_item_grouping"


def _is_exploded_reasoning(block: Any) -> bool:
    """Whether upstream would fold ``block`` into the preceding reasoning item.

    Mirrors the branch ``_implode_reasoning_blocks`` merges on: a reasoning
    block still carrying langchain's per-part ``reasoning`` text rather than an
    assembled Responses ``summary``. Everything else is passed through by
    upstream untouched and cannot absorb a neighbour.
    """

    return (
        isinstance(block, dict)
        and block.get("type") == "reasoning"
        and "summary" not in block
        and "reasoning" in block
    )


def _runs_by_item_id(blocks: Iterable[Any]) -> Iterator[list[Any]]:
    """Split ``blocks`` into consecutive runs holding at most one reasoning item.

    A run is cut only where one reasoning item's blocks directly precede a
    different item's, which is exactly where upstream's adjacency-only merge
    would fuse the two. Any other block already ends the merge upstream, so it
    clears the current id without forcing a cut.
    """

    run: list[Any] = []
    item_id: str | None = None
    for block in blocks:
        if _is_exploded_reasoning(block):
            if item_id is not None and block.get("id") != item_id:
                yield run
                run = []
            item_id = block.get("id")
        else:
            item_id = None
        run.append(block)
    if run:
        yield run


def _wrap_implode(original: Callable[[list], Iterable]) -> Callable[[list], Iterable]:
    """An ``_implode_reasoning_blocks`` that merges within one item id only."""

    if getattr(original, _PATCH_FLAG, False):
        return original

    def implode_reasoning_blocks(blocks: list) -> Iterator:
        for run in _runs_by_item_id(blocks):
            yield from original(run)

    implode_reasoning_blocks.__wrapped__ = original  # type: ignore[attr-defined]
    setattr(implode_reasoning_blocks, _PATCH_FLAG, True)
    return implode_reasoning_blocks


def enable_reasoning_item_grouping() -> None:
    """Stop langchain-openai merging reasoning blocks across item ids.

    Idempotent and safe to call on every agent build; a no-op when
    langchain-openai isn't installed or has moved the helper. The module global
    is resolved at call time, so patching it reaches every already-built model.
    """

    try:
        from langchain_openai.chat_models import _compat
    except ImportError:  # pragma: no cover - langchain-openai is a hard dep
        return
    current = getattr(_compat, "_implode_reasoning_blocks", None)
    if current is not None:
        _compat._implode_reasoning_blocks = _wrap_implode(current)
