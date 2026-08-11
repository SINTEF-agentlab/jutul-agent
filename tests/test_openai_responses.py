"""Responses-API reasoning items survive being replayed.

The shim is exercised through the real ``langchain_openai`` request builder:
what matters is not the intermediate block shape but whether every reasoning
item a ``function_call`` cites actually reaches the wire. The fixture is the
failing shape observed live: one response carrying two back-to-back reasoning
items, the second cited by the tool call that follows.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_openai.chat_models.base import _construct_responses_api_input

from jutul_agent.agent.openai_responses import enable_reasoning_item_grouping

_FIRST = "rs_first00000000"
_SECOND = "rs_second0000000"


def _reasoning(item_id: str, text: str, *, encrypted: bool, index: str) -> dict:
    block: dict = {"id": item_id, "type": "reasoning", "reasoning": text, "index": index}
    if encrypted:
        block["extras"] = {"content": [], "encrypted_content": f"enc-{item_id}"}
    return block


def _turn() -> AIMessage:
    """Two reasoning items in one response, then the tool call citing the second."""
    return AIMessage(
        content=[
            _reasoning(_FIRST, "planning", encrypted=True, index="lc_rs_305f30"),
            _reasoning(_FIRST, "still planning", encrypted=False, index="lc_rs_305f31"),
            _reasoning(_SECOND, "deciding", encrypted=True, index="lc_rs_315f30"),
            {
                "type": "tool_call",
                "id": "call_abc",
                "name": "plot_julia",
                "args": {"code": "fig"},
                "extras": {"item_id": "fc_call0000000000"},
            },
        ],
        tool_calls=[{"name": "plot_julia", "args": {"code": "fig"}, "id": "call_abc"}],
        response_metadata={"model_provider": "openai", "output_version": "v1"},
    )


def _items(message: AIMessage) -> list[dict]:
    return _construct_responses_api_input([message])


def _reasoning_ids(items: list[dict]) -> list[str]:
    return [i["id"] for i in items if i.get("type") == "reasoning"]


@pytest.fixture
def shim() -> None:
    enable_reasoning_item_grouping()


def test_upstream_still_merges_reasoning_across_item_ids() -> None:
    """The upstream bug, pinned. Asserted against the unwrapped helper, since the
    shim is a process-global install another test may already have done."""
    from langchain_openai.chat_models import _compat

    upstream = getattr(_compat._implode_reasoning_blocks, "__wrapped__", None)
    upstream = upstream or _compat._implode_reasoning_blocks
    reasoning = [b for b in _turn().content if b["type"] == "reasoning"]

    assert [item["id"] for item in upstream(reasoning)] == [_FIRST], (
        "langchain-openai no longer merges reasoning blocks across item ids; "
        "agent/openai_responses.enable_reasoning_item_grouping is obsolete."
    )


def test_every_cited_reasoning_item_reaches_the_wire(shim: None) -> None:
    items = _items(_turn())
    assert _reasoning_ids(items) == [_FIRST, _SECOND]
    # Each item keeps its own replayable payload; the summaries stay split by item.
    by_id = {i["id"]: i for i in items if i.get("type") == "reasoning"}
    assert by_id[_FIRST]["encrypted_content"] == f"enc-{_FIRST}"
    assert by_id[_SECOND]["encrypted_content"] == f"enc-{_SECOND}"
    assert [len(by_id[i]["summary"]) for i in (_FIRST, _SECOND)] == [2, 1]
    # And the function call still follows its reasoning.
    assert [i.get("type") for i in items] == ["reasoning", "reasoning", "function_call"]


def test_parts_of_one_reasoning_item_still_merge(shim: None) -> None:
    """No regression: same-id blocks are one item with one summary per part."""
    message = AIMessage(
        content=[
            _reasoning(_FIRST, "a", encrypted=True, index="lc_rs_305f30"),
            _reasoning(_FIRST, "b", encrypted=False, index="lc_rs_305f31"),
            _reasoning(_FIRST, "c", encrypted=False, index="lc_rs_305f32"),
            {"type": "text", "text": "done", "id": "msg_1"},
        ],
        response_metadata={"model_provider": "openai", "output_version": "v1"},
    )
    items = _items(message)
    assert _reasoning_ids(items) == [_FIRST]
    assert [b["text"] for b in items[0]["summary"]] == ["a", "b", "c"]


def test_installing_the_shim_twice_does_not_stack(shim: None) -> None:
    enable_reasoning_item_grouping()
    enable_reasoning_item_grouping()
    assert _reasoning_ids(_items(_turn())) == [_FIRST, _SECOND]
