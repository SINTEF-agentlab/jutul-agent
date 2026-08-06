"""Tests for streamed tool-output normalization."""

from __future__ import annotations

from langchain_core.messages import ToolMessage

from jutul_agent.agent.tool_output import is_interrupt_payload, normalize_tool_output


def test_normalize_tool_message_object() -> None:
    message = ToolMessage(content="hello", tool_call_id="call-1", name="read_file")
    assert normalize_tool_output(message) == "hello"


def test_normalize_tool_message_repr() -> None:
    raw = (
        "[ToolMessage(content='     1\\tline one\\n     2\\tline two', "
        "name='read_file', tool_call_id='call-1', additional_kwargs={})]"
    )
    assert normalize_tool_output(raw) == "     1\tline one\n     2\tline two"


def test_interrupt_payload_detection() -> None:
    text = "Interrupt(value={'action_requests': [{'name': 'write_file'}]})"
    assert is_interrupt_payload(text) is True


def test_normalize_multimodal_content_keeps_text_and_drops_the_image() -> None:
    """A tool that hands the model an image must not paste it into the transcript.

    The plot tools reply with a summary plus the rendered PNG as base64, so
    serializing the whole list puts a screenful of encoded bytes where the
    one-line summary belongs. The image reaches the user as the artifact card.
    """

    blocks = [
        {"type": "text", "text": "recaptured view to artifacts/recapture-ab12.png"},
        {"type": "image", "mime_type": "image/png", "base64": "iVBORw0KGgo" + "A" * 4000},
    ]
    assert normalize_tool_output(blocks) == "recaptured view to artifacts/recapture-ab12.png"


def test_normalize_plain_list_data_is_still_json() -> None:
    """Only tagged content blocks are flattened; list data keeps its structure."""

    assert normalize_tool_output([{"a": 1}, {"b": 2}]) == '[{"a": 1}, {"b": 2}]'
    assert normalize_tool_output(["x", {"y": 1}]) == '["x", {"y": 1}]'
