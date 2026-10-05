"""OpenRouter tool-image compatibility at the request boundary."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.messages.block_translators.openai import convert_to_openai_image_block
from langchain_openrouter import ChatOpenRouter


class ChatOpenRouterWithToolImages(ChatOpenRouter):
    """Serialize plot tool images in the format accepted by OpenRouter's SDK.

    langchain-openrouter 0.2.9 normalizes user images but forwards tool content
    verbatim. Convert canonical LangChain image blocks on every request, including
    resumed histories, without changing stored messages or the native handling of
    reasoning and tool calls. Remove this shim when upstream handles tool images.
    """

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        payloads, params = super()._create_message_dicts(messages, stop)
        for payload in payloads:
            content = payload.get("content")
            if payload.get("role") == "tool" and isinstance(content, list):
                payload["content"] = [
                    convert_to_openai_image_block(block)
                    if isinstance(block, dict) and block.get("type") == "image"
                    else block
                    for block in content
                ]
        return payloads, params
