"""Regression for plot images rejected by the OpenRouter SDK on every retry."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from openrouter import OpenRouter
from PIL import Image

from jutul_agent.agent.openrouter import ChatOpenRouterWithToolImages
from jutul_agent.agent.plot_julia import _reply


@pytest.mark.parametrize("stream", [False, True])
async def test_plot_image_retry_and_saved_history(tmp_path: Path, stream: bool) -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        response = {
            "id": "test-response",
            "object": "chat.completion",
            "created": 0,
            "model": "vendor/test-model",
            "system_fingerprint": None,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Plot received"},
                    "finish_reason": "stop",
                }
            ],
        }
        if stream:
            response["object"] = "chat.completion.chunk"
            response["choices"][0]["delta"] = response["choices"][0].pop("message")
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(200, json=response)

    png = tmp_path / "plot.png"
    Image.new("RGB", (2, 2)).save(png)
    content = _reply("Geometry plot", png, view=True)
    assert isinstance(content, list)
    history = [
        HumanMessage(content="Show the plot"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "plot_julia", "args": {}, "id": "plot-call"},
                {"name": "run_julia", "args": {}, "id": "text-call"},
            ],
            additional_kwargs={"reasoning_content": "Check the geometry"},
        ),
        ToolMessage(content=content, tool_call_id="plot-call", artifact={"path": "plot.png"}),
        ToolMessage(content="Text-only result", tool_call_id="text-call"),
    ]
    saved = json.dumps(messages_to_dict(history))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        # Keep the real SDK's validation and replace only its HTTP transport.
        sdk = OpenRouter(api_key="test-key", async_client=client)
        model = ChatOpenRouterWithToolImages(
            model="vendor/test-model", api_key="test-key", client=sdk
        )
        # Retry the original history, then simulate reopening its saved messages.
        for messages in (history, messages_from_dict(json.loads(saved))):
            if stream:
                answer = "".join([chunk.text async for chunk in model.astream(messages)])
            else:
                answer = (await model.ainvoke(messages)).text
            assert answer == "Plot received"
            assert json.dumps(messages_to_dict(messages)) == saved

    assert len(requests) == 2
    for request in requests:
        tool = request["messages"][2]
        assert tool["tool_call_id"] == "plot-call"
        assert tool["content"] == [
            {"type": "text", "text": "Geometry plot"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{content[1]['base64']}"},
            },
        ]
        assert request["messages"][3]["content"] == "Text-only result"
        assert request["messages"][1]["reasoning"] == "Check the geometry"
        assert request["messages"][1]["tool_calls"][0]["id"] == "plot-call"
