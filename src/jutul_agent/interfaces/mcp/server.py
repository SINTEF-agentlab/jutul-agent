"""The MCP server: the session's tools, served over stdio.

Protocol glue only. The runtime owns the session; this maps it to what an MCP
client sees: tool descriptors (the same specs the bundled agent's model is
given), results (text, images, and links into the viewer), Julia's live output
as progress notifications, and the instructions that orient the client's model.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import MCPError

from jutul_agent.interfaces.mcp.runtime import (
    SURFACE,
    TEXT_LIMIT,
    McpLaunch,
    McpRuntime,
    StartupFailed,
    ToolOutcome,
    UnknownTool,
    split_finished,
    status_line,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from jutul_agent.agent.capabilities import Capability
    from jutul_agent.simulators.base import SimulatorAdapter

# How often live output is forwarded: often enough to watch a solve, not so often
# that a chatty one floods the client.
PROGRESS_INTERVAL = 0.5
# How much of an unfinished line is kept for the next window: a line longer than
# this is never shown whole anyway.
_CARRY_LIMIT = 4096
# The longest a shutdown waits for the session to close (Julia, then the viewer).
_CLOSE_TIMEOUT = 60.0


def build_server(runtime: McpRuntime) -> Server[McpRuntime]:
    """The MCP server for ``runtime``'s session, which it starts and stops."""
    from jutul_agent import __version__

    @contextlib.asynccontextmanager
    async def lifespan(_server: Server[McpRuntime]) -> AsyncIterator[McpRuntime]:
        try:
            await runtime.start()
            yield runtime
        finally:
            # Shielded, so a shutdown that arrives as a cancellation still stops
            # Julia rather than orphaning it; bounded, as the kernel's close is.
            with anyio.move_on_after(_CLOSE_TIMEOUT, shield=True):
                await runtime.aclose()

    async def list_tools(_ctx: Any, _params: Any) -> types.ListToolsResult:
        try:
            tools = await runtime.tools()
        except StartupFailed as exc:
            raise MCPError(types.INTERNAL_ERROR, str(exc)) from exc
        return types.ListToolsResult(tools=[tool_descriptor(tool) for tool in tools])

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        relay = ProgressRelay(ctx.session.report_progress)
        try:
            outcome = await runtime.call(
                params.name, dict(params.arguments or {}), on_output=relay.write
            )
        except StartupFailed as exc:
            return _error(str(exc))
        except UnknownTool:
            return _error(f"Unknown tool: {params.name}")
        finally:
            relay.close()
        return call_result(outcome, runtime.artifact_links(outcome.artifacts))

    return Server(
        "jutul-agent",
        version=__version__,
        title="jutul-agent",
        instructions=server_instructions(runtime.launch),
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def serve_stdio(launch: McpLaunch) -> None:
    """Serve one session over stdin/stdout until the client disconnects."""
    from mcp.server.stdio import stdio_server

    server = build_server(McpRuntime(launch))
    # While serving, the transport points fd 1 at stderr, so nothing else in the
    # process (Julia's output included) can write into the protocol stream.
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def tool_descriptor(tool: BaseTool) -> types.Tool:
    """The MCP view of a tool: the spec the bundled agent's model gets."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    from jutul_agent.tool_labels import tool_label

    spec = convert_to_openai_tool(tool)["function"]
    return types.Tool(
        name=spec["name"],
        title=tool_label(tool.name),
        description=spec.get("description") or None,
        input_schema=spec.get("parameters") or {"type": "object", "properties": {}},
    )


def call_result(outcome: ToolOutcome, links: Sequence[str]) -> types.CallToolResult:
    """A finished call as MCP content, with links to whatever it produced."""
    blocks = content_blocks(outcome.content)
    if links:
        blocks.append(types.TextContent(text="\n".join(links)))
    # The tools report their own failures as text ("ERROR: ..."), as the agent
    # reads them; MCP also flags them, so a client can show the call failed.
    first = blocks[0] if blocks else None
    failed = outcome.status == "error" or (
        isinstance(first, types.TextContent) and first.text.startswith("ERROR")
    )
    if outcome.notice:
        blocks.insert(0, types.TextContent(text=outcome.notice))
    return types.CallToolResult(content=blocks, is_error=failed)


def content_blocks(content: str | list[Any]) -> list[types.ContentBlock]:
    """A tool's reply (a string, or text and image blocks) as MCP content blocks."""
    if isinstance(content, str):
        return [types.TextContent(text=content)]
    blocks: list[types.ContentBlock] = []
    for item in content:
        if isinstance(item, str):
            blocks.append(types.TextContent(text=item))
        elif isinstance(item, dict) and item.get("type") == "text":
            blocks.append(types.TextContent(text=str(item.get("text", ""))))
        elif isinstance(item, dict) and (image := _image(item)) is not None:
            blocks.append(image)
        else:
            blocks.append(types.TextContent(text=json.dumps(item, default=str)))
    return blocks or [types.TextContent(text="")]


def _image(block: dict[str, Any]) -> types.ImageContent | None:
    """An image block (LangChain's standard or data-URL form) as MCP image content."""
    if block.get("type") == "image" and block.get("base64"):
        return types.ImageContent(
            data=str(block["base64"]), mime_type=str(block.get("mime_type") or "image/png")
        )
    url = block.get("url") if block.get("type") == "image" else None
    if block.get("type") == "image_url":
        inner = block.get("image_url")
        url = inner.get("url") if isinstance(inner, dict) else inner
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        header, data = url[len("data:") :].split(";base64,", 1)
        return types.ImageContent(data=data, mime_type=header or "image/png")
    return None


def _error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(text=message)], is_error=True)


class ProgressRelay:
    """Forward a call's live output to the client as progress, at most every interval.

    ``write`` is called from the kernel's output pump; each notification carries
    the last finished line that says something, which a client displays under the
    running call. A client that asked for no progress gets none (``report`` is a
    no-op then), and losing a notification never fails the call.
    """

    def __init__(
        self,
        report: Callable[[float, float | None, str | None], Awaitable[None]],
        *,
        interval: float = PROGRESS_INTERVAL,
    ) -> None:
        self._report = report
        self._interval = interval
        self._pending: list[str] = []
        # Output after the last line ending, held until its line is finished: a
        # chunk can end mid-line, or mid-escape.
        self._unfinished = ""
        self._sent = 0
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def write(self, text: str) -> None:
        self._pending.append(text)
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._pump())
        self._wake.set()

    async def _pump(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            finished, rest = split_finished(self._unfinished + "".join(self._pending))
            self._pending.clear()
            self._unfinished = rest[-_CARRY_LIMIT:]
            line = status_line(finished)
            if line:
                self._sent += 1
                with contextlib.suppress(Exception):
                    await self._report(self._sent, None, line)
            await asyncio.sleep(self._interval)

    def close(self) -> None:
        """Stop forwarding; the result carries the full output."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()


def server_instructions(launch: McpLaunch) -> str:
    """What the client's model is told about this server when it connects."""
    from jutul_agent.agent.capabilities import select_for_surface

    adapter = launch.simulator
    imports = ", ".join(adapter.package_imports)
    primary = adapter.primary_package
    lines = [
        f"jutul-agent runs one persistent Julia session for {adapter.display_name} "
        f"({imports}), working in {launch.workspace}.",
        "",
        "- `run_julia` evaluates code in that session; state persists across calls. Run "
        "Julia only through it, never as a separate `julia` process, which shares no "
        "state and pays a full precompile.",
        f"- Start with `using {imports}`: the session loads them but does not import them.",
        "- Check an API before relying on it: `@doc`, `methods`, `names`, and the "
        f"installed source under `pkgdir({primary})` (its `examples/`, `docs/`, `src/`).",
        "- For code the user keeps, write a `.jl` file in the workspace and "
        "`run_julia('include(\"file.jl\")')` it.",
        "- Print summaries rather than whole arrays, states or logs (`size`, `extrema`, "
        "a few entries); a value a call returns is already shown shortened. Text past "
        f"{TEXT_LIMIT:,} characters reaches you as its start and end, and the result "
        "names a file holding all of it.",
        "- Build figures only with `plot_julia`, never `run_julia`. Each plot is saved "
        "as a PNG and the result links to an interactive view the user opens in a "
        "browser; share that link. Give related plots a stable `slot`, and pass "
        "`view=true` only when you need to see the image yourself. A simulator's own "
        "plotting functions may open windows of their own; the guides say how to get "
        "the figure back instead.",
        "- `recapture_plot(slot=...)` shows you a live view as the user left it "
        "(rotated, zoomed, stepped). It reads the view from the browser, so the plot's "
        "link must be open in one: if it is not, open it (in your browser canvas, if "
        "you have one) or ask the user to. You can't step a live view or change its "
        "field yourself: plot again with the same `slot`.",
        "- `record_attempt` logs an approach and its outcome; `write_report` writes an "
        "HTML report of the session.",
        "- The first call can take a while: Julia loads the simulator once per session.",
    ]
    skills = skill_index(adapter, select_for_surface(list(launch.capabilities), SURFACE))
    if skills:
        lines += ["", "Guides (read the file before working on its topic):"]
        lines += [f"- {name}: {description} ({path})" for name, description, path in skills]
    return "\n".join(lines)


def skill_index(
    adapter: SimulatorAdapter, capabilities: Sequence[Capability]
) -> list[tuple[str, str, Path]]:
    """The skills the bundled agent would have, as (name, description, SKILL.md)."""
    from jutul_agent.agent.builder import skill_sources
    from jutul_agent.agent.capabilities import collect_skill_dirs

    roots = [
        Path(source if isinstance(source, str) else source[0]) for source in skill_sources(adapter)
    ]
    roots += [Path(directory) for directory, _label in collect_skill_dirs(capabilities)]
    index: list[tuple[str, str, Path]] = []
    for root in roots:
        for skill_md in sorted(root.glob("*/SKILL.md")):
            meta = _frontmatter(skill_md)
            name = str(meta.get("name") or skill_md.parent.name)
            index.append((name, str(meta.get("description") or "").strip(), skill_md))
    return index


def _frontmatter(path: Path) -> dict[str, Any]:
    """A SKILL.md's YAML frontmatter, or nothing if it has none that parses."""
    import yaml

    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    block = text[3:].split("\n---", 1)[0]
    with contextlib.suppress(yaml.YAMLError):
        meta = yaml.safe_load(block)
        if isinstance(meta, dict):
            return meta
    return {}
