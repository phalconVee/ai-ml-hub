"""The switchboard itself: many live MCP connections behind one tool list.

What this file does, in four lines:
    connect(spec)      open a client session to one server, list its tools
    tools()            every connected server's tools, renamed "server__tool"
    needs_approval()   should a human OK this call first?
    call(name, args)   route a call to the right server, return blocks for Claude

Two problems make this more than a dict of clients.

1. **Name clashes.** Two servers can both have a tool called `search`.
   Claude needs unique names, and the person reading the transcript needs
   to know which server ran what. So every tool is exposed to Claude as
   `<server>__<tool>`, and we keep a map back to (server, original name).

2. **Connection lifetime.** An MCP client is an async context manager
   (`async with Client(...) as c:`) and, under the hood, anyio insists it is
   entered and exited by *the same task*. A UI wants to open a connection
   when you flick a toggle and close it minutes later on another click,
   from a different call stack. The fix: each connection gets its own
   long-lived "keeper" task that enters the `async with`, reports the client
   back, then simply waits for a stop signal. Closing = set the signal and
   let the keeper exit its own `async with`. One task in, same task out.
"""

import asyncio
import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx2  # the HTTP library MCP 2.x and anthropic 2.x are built on
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from switchboard.config import ServerSpec, expand, missing_vars, resolve_cwd
from switchboard.settings import CONNECT_TIMEOUT_S, MAX_TOOL_RESULT_CHARS, TOOL_TIMEOUT_S

SEP = "__"  # server/tool separator in the names Claude sees


@dataclass
class HubTool:
    """One tool of one server, as the switchboard sees it."""

    server: str
    name: str            # the server's own name for it
    qualified: str       # the unique name Claude sees: server__name
    description: str
    input_schema: dict
    read_only: bool      # from the server's MCP annotations (a hint, not a guarantee)
    destructive: bool

    def for_claude(self) -> dict:
        """Translate the MCP tool definition into a Messages API tool definition.

        The two formats are nearly identical: MCP already describes inputs with
        JSON Schema, which is exactly what Claude's `input_schema` wants.
        """
        return {
            "name": self.qualified,
            "description": f"[{self.server}] {self.description}".strip(),
            "input_schema": self.input_schema or {"type": "object", "properties": {}},
        }


@dataclass
class Connection:
    spec: ServerSpec
    client: Client
    tools: list[HubTool]
    instructions: str | None
    keeper: asyncio.Task
    stop: asyncio.Event


@dataclass
class ToolOutcome:
    """A finished tool call: what goes back to Claude, plus a plain-text copy for the UI."""

    content: list[dict]   # tool_result content blocks (text and/or image)
    is_error: bool
    preview: str = ""
    server: str = ""
    tool: str = ""


def _qualify(server: str, tool: str) -> str:
    # Claude's tool names allow [A-Za-z0-9_-], max 64 chars. MCP names can contain
    # dots, slashes, etc., so normalise and clip.
    return re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}{SEP}{tool}")[:64]


def _hub_tools(server: str, mcp_tools: list[Any]) -> list[HubTool]:
    out = []
    for t in mcp_tools:
        ann = t.annotations
        # MCP's defaults when a hint is missing: read_only=False, destructive=True.
        # i.e. an unannotated tool is assumed to be able to write AND destroy.
        read_only = bool(ann and ann.read_only_hint)
        destructive = not read_only and (ann is None or ann.destructive_hint is not False)
        out.append(HubTool(
            server=server, name=t.name, qualified=_qualify(server, t.name),
            description=t.description or "", input_schema=t.input_schema,
            read_only=read_only, destructive=destructive,
        ))
    return out


def _flatten_error(exc: BaseException) -> str:
    """anyio wraps failures in ExceptionGroups; dig out the first real message."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


def to_claude_blocks(result: Any) -> tuple[list[dict], str]:
    """Convert an MCP CallToolResult's content into tool_result blocks for Claude.

    MCP content blocks and Claude content blocks are cousins:
        text           -> text
        image (base64) -> image  (Claude can actually look at it)
        resource       -> text   (embedded file contents)
        resource_link  -> text   (just the URI; Claude can ask to read it)
    Anything else becomes a text note so nothing silently disappears.
    """
    blocks: list[dict] = []
    for c in result.content:
        kind = getattr(c, "type", "")
        if kind == "text":
            blocks.append({"type": "text", "text": c.text})
        elif kind == "image":
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": c.mime_type, "data": c.data}})
        elif kind == "resource":
            res = c.resource
            body = getattr(res, "text", None) or f"(binary resource, {getattr(res, 'mime_type', '?')})"
            blocks.append({"type": "text", "text": f"[resource {res.uri}]\n{body}"})
        elif kind == "resource_link":
            blocks.append({"type": "text", "text": f"[resource link] {c.name}: {c.uri}"})
        else:
            blocks.append({"type": "text", "text": f"[{kind or 'unknown'} content omitted]"})
    # Tools with an output schema may return only structured content.
    if not blocks and result.structured_content is not None:
        blocks.append({"type": "text", "text": json.dumps(result.structured_content, indent=1)})
    if not blocks:
        blocks.append({"type": "text", "text": "(no output)"})

    # Clip long text: a 2 MB page dump costs real money on every later turn.
    budget = MAX_TOOL_RESULT_CHARS
    for b in blocks:
        if b["type"] == "text":
            if len(b["text"]) > budget:
                b["text"] = b["text"][:budget] + "\n...[truncated by switchboard]"
            budget = max(0, budget - len(b["text"]))
    preview = "\n".join(b["text"] if b["type"] == "text" else f"[{b['type']}]" for b in blocks)
    return blocks, preview


@asynccontextmanager
async def _http_transport(url: str, headers: dict[str, str]):
    """A Streamable HTTP transport with our own headers (e.g. an auth token).

    `Client("http://...")` alone would work for keyless servers; we build the
    transport ourselves so headers can be added, and own the HTTP client so it
    is closed together with the session.
    """
    async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30.0, read=300.0)) as http:
        async with streamable_http_client(url, http_client=http) as streams:
            yield streams


class Switchboard:
    """All live MCP connections. Every method must run on one event loop."""

    def __init__(self) -> None:
        self.connections: dict[str, Connection] = {}
        self.errors: dict[str, str] = {}   # last connect error per server, for the UI
        self.trusted: set[str] = set()     # qualified tool names the user said "always allow"
        self.specs: dict[str, ServerSpec] = {}

    # --- connecting -----------------------------------------------------------------

    def _target(self, spec: ServerSpec) -> Any:
        """What `mcp.Client` should connect to for this spec."""
        if spec.command:
            # stdio: Client launches the command as a subprocess and talks over its pipes.
            return StdioServerParameters(
                command=expand(spec.command),
                args=[expand(a) for a in spec.args],
                env={k: expand(v) for k, v in spec.env.items()} or None,
                cwd=resolve_cwd(spec),
            )
        # Streamable HTTP: one POST per message, responses streamed back as SSE.
        return _http_transport(expand(spec.url), {k: expand(v) for k, v in spec.headers.items()})

    async def connect(self, spec: ServerSpec, target: Any = None) -> None:
        """Open a connection and list its tools. `target` lets tests pass an in-process server."""
        await self.disconnect(spec.name)  # reconnecting = clean slate
        self.specs[spec.name] = spec
        self.errors.pop(spec.name, None)
        if target is None and (missing := missing_vars(spec)):
            self.errors[spec.name] = f"Set {', '.join(missing)} in .env first."
            return

        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        stop = asyncio.Event()

        async def keeper() -> None:
            # This task owns the connection from open to close (see module docstring).
            try:
                async with Client(target or self._target(spec), read_timeout_seconds=TOOL_TIMEOUT_S) as client:
                    tools, cursor = [], None
                    while True:  # servers may paginate their tool list
                        page = await client.list_tools(cursor=cursor)
                        tools += page.tools
                        if not (cursor := page.next_cursor):
                            break
                    ready.set_result((client, tools, client.instructions))
                    await stop.wait()  # park here until disconnect()
            except BaseException as exc:  # noqa: BLE001 - report *anything* to the UI
                if not ready.done():
                    ready.set_exception(RuntimeError(_flatten_error(exc)))
                if isinstance(exc, asyncio.CancelledError):
                    raise

        task = asyncio.create_task(keeper(), name=f"mcp:{spec.name}")
        try:
            client, tools, instructions = await asyncio.wait_for(ready, CONNECT_TIMEOUT_S)
        except Exception as exc:  # timeout, crash on start, bad URL...
            stop.set()
            task.cancel()
            self.errors[spec.name] = _flatten_error(exc) if not isinstance(exc, TimeoutError) \
                else f"No answer within {CONNECT_TIMEOUT_S:.0f}s."
            return
        self.connections[spec.name] = Connection(
            spec=spec, client=client, tools=_hub_tools(spec.name, tools),
            instructions=instructions, keeper=task, stop=stop,
        )

    async def disconnect(self, name: str) -> None:
        conn = self.connections.pop(name, None)
        if conn is None:
            return
        conn.stop.set()  # the keeper leaves its `async with`, closing the session (and the subprocess)
        try:
            await asyncio.wait_for(conn.keeper, 10)
        except (Exception, asyncio.CancelledError):
            conn.keeper.cancel()  # stuck server: stop waiting politely

    async def disconnect_all(self) -> None:
        for name in list(self.connections):
            await self.disconnect(name)

    # --- the merged tool list ----------------------------------------------------

    def tools(self) -> list[HubTool]:
        return [t for conn in self.connections.values() for t in conn.tools]

    def find(self, qualified: str) -> HubTool | None:
        return next((t for t in self.tools() if t.qualified == qualified), None)

    def instructions(self) -> dict[str, str]:
        """Each connected server's self-description, for the system prompt."""
        return {n: c.instructions for n, c in self.connections.items() if c.instructions}

    def needs_approval(self, qualified: str) -> bool:
        """The approval policy, in one place.

        per server  "never"  -> trust everything it offers
                    "always" -> ask before every call
                    "writes" -> ask unless the tool says it's read-only (the default)
        per tool    "always allow" clicked in the UI -> stop asking this session
        Unknown tools are never auto-run.
        """
        tool = self.find(qualified)
        if tool is None:
            return False  # call() will refuse it anyway
        if qualified in self.trusted:
            return False
        policy = self.connections[tool.server].spec.approval
        return policy == "always" or (policy == "writes" and not tool.read_only)

    # --- calling ----------------------------------------------------------------------

    async def call(self, qualified: str, args: dict) -> ToolOutcome:
        """Run one tool. Failures come back as error outcomes, never exceptions:
        a failed call is information Claude can react to, not a reason to crash."""
        tool = self.find(qualified)
        if tool is None:
            msg = f"Tool {qualified!r} is not available (its server may have been switched off)."
            return ToolOutcome([{"type": "text", "text": msg}], True, msg)
        client = self.connections[tool.server].client
        try:
            result = await asyncio.wait_for(client.call_tool(tool.name, args), TOOL_TIMEOUT_S)
        except Exception as exc:  # transport failure: server died, timed out, ...
            msg = f"Calling {tool.name} on {tool.server} failed: {_flatten_error(exc)}"
            return ToolOutcome([{"type": "text", "text": msg}], True, msg, tool.server, tool.name)
        blocks, preview = to_claude_blocks(result)
        return ToolOutcome(blocks, bool(result.is_error), preview, tool.server, tool.name)
