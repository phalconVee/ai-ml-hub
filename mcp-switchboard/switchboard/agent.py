"""The agent loop, written so it can *pause* for a human and pick up again.

A normal tool loop is:

    ask Claude -> Claude wants tools -> run them -> send results -> ask again ... -> text answer

Ours has one extra exit. If Claude asks for a tool that needs approval (a
write, by default), we stop and hand the pending calls to the UI. Streamlit
can't block inside a function waiting for a button click: every click re-runs
the whole script. So "pausing" here means *returning*, and "resuming" means
calling `run_turn()` again with the user's decisions.

That works because the conversation itself is the state. If the last message
is an assistant turn containing tool_use blocks with no results yet, there
is unfinished business; `run_turn()` looks at that first, every time. No
hidden variables, nothing to keep in sync: the message list says where we are.

    messages[-1] is ...                        run_turn() does ...
    user text / tool results                   ask Claude for the next step
    assistant with tool_use, all decided       run approved calls, refuse denied ones
    assistant with tool_use, some undecided    return "needs_approval" (pause)
    assistant with only text                   return "done"
"""

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Literal

from switchboard.hub import Switchboard, ToolOutcome
from switchboard.llm import LLM, block_to_dict
from switchboard.settings import MAX_AGENT_STEPS

SYSTEM = """You are the assistant in MCP Switchboard, a chat app connected to several MCP servers.
Each tool name is "<server>__<tool>", and its description starts with [server].
Use tools whenever they give a better answer than memory: real files, real time, real web pages.
Call independent tools in parallel. Some calls need the user's approval; if one is denied, don't retry it -
say what you would have done and ask how to proceed. Be concise and show key results."""

# A built-in, client-side tool that is always present. Two jobs: it lets Claude
# answer "what can you do?" accurately, and it means `tools` is never empty.
# (The API rejects a request whose history has tool_use blocks but whose
# `tools` list is empty, which would happen if you switched every server off.)
LIST_SERVERS = {
    "name": "switchboard__list_servers",
    "description": "List the MCP servers currently switched on and the tools each one offers.",
    "input_schema": {"type": "object", "properties": {}},
}


@dataclass
class PendingCall:
    """A tool call waiting for the user's yes/no."""

    id: str
    qualified: str
    server: str
    tool: str
    args: dict
    destructive: bool


@dataclass
class TurnResult:
    status: Literal["done", "needs_approval", "step_limit", "refused"]
    pending: list[PendingCall] = field(default_factory=list)


# Progress events for the UI: on_event("tool_start" | "tool_end" | "thinking", data)
EventFn = Callable[[str, dict], None]


def build_system(board: Switchboard) -> str:
    notes = board.instructions()
    if not notes:
        return SYSTEM
    # Servers describe themselves in their `instructions`; pass them along.
    lines = "\n".join(f"- {name}: {text}" for name, text in notes.items())
    return f"{SYSTEM}\n\nNotes from the connected servers:\n{lines}"


def _tool_uses(message: dict) -> list[dict]:
    if message["role"] != "assistant" or isinstance(message["content"], str):
        return []
    return [b for b in message["content"] if b.get("type") == "tool_use"]


def _list_servers(board: Switchboard) -> ToolOutcome:
    lines = []
    for name, conn in board.connections.items():
        tools = ", ".join(t.name + ("" if t.read_only else " (writes)") for t in conn.tools)
        lines.append(f"{name} [{conn.spec.transport}]: {tools or 'no tools'}")
    text = "\n".join(lines) or "No servers are switched on."
    return ToolOutcome([{"type": "text", "text": text}], False, text, "switchboard", "list_servers")


async def _execute(board: Switchboard, use: dict, approved: bool | None, on_event: EventFn) -> dict:
    """Run (or refuse) one tool_use and build its tool_result block."""
    name, args = use["name"], use.get("input") or {}
    if name == LIST_SERVERS["name"]:
        outcome = _list_servers(board)
    elif approved is False:
        msg = "The user denied this tool call. Do not retry it; ask the user how to proceed."
        outcome = ToolOutcome([{"type": "text", "text": msg}], True, msg)
    else:
        on_event("tool_start", {"id": use["id"], "name": name, "args": args})
        outcome = await board.call(name, args)
    on_event("tool_end", {"id": use["id"], "name": name, "outcome": outcome})
    return {"type": "tool_result", "tool_use_id": use["id"], "content": outcome.content, "is_error": outcome.is_error}


async def run_turn(
    llm: LLM,
    board: Switchboard,
    messages: list[dict],
    *,
    model: str,
    effort: str,
    decisions: dict[str, bool] | None = None,
    on_event: EventFn = lambda kind, data: None,
) -> TurnResult:
    """Advance the conversation until Claude answers, or a call needs approval.

    `messages` is modified in place (that's the state). `decisions` maps
    tool_use ids to True (approve) / False (deny), filled in by the UI.
    """
    decisions = decisions or {}
    for _ in range(MAX_AGENT_STEPS):
        uses = _tool_uses(messages[-1])

        if messages[-1]["role"] == "assistant" and not uses:
            return TurnResult("done")  # a plain text answer: the turn is over

        if uses:
            # Unfinished business: Claude asked for tools. Any still awaiting a human?
            pending = []
            for u in uses:
                if board.needs_approval(u["name"]) and u["id"] not in decisions:
                    tool = board.find(u["name"])
                    pending.append(PendingCall(u["id"], u["name"], tool.server, tool.name,
                                               u.get("input") or {}, tool.destructive))
            if pending:
                return TurnResult("needs_approval", pending)  # pause; the UI calls us again

            # Everything decided: run all calls concurrently (Claude often asks for
            # several at once) and return ALL results in ONE user message, in order.
            results = await asyncio.gather(*(
                _execute(board, u, decisions.get(u["id"]), on_event) for u in uses
            ))
            messages.append({"role": "user", "content": list(results)})

        # The last message is now the user's (text or tool results): ask Claude.
        on_event("thinking", {})
        tools = [t.for_claude() for t in board.tools()] + [LIST_SERVERS]
        response = await llm.step(model=model, effort=effort, system=build_system(board),
                                  messages=messages, tools=tools)
        # Store the assistant turn exactly as returned (thinking blocks included).
        messages.append({"role": "assistant", "content": [block_to_dict(b) for b in response.content]})
        if response.stop_reason == "refusal":
            if not messages[-1]["content"]:
                messages.pop()  # an empty assistant turn can't be sent back next time
            return TurnResult("refused")
        # Loop: the top of the loop decides whether this answer ends the turn.

    return TurnResult("step_limit")


def close_dangling(messages: list[dict]) -> None:
    """Before a new user message: answer any tool_use left without a result.

    Happens if the step limit hit mid-loop, or the user typed instead of
    approving. The API requires every tool_use to be followed by its
    tool_result, so we record those calls as skipped.
    """
    if messages and (uses := _tool_uses(messages[-1])):
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": u["id"], "is_error": True,
             "content": [{"type": "text", "text": "Skipped: the user moved on without running this call."}]}
            for u in uses
        ]})
