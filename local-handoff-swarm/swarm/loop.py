"""The orchestration loop: the whole "framework", in about 80 lines.

One user message goes in. Then, until an agent answers in plain text:

    ┌─► ask the ACTIVE agent's model (its instructions + its tools + the shared history)
    │        │
    │        ├── plain text, no tool calls  → done: that's the answer, stop
    │        │
    │        └── tool calls → run each one:
    │               returns a str    → append it as a tool result
    │               returns an Agent → HANDOFF: that agent becomes the active one
    └────────────────── loop (at most SETTINGS.max_steps model calls)

Three things are shared by every agent, and that is what makes it a *team*:
  * `messages` — one transcript. The writer sees what the researcher found.
  * `context`  — a dict tools can read/write (e.g. the list of sources found).
  * `active`   — who's holding the conversation. It stays put after the turn,
                 so a follow-up question goes to the same specialist.

The loop is a generator: it `yield`s an Event at every step, so the Streamlit
UI can draw "🔀 Triage → Researcher" the moment it happens, and tests can
assert the exact sequence of events.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Literal

from .agent import Agent
from .backends import Backend, ToolCall
from .config import SETTINGS
from .schema import CONTEXT_PARAM, function_to_schema


@dataclass
class SwarmState:
    """Everything that survives between user messages (Streamlit keeps it in session_state)."""

    active: Agent                                   # who answers next
    messages: list[dict] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Event:
    """One thing that happened, for the UI and the tests."""

    kind: Literal["tool_call", "tool_result", "handoff", "message", "error"]
    agent: str                      # the agent that was active when it happened
    text: str = ""                  # message text, tool result, or error
    tool: str = ""                  # tool name (tool_call / tool_result)
    arguments: dict = field(default_factory=dict)
    target: str = ""                # handoff destination


def run_tool(agent: Agent, call: ToolCall, context: dict) -> Any:
    """Run one tool call safely. Errors become text the model can read and recover from."""
    fn = agent.tool(call.name)
    if fn is None:
        # Small models sometimes invent tools. Tell them, don't crash.
        known = ", ".join(f.__name__ for f in agent.functions)
        return f"Error: {agent.name} has no tool named {call.name!r}. Available: {known}."

    params = inspect.signature(fn).parameters
    # Keep only arguments the function accepts (models sometimes add extras)…
    args = {k: v for k, v in call.arguments.items() if k in params and k != CONTEXT_PARAM}
    # …and hand the shared context dict to tools that ask for it.
    if CONTEXT_PARAM in params:
        args[CONTEXT_PARAM] = context
    try:
        return fn(**args)
    except Exception as exc:   # a tool failure is information for the model, not a crash
        return f"Error running {call.name}: {exc}"


def run_turn(state: SwarmState, backend: Backend, user_text: str,
             max_steps: int = SETTINGS.max_steps) -> Iterator[Event]:
    """Handle one user message, yielding Events. Mutates `state` in place."""
    state.messages.append({"role": "user", "content": user_text})

    for _ in range(max_steps):
        agent = state.active  # the agent making THIS call (may change below)

        # The handoff mechanism lives in these two arguments: whoever is active
        # supplies the system prompt and the tool list for this one call.
        reply = backend.complete(
            system=agent.instructions,
            messages=state.messages,
            tools=[function_to_schema(f) for f in agent.functions],
        )

        # Record the reply in the shared transcript, tagged with who said it.
        state.messages.append({
            "role": "assistant",
            "agent": agent.name,
            "content": reply.text,
            "tool_calls": [asdict(c) for c in reply.tool_calls],
        })
        if reply.text:
            yield Event("message", agent.name, text=reply.text)

        if not reply.tool_calls:
            return  # plain answer → the turn is over, control goes back to the user

        for call in reply.tool_calls:
            yield Event("tool_call", agent.name, tool=call.name, arguments=call.arguments)
            result = run_tool(agent, call, state.context)

            if isinstance(result, Agent):
                # ── THE HANDOFF ──────────────────────────────────────────────
                # Swap the active agent. The next loop iteration calls the model
                # with the new agent's instructions and tools, over the same
                # history. Nothing else changes.
                state.active = result
                content = f"Handed off to {result.name}. {result.name} now has the conversation."
                yield Event("handoff", agent.name, target=result.name)
            else:
                content = str(result)
                yield Event("tool_result", agent.name, tool=call.name, text=content)

            # Every call gets a result, even handoffs: both APIs require each
            # tool call to be answered before the next model call.
            state.messages.append({"role": "tool", "tool_call_id": call.id,
                                   "name": call.name, "content": content})

    # Only reached if the agents never settled on a plain answer.
    yield Event("error", state.active.name,
                text=f"Stopped after {max_steps} steps without a final answer. Try rephrasing.")
