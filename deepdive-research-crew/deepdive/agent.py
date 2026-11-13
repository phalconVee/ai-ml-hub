"""One agent = one system prompt + (optionally) some MCP tools + a tool-use loop.

Every member of the crew (planner, scouts, analyst, writer) is an instance of
the same `Agent` class. They differ only in their instructions, their effort
level, and whether they get tools. That is the whole trick of "multi-agent":
the agents are cheap, specialised *views* of one model, and the orchestration
in `crew.py` decides who sees what.

The loop below is the classic Claude tool-use loop, written by hand:

    send messages ──► Claude replies
                        │
          stop_reason == "tool_use"? ──no──► done, return its text
                        │ yes
          run every requested tool (in parallel) over MCP
                        │
          append the results as one user message, go round again

plus a budget: after `max_turns` tool rounds the agent must write its answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic

from deepdive.config import FALLBACK_BETA, CrewSettings
from deepdive.mcp_bridge import ToolBox


@dataclass
class Event:
    """A progress update for whoever is watching (the Streamlit UI, a log)."""

    kind: str  # "stage" | "start" | "tool" | "done" | "warning"
    agent: str
    text: str


EventSink = Callable[[Event], None]


def _ignore(_: Event) -> None:
    pass


@dataclass
class Usage:
    """Token counts, summed over every API call an agent makes."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: Any) -> None:
        # Works for both an API `usage` object and another Usage.
        self.input_tokens += getattr(other, "input_tokens", 0) or 0
        self.output_tokens += getattr(other, "output_tokens", 0) or 0


@dataclass
class AgentResult:
    text: str
    stop_reason: str
    turns: int
    usage: Usage = field(default_factory=Usage)


class AgentRefused(RuntimeError):
    """Claude (and its fallback model) declined the request."""


# Appended after the last tool results when the budget runs out.
_BUDGET_NOTE = (
    "Tool budget used up. Do not call any more tools. "
    "Write your final answer now from what you have gathered."
)


class Agent:
    def __init__(
        self,
        name: str,
        role: str,
        system: str,
        client: AsyncAnthropic,
        settings: CrewSettings,
        toolbox: ToolBox | None = None,
        max_turns: int = 0,
        json_schema: dict | None = None,
        on_event: EventSink = _ignore,
    ) -> None:
        self.name = name  # unique label, e.g. "scout-2"
        self.role = role  # key into settings.effort, e.g. "scout"
        self.system = system
        self.client = client
        self.settings = settings
        self.toolbox = toolbox if max_turns > 0 else None  # no budget = no tools
        self.max_turns = max_turns
        self.json_schema = json_schema
        self.on_event = on_event

    # -- one API call --------------------------------------------------------

    async def _call(self, messages: list[dict], final: bool):
        output_config: dict[str, Any] = {"effort": self.settings.effort.get(self.role, "medium")}
        if self.json_schema:
            # Structured output: the reply is guaranteed to be JSON matching
            # this schema. The planner uses it so we can parse its plan safely.
            output_config["format"] = {"type": "json_schema", "schema": self.json_schema}

        kwargs: dict[str, Any] = dict(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=self.system,
            messages=messages,
            # Adaptive thinking: Claude decides how much to think per turn.
            # (On Opus 5.5 thinking can't be switched off; `effort` is the dial.)
            thinking={"type": "adaptive"},
            output_config=output_config,
            # Opt in to server-side refusal fallback (see config.py).
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if self.toolbox:
            # Once a conversation contains tool calls, the tool definitions must
            # stay in every request. On the last turn we keep them but forbid
            # their use with tool_choice "none".
            kwargs["tools"] = self.toolbox.tools
            if final:
                kwargs["tool_choice"] = {"type": "none"}
        return await self.client.beta.messages.create(**kwargs)

    # -- the loop ------------------------------------------------------------

    async def run(self, task: str) -> AgentResult:
        self.on_event(Event("start", self.name, task.splitlines()[0][:160]))
        messages: list[dict] = [{"role": "user", "content": task}]
        usage = Usage()

        turn = 0
        while True:
            final = turn >= self.max_turns
            resp = await self._call(messages, final)
            usage.add(resp.usage)

            # Always append the *whole* reply (thinking blocks included,
            # unchanged). Claude needs its own earlier reasoning to continue a
            # tool loop, and editing it would invalidate it.
            messages.append({"role": "assistant", "content": resp.content})

            if resp.stop_reason == "refusal":
                raise AgentRefused(f"{self.name}: request declined ({getattr(resp, 'stop_details', None)})")

            if resp.stop_reason == "pause_turn":
                # The server paused a long turn; sending the history back resumes it.
                continue

            if resp.stop_reason != "tool_use":
                # "end_turn" (done) or "max_tokens" (cut off: keep what we got).
                if resp.stop_reason == "max_tokens":
                    self.on_event(Event("warning", self.name, "hit max_tokens; answer may be cut short"))
                text = "\n".join(b.text for b in resp.content if b.type == "text").strip()
                self.on_event(Event("done", self.name, f"finished after {turn} tool round(s)"))
                return AgentResult(text, resp.stop_reason, turn, usage)

            # Claude asked for tools. It may ask for several at once (e.g. read
            # three pages): run them concurrently, then send *all* results back
            # in a *single* user message, matched up by tool_use_id.
            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            for b in tool_uses:
                self.on_event(Event("tool", self.name, f"{b.name}({_short_args(b.input)})"))
            outputs = await asyncio.gather(*(self.toolbox.call(self.name, b.name, b.input) for b in tool_uses))

            results: list[dict] = [
                {"type": "tool_result", "tool_use_id": b.id, "content": text or "(empty)", "is_error": is_error}
                for b, (text, is_error) in zip(tool_uses, outputs)
            ]
            turn += 1
            if turn >= self.max_turns:
                results.append({"type": "text", "text": _BUDGET_NOTE})
            messages.append({"role": "user", "content": results})


def _short_args(args: dict) -> str:
    """`{"query": "x"}` -> `query='x'`, trimmed for one-line display."""
    s = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
    return s if len(s) <= 120 else s[:117] + "..."
