"""The LLM, behind one tiny interface, so the loop doesn't care who answers.

The loop keeps the conversation in ONE neutral format (close to what Ollama
uses natively):

    {"role": "user",      "content": "Find me ..."}
    {"role": "assistant", "content": "...", "agent": "Triage",
                          "tool_calls": [{"id": "call_1", "name": "transfer_to_researcher", "arguments": {}}]}
    {"role": "tool",      "tool_call_id": "call_1", "name": "transfer_to_researcher",
                          "content": "Handed off to Researcher."}

Each backend has one job: translate that list into its API's format, make one
call, and translate the answer back into a `Reply`. That's ~40 lines each, and
it is the only place either SDK is imported.

Why not let the SDKs' own "tool runners" drive the loop? Because handoffs
need to change the system prompt and the tool list *between* calls, which is
exactly the part we want to own and show.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import SETTINGS


# ── The neutral answer every backend returns ────────────────────────────────
@dataclass
class ToolCall:
    id: str                    # ties the result back to the call (Claude requires it)
    name: str                  # which function the model wants run
    arguments: dict[str, Any]  # already-parsed JSON arguments


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class Backend(Protocol):
    """Anything with this one method can drive the swarm (tests use a fake)."""

    name: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> Reply: ...


def new_call_id() -> str:
    """Ollama doesn't give tool calls ids, so we make our own (Claude-safe charset)."""
    return f"call_{uuid.uuid4().hex[:12]}"


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Small models sometimes send arguments as a JSON *string*. Accept both."""
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


# ═════════════════════════════════════════════════════════════════════════════
# Ollama: runs on your machine, no key, the default.
# ═════════════════════════════════════════════════════════════════════════════
class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = SETTINGS.ollama_model, client: Any = None):
        self.model = model
        if client is None:                       # tests pass a fake client instead
            import ollama
            client = ollama.Client(host=SETTINGS.ollama_host)
        self.client = client

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        # 1) neutral → Ollama. The system prompt is just the first message.
        wire: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                wire.append({
                    "role": "assistant",
                    "content": m.get("content", ""),
                    "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                   for c in m.get("tool_calls", [])],
                })
            elif m["role"] == "tool":
                # Ollama matches results to calls by *name*, not id.
                wire.append({"role": "tool", "content": m["content"], "tool_name": m["name"]})
            else:
                wire.append({"role": "user", "content": m["content"]})

        # 2) one call. Tools use the OpenAI-style {"type": "function", ...} wrapper.
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=wire,
            tools=[{"type": "function", "function": t} for t in tools] or None,
        )
        try:
            # Say explicitly whether to think: qwen3-style models think by default.
            response = self.client.chat(**kwargs, think=SETTINGS.ollama_think)
        except Exception as exc:
            # Models without a thinking mode may reject the flag; ask again without it.
            if "think" not in str(exc).lower():
                raise
            response = self.client.chat(**kwargs)

        # 3) Ollama → neutral.
        msg = response.message
        return Reply(
            text=msg.content or "",
            tool_calls=[ToolCall(new_call_id(), c.function.name, parse_arguments(c.function.arguments))
                        for c in (msg.tool_calls or [])],
        )


# ═════════════════════════════════════════════════════════════════════════════
# Claude: optional, needs ANTHROPIC_API_KEY.
# ═════════════════════════════════════════════════════════════════════════════
# If a safety classifier declines a request, this beta re-runs it on a fallback
# model inside the same API call ("default" lets Anthropic pick the fallback).
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeBackend:
    name = "claude"

    def __init__(self, model: str = SETTINGS.claude_model, client: Any = None):
        self.model = model
        if client is None:
            import anthropic
            client = anthropic.Anthropic()       # reads ANTHROPIC_API_KEY from the env
        self.client = client

    @staticmethod
    def to_wire(messages: list[dict]) -> list[dict]:
        """neutral → Claude Messages API.

        Two differences from Ollama matter:
          * tool results are `tool_result` blocks inside a *user* message, tied
            to the call by `tool_use_id`;
          * roles must alternate, so back-to-back user content (several tool
            results, or results followed by a new question) is merged into one
            user message.
        """
        wire: list[dict] = []

        def add(role: str, blocks: list[dict]) -> None:
            if wire and wire[-1]["role"] == role:
                wire[-1]["content"].extend(blocks)   # merge into the previous turn
            else:
                wire.append({"role": role, "content": blocks})

        for m in messages:
            if m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for c in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]})
                # Claude rejects an empty assistant turn; this only happens if a
                # small local model produced nothing earlier in the same chat.
                add("assistant", blocks or [{"type": "text", "text": "(no reply)"}])
            elif m["role"] == "tool":
                add("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}])
            else:
                add("user", [{"type": "text", "text": m["content"]}])
        return wire

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> Reply:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=SETTINGS.claude_max_tokens,
            system=system,                       # changes on every handoff: that's the point
            messages=self.to_wire(messages),
            output_config={"effort": SETTINGS.claude_effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if tools:
            # Claude calls the JSON Schema "input_schema" instead of "parameters".
            kwargs["tools"] = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["parameters"]} for t in tools]
        response = self.client.beta.messages.create(**kwargs)

        # A refusal is an HTTP 200 with stop_reason "refusal"; say so plainly.
        if response.stop_reason == "refusal":
            return Reply(text="(Claude declined to answer this request.)")

        # Thinking blocks are not kept: we only replay text + tool calls, which
        # keeps the transcript backend-neutral (you can switch mid-chat).
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, parse_arguments(b.input))
                 for b in response.content if b.type == "tool_use"]
        return Reply(text=text, tool_calls=calls)


def make_backend(name: str | None = None) -> Backend:
    """Pick a backend by name ("ollama" | "claude"); defaults to SWARM_BACKEND."""
    name = (name or SETTINGS.backend).lower()
    if name == "claude":
        return ClaudeBackend()
    if name == "ollama":
        return OllamaBackend()
    raise ValueError(f"Unknown backend {name!r}: use 'ollama' or 'claude'.")
