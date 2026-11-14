"""Each backend's translation to and from its API, with fake SDK clients (no network)."""

from __future__ import annotations

from types import SimpleNamespace as NS

from swarm.backends import ClaudeBackend, OllamaBackend
from swarm.schema import function_to_schema
from swarm.team import transfer_to_writer
from swarm.tools import web_search

TOOLS = [function_to_schema(web_search), function_to_schema(transfer_to_writer)]

# A neutral transcript with two parallel tool calls and then a new user message.
HISTORY = [
    {"role": "user", "content": "find two things"},
    {"role": "assistant", "agent": "Researcher", "content": "",
     "tool_calls": [{"id": "call_a", "name": "web_search", "arguments": {"query": "a"}},
                    {"id": "call_b", "name": "web_search", "arguments": {"query": "b"}}]},
    {"role": "tool", "tool_call_id": "call_a", "name": "web_search", "content": "A!"},
    {"role": "tool", "tool_call_id": "call_b", "name": "web_search", "content": "B!"},
    {"role": "assistant", "agent": "Researcher", "content": "Found A and B.", "tool_calls": []},
    {"role": "user", "content": "thanks"},
]


class FakeOllama:
    def __init__(self, message):
        self.message, self.kwargs = message, None

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return NS(message=self.message)


def test_ollama_wire_format_and_reply_parsing():
    # The model replies with a tool call whose arguments arrive as a JSON *string*.
    msg = NS(content="", tool_calls=[NS(function=NS(name="web_search", arguments='{"query": "x"}'))])
    fake = FakeOllama(msg)
    reply = OllamaBackend(model="qwen3:4b", client=fake).complete("SYS", HISTORY, TOOLS)

    wire = fake.kwargs["messages"]
    assert wire[0] == {"role": "system", "content": "SYS"}           # system prompt goes first
    assert wire[2]["tool_calls"][0] == {"function": {"name": "web_search", "arguments": {"query": "a"}}}
    assert wire[3] == {"role": "tool", "content": "A!", "tool_name": "web_search"}
    assert fake.kwargs["tools"][0]["type"] == "function"
    assert fake.kwargs["model"] == "qwen3:4b"
    assert "context" not in fake.kwargs["tools"][0]["function"]["parameters"]["properties"]

    assert reply.tool_calls[0].name == "web_search"
    assert reply.tool_calls[0].arguments == {"query": "x"}           # string parsed to a dict
    assert reply.tool_calls[0].id.startswith("call_")                # we minted an id


def test_claude_merges_tool_results_into_one_user_turn():
    wire = ClaudeBackend.to_wire(HISTORY)
    assert [m["role"] for m in wire] == ["user", "assistant", "user", "assistant", "user"]
    tool_uses = wire[1]["content"]
    assert [b["type"] for b in tool_uses] == ["tool_use", "tool_use"]   # no empty text block
    results = wire[2]["content"]
    assert [(b["type"], b["tool_use_id"]) for b in results] == [
        ("tool_result", "call_a"), ("tool_result", "call_b")]


def test_claude_merges_results_followed_by_a_new_question():
    # Can happen if the loop hit max_steps right after tool results.
    history = HISTORY[:4] + [{"role": "user", "content": "and c?"}]
    wire = ClaudeBackend.to_wire(history)
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert [b["type"] for b in wire[2]["content"]] == ["tool_result", "tool_result", "text"]


class FakeClaude:
    def __init__(self, response):
        self.kwargs = None
        self.beta = NS(messages=NS(create=self._create))
        self.response = response

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_claude_request_settings_and_reply_parsing():
    response = NS(stop_reason="tool_use", content=[
        NS(type="thinking", thinking=""),
        NS(type="text", text="Passing to the writer."),
        NS(type="tool_use", id="toolu_1", name="transfer_to_writer", input={}),
    ])
    fake = FakeClaude(response)
    reply = ClaudeBackend(model="claude-opus-5-5", client=fake).complete("SYS", HISTORY, TOOLS)

    k = fake.kwargs
    assert k["model"] == "claude-opus-5-5" and k["system"] == "SYS"
    assert k["output_config"] == {"effort": "low"}
    assert k["fallbacks"] == "default" and k["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in k                                 # Opus 5.5: adaptive by default
    assert k["tools"][0]["input_schema"]["required"] == ["query"]

    assert reply.text == "Passing to the writer."            # thinking block ignored
    assert reply.tool_calls[0].id == "toolu_1"


def test_claude_refusal_becomes_a_plain_message():
    fake = FakeClaude(NS(stop_reason="refusal", content=[]))
    reply = ClaudeBackend(client=fake).complete("SYS", HISTORY, TOOLS)
    assert "declined" in reply.text and reply.tool_calls == []


def test_ollama_retries_without_think_flag_for_models_that_reject_it():
    calls = []

    class Picky:
        def chat(self, **kwargs):
            calls.append(kwargs)
            if "think" in kwargs:
                raise RuntimeError('"llama3.2" does not support thinking')
            return NS(message=NS(content="hi", tool_calls=None))

    reply = OllamaBackend(model="llama3.2", client=Picky()).complete("SYS", HISTORY, TOOLS)
    assert reply.text == "hi"
    assert "think" in calls[0] and "think" not in calls[1]
