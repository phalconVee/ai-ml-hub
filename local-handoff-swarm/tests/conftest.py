"""Shared test helpers: a scripted fake LLM, so every test runs offline and free."""

from __future__ import annotations

import pytest

from swarm import tools
from swarm.backends import Reply, ToolCall


class ScriptedBackend:
    """Plays back a fixed list of Replies and records every request it got.

    The recording is the point: tests can check *which agent's* system prompt
    and tools were sent on each step, which is how we prove a handoff happened.
    """

    name = "fake"

    def __init__(self, replies: list[Reply]):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def complete(self, system, messages, tools):
        self.requests.append({"system": system, "tools": [t["name"] for t in tools],
                              "n_messages": len(messages)})
        if not self.replies:
            raise AssertionError("The fake LLM ran out of scripted replies")
        return self.replies.pop(0)


def call(name: str, **arguments) -> Reply:
    """Shorthand: a reply that is a single tool call."""
    return Reply(tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)])


@pytest.fixture
def fake_search(monkeypatch):
    """Replace the network search with two canned hits; returns the list of queries seen."""
    queries: list[str] = []

    def fake(query, max_results):
        queries.append(query)
        return [
            {"title": "Ollama 1.0 released", "href": "https://example.com/ollama", "body": "Ollama 1.0 adds X."},
            {"title": "Release notes", "href": "https://example.com/notes", "body": "Faster tool calls."},
        ]

    monkeypatch.setattr(tools, "_ddgs_text", fake)
    return queries
