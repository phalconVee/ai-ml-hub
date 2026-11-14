"""The orchestration loop and the handoff mechanism, with a scripted fake LLM."""

from __future__ import annotations

from swarm.agent import Agent
from swarm.backends import Reply, ToolCall
from swarm.loop import SwarmState, run_turn
from swarm.schema import function_to_schema
from swarm.team import researcher, summarizer, triage, writer

from .conftest import ScriptedBackend, call


def kinds(events):
    return [e.kind for e in events]


def test_schema_comes_from_signature_and_docstring():
    def lookup(city: str, days: int = 3, context: dict | None = None):
        """Look up the weather."""

    schema = function_to_schema(lookup)
    assert schema["name"] == "lookup"
    assert schema["description"] == "Look up the weather."
    props = schema["parameters"]["properties"]
    assert props == {"city": {"type": "string"}, "days": {"type": "integer"}}  # context is hidden
    assert schema["parameters"]["required"] == ["city"]                       # days has a default


def test_research_then_write_hands_off_twice(fake_search):
    """Triage → Researcher (searches) → Writer (answers), all in one user turn."""
    backend = ScriptedBackend([
        call("transfer_to_researcher"),                 # Triage routes
        call("web_search", query="ollama release"),     # Researcher searches
        call("transfer_to_writer"),                     # Researcher passes to Writer
        Reply(text="Ollama 1.0 is out [1]."),           # Writer answers
    ])
    state = SwarmState(active=triage)
    events = list(run_turn(state, backend, "Research the Ollama release and write a post"))

    assert kinds(events) == ["tool_call", "handoff", "tool_call", "tool_result",
                             "tool_call", "handoff", "message"]
    assert [(e.agent, e.target) for e in events if e.kind == "handoff"] == [
        ("Triage", "Researcher"), ("Researcher", "Writer")]
    assert events[-1].agent == "Writer" and "Ollama 1.0" in events[-1].text
    assert state.active is writer                       # the Writer keeps the conversation

    # Each model call used the ACTIVE agent's prompt and tools: that's the handoff.
    systems = [r["system"] for r in backend.requests]
    assert systems == [triage.instructions, researcher.instructions,
                       researcher.instructions, writer.instructions]
    assert "web_search" in backend.requests[1]["tools"]
    assert "web_search" not in backend.requests[3]["tools"]

    # The shared context collected the sources; the query reached the tool.
    assert fake_search == ["ollama release"]
    assert [s["url"] for s in state.context["sources"]] == [
        "https://example.com/ollama", "https://example.com/notes"]


def test_every_tool_call_gets_a_matching_result_in_the_transcript(fake_search):
    backend = ScriptedBackend([call("transfer_to_researcher"), Reply(text="Hi")])
    state = SwarmState(active=triage)
    list(run_turn(state, backend, "hello"))

    roles = [m["role"] for m in state.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assistant, tool = state.messages[1], state.messages[2]
    assert assistant["agent"] == "Triage"
    assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert "Researcher" in tool["content"]


def test_follow_up_goes_to_the_agent_that_spoke_last():
    backend = ScriptedBackend([call("transfer_to_summarizer"), Reply(text="• point"),
                               Reply(text="• shorter")])
    state = SwarmState(active=triage)
    list(run_turn(state, backend, "Summarize: ..."))
    list(run_turn(state, backend, "Shorter please"))
    # The second turn went straight to the Summarizer: no Triage call in between.
    assert backend.requests[2]["system"] == summarizer.instructions


def test_unknown_tool_and_bad_arguments_become_readable_errors():
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    agent = Agent("Calc", "Add numbers.", [add])
    backend = ScriptedBackend([
        Reply(tool_calls=[ToolCall("c1", "multiply", {}),               # invented tool
                          ToolCall("c2", "add", {"a": 1}),               # missing argument
                          ToolCall("c3", "add", {"a": 1, "b": 2, "x": 9})]),  # extra argument dropped
        Reply(text="3"),
    ])
    state = SwarmState(active=agent)
    events = list(run_turn(state, backend, "1+2"))
    results = [e.text for e in events if e.kind == "tool_result"]
    assert results[0].startswith("Error: Calc has no tool named 'multiply'")
    assert results[1].startswith("Error running add")
    assert results[2] == "3"
    assert events[-1].text == "3"   # the loop recovered and finished


def test_ping_pong_is_stopped_by_max_steps():
    backend = ScriptedBackend([call("transfer_to_writer"), call("transfer_back_to_triage")] * 5)
    state = SwarmState(active=triage)
    events = list(run_turn(state, backend, "loop forever", max_steps=4))
    assert len(backend.requests) == 4
    assert events[-1].kind == "error" and "4 steps" in events[-1].text


def test_web_search_failure_is_reported_to_the_model(monkeypatch):
    from swarm import tools

    def boom(query, max_results):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(tools, "_ddgs_text", boom)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    assert tools.web_search("x", context={}).startswith("Search failed (RuntimeError)")


def test_sources_are_numbered_once_across_searches(fake_search):
    from swarm import tools

    ctx: dict = {}
    tools.web_search("a", ctx)
    second = tools.web_search("b", ctx)        # same two URLs again
    assert len(ctx["sources"]) == 2            # not duplicated
    assert second.startswith("[1] Ollama 1.0 released")
