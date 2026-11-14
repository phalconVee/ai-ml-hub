"""The chat agent loop with a scripted fake Claude and the real MCP server."""

import json
from types import SimpleNamespace as NS

from soundcheck import server as srv
from soundcheck import transcript as store
from soundcheck.agent import run_turn
from soundcheck.config import settings
from soundcheck.toolbox import open_toolbox


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


class ScriptedLLM:
    """Plays back canned responses and records what it was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    async def step(self, *, system, messages, tools):
        self.requests.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        return self.responses.pop(0)


REC = {"audio_id": "abc123def456", "file": "show.wav", "duration": "00:45", "language": "en"}


async def test_parallel_tools_then_answer(sample_transcript):
    store.save(sample_transcript)
    llm = ScriptedLLM(
        NS(stop_reason="tool_use", content=[
            text("Let me look."),
            tool_use("t1", "search_transcript", audio_id=REC["audio_id"], query="outage"),
            tool_use("t2", "sentiment_timeline", audio_id=REC["audio_id"]),
        ]),
        NS(stop_reason="end_turn", content=[text("The outage comes up at [00:09].")]),
    )
    events = []
    messages = [{"role": "user", "content": "When do they talk about the outage?"}]
    async with open_toolbox(srv.server) as tb:
        answer = await run_turn(messages, tb, llm, REC, lambda k, d: events.append((k, d["name"])))

    assert answer == "The outage comes up at [00:09]."
    # System prompt carries the recording and the server's own instructions.
    first = llm.requests[0]
    assert "audio_id=abc123def456" in first["system"] and "transcribe_audio" in first["system"]
    assert {t["name"] for t in first["tools"]} >= {"search_transcript", "sentiment_timeline"}
    # Both tool results go back in ONE user message, matched by id.
    results = messages[2]["content"]
    assert messages[2]["role"] == "user" and [r["tool_use_id"] for r in results] == ["t1", "t2"]
    hits = json.loads(results[0]["content"])
    assert hits["hits"][0]["time"] == "00:09" and not results[0]["is_error"]
    assert ("tool_call", "search_transcript") in events and ("tool_result", "sentiment_timeline") in events
    # History is append-only: user, assistant(tool_use), user(results), assistant(answer).
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]


async def test_tool_error_is_fed_back_not_raised():
    llm = ScriptedLLM(
        NS(stop_reason="tool_use", content=[tool_use("t1", "get_transcript", audio_id="ffffffffffff")]),
        NS(stop_reason="end_turn", content=[text("I couldn't find that recording.")]),
    )
    messages = [{"role": "user", "content": "Read it"}]
    async with open_toolbox(srv.server) as tb:
        answer = await run_turn(messages, tb, llm, None)
    result = messages[2]["content"][0]
    assert result["is_error"] and "transcribe_audio first" in result["content"]
    assert "No recording is loaded" in llm.requests[0]["system"]
    assert answer.startswith("I couldn't")


async def test_turn_budget_forces_an_answer(sample_transcript, monkeypatch):
    store.save(sample_transcript)
    monkeypatch.setattr(settings, "max_agent_turns", 2)
    loop = NS(stop_reason="tool_use", content=[tool_use("t", "list_recordings")])
    llm = ScriptedLLM(loop, loop, NS(stop_reason="end_turn", content=[text("Here's what I have.")]))
    messages = [{"role": "user", "content": "Go"}]
    async with open_toolbox(srv.server) as tb:
        answer = await run_turn(messages, tb, llm, REC)
    assert answer == "Here's what I have." and len(llm.requests) == 3
    assert "Tool budget reached" in llm.requests[-1]["messages"][-1]["content"]
