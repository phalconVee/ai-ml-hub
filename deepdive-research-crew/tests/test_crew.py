"""The whole crew, end to end, with a scripted fake Claude and the real MCP tools.

No API key and no network: `FakeClaude` stands in for `AsyncAnthropic` and
plays each role from a script, while the tools are the real `tools_server`
running in-process with the offline "fake" search backend. What we check is
the plumbing: who calls what, what each agent gets to see, and that tool
results flow back to Claude in the shape the API requires.
"""

import json
from types import SimpleNamespace as NS

import pytest

from deepdive.agent import Agent
from deepdive.config import FALLBACK_BETA, CrewSettings
from deepdive.crew import ANALYST_SYSTEM, PLANNER_SYSTEM, SCOUT_SYSTEM, WRITER_SYSTEM, run_research
from deepdive.mcp_bridge import open_toolbox
from deepdive.tools_server import server as tools_server


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("DEEPDIVE_SEARCH_BACKEND", "fake")


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def reply(*blocks, stop="end_turn"):
    return NS(content=list(blocks), stop_reason=stop, usage=NS(input_tokens=100, output_tokens=10))


class FakeClaude:
    """Duck-types `client.beta.messages.create` and records every request."""

    def __init__(self, fail_scout: str | None = None):
        self.requests: list[dict] = []
        self.fail_scout = fail_scout
        self.beta = NS(messages=NS(create=self.create))

    async def create(self, **kw):
        # Snapshot the message list: the agent keeps appending to the same list.
        self.requests.append(dict(kw, messages=list(kw["messages"])))
        system, msgs = kw["system"], kw["messages"]
        first = msgs[0]["content"]

        if system == PLANNER_SYSTEM:
            plan = {"angles": [
                {"title": "Big firms", "question": "How fast are big firms adopting?", "search_queries": ["adoption large firms"]},
                {"title": "Small firms", "question": "And small firms?", "search_queries": ["adoption small firms"]},
            ]}
            return reply(text(json.dumps(plan)))

        if system == SCOUT_SYSTEM:
            if self.fail_scout and self.fail_scout in first:
                raise RuntimeError("simulated scout crash")
            if len(msgs) == 1:  # first turn: search and read two pages at once
                return reply(
                    tool_use("t1", "web_search", query="adoption"),
                    tool_use("t2", "read_page", url="https://alpha.example.test/report"),
                    stop="tool_use",
                )
            return reply(text("## Findings\n- Adoption rose 42% (source: https://alpha.example.test/report)"))

        if system == ANALYST_SYSTEM:
            return reply(text("## Established (high confidence)\n- 42% rise among large firms"))

        if system == WRITER_SYSTEM:
            return reply(text("# Report\nAdoption rose 42% [1].\n\n## References\n[1] Alpha - https://alpha.example.test/report"))

        raise AssertionError("unknown agent")


def in_memory_toolbox():
    # The real MCP server object, connected in-process (no subprocess needed).
    return open_toolbox(tools_server)


async def test_full_run_produces_a_cited_report():
    claude = FakeClaude()
    events = []
    out = await run_research(
        "How fast is adoption growing?",
        CrewSettings(num_angles=2),
        client=claude,
        toolbox_factory=in_memory_toolbox,
        on_event=events.append,
    )

    assert [a.title for a in out.angles] == ["Big firms", "Small firms"]
    assert set(out.notes) == {"Big firms", "Small firms"}
    assert out.report.startswith("# Report") and "[1]" in out.report
    assert out.sources == ["https://alpha.example.test/report"]
    # planner 1 + scouts 2x2 + analyst 1 + writer 1 = 7 API calls, all counted.
    assert len(claude.requests) == 7
    assert out.usage.input_tokens == 700
    assert [e.text for e in events if e.kind == "stage"][-1] == "Done"


async def test_every_request_uses_adaptive_thinking_and_refusal_fallback():
    claude = FakeClaude()
    await run_research("q", CrewSettings(num_angles=2), client=claude, toolbox_factory=in_memory_toolbox)
    for kw in claude.requests:
        assert kw["thinking"] == {"type": "adaptive"}
        assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
        assert kw["output_config"]["effort"] in {"low", "medium", "high"}
    planner_kw = claude.requests[0]
    assert planner_kw["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in planner_kw  # the planner and writer never get tools


async def test_parallel_tool_results_go_back_in_one_message_matched_by_id():
    claude = FakeClaude()
    await run_research("q", CrewSettings(num_angles=2), client=claude, toolbox_factory=in_memory_toolbox)
    # The scout's second request carries the results of its two tool calls.
    second_scout_turn = next(kw for kw in claude.requests if kw["system"] == SCOUT_SYSTEM and len(kw["messages"]) == 3)
    results = second_scout_turn["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert all(r["type"] == "tool_result" and not r["is_error"] for r in results)
    assert "42%" in results[1]["content"]


async def test_tool_budget_forces_a_final_answer():
    claude = FakeClaude()
    settings = CrewSettings()
    async with in_memory_toolbox() as box:
        scout = Agent("scout-1", "scout", SCOUT_SYSTEM, claude, settings, toolbox=box, max_turns=1)
        res = await scout.run("Your angle: anything")
    assert res.turns == 1
    last = claude.requests[-1]
    assert last["tool_choice"] == {"type": "none"}  # tools still defined, but off
    assert last["messages"][-1]["content"][-1]["text"].startswith("Tool budget used up")


async def test_one_failed_scout_does_not_sink_the_run():
    claude = FakeClaude(fail_scout="Small firms")
    out = await run_research("q", CrewSettings(num_angles=2), client=claude, toolbox_factory=in_memory_toolbox)
    assert "scout failed" in out.notes["Small firms"]
    assert out.report.startswith("# Report")
