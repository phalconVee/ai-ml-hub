"""The whole crew, end to end, with a scripted fake Claude and fake sources.

Nothing here touches the network or needs an API key, but every real piece runs:
the planner's validation, each scout's tool loop, the MCP server and client,
the source merging and the final report assembly.
"""

import itertools
from types import SimpleNamespace as NS

from scout import sources
from scout.mcp_server import server
from scout.pipeline import research
from scout.schemas import PlatformTask, ResearchPlan

SHARED_URL = "https://example.com/mcp-launch"
_ids = itertools.count()


def tool_use(name, args):
    return NS(type="tool_use", id=f"tu_{next(_ids)}", name=name, input=args)


def reply(*blocks, stop="tool_use"):
    return NS(content=list(blocks), stop_reason=stop)


def finding(url, claim):
    return {"claim": claim, "source_url": url, "source_title": "t", "kind": "fact"}


class FakeLLM:
    """Plays Claude's part from a script, and records what it was sent."""

    def __init__(self):
        self.turns: dict[str, int] = {}
        self.seen_tool_results: dict[str, list] = {}
        self.synth_prompt = ""

    async def structured(self, *, schema, prompt, **_):
        assert schema is ResearchPlan
        return ResearchPlan(restated_question="What is MCP?", tasks=[
            PlatformTask(platform="web", brief="official info", queries=["mcp"]),
            PlatformTask(platform="hackernews", brief="community take", queries=["mcp"]),
            PlatformTask(platform="tiktok", brief="not a real platform", queries=["mcp"]),  # must be dropped
        ])

    async def step(self, *, system, messages, tools, **_):
        platform = "web" if "web scout" in system else "hackernews"
        turn = self.turns.get(platform, 0)
        self.turns[platform] = turn + 1
        names = {t["name"] for t in tools}

        if turn == 0:
            # Tool scoping: each scout sees only its own platform's search tool.
            assert f"search_{platform}" in names and "submit_findings" in names
            assert ("search_web" in names) == (platform == "web")
            return reply(tool_use(f"search_{platform}", {"query": "mcp"}))

        # Record the tool_result the loop sent back after turn 0.
        self.seen_tool_results.setdefault(platform, []).append(messages[-1]["content"])

        if platform == "hackernews" and turn == 1:
            # First submission is invalid (findings missing): the loop must
            # return an error and let us retry rather than crash.
            return reply(tool_use("submit_findings", {"summary": "oops"}))
        return reply(tool_use("submit_findings", {
            "summary": f"{platform} summary",
            "findings": [finding(SHARED_URL, f"{platform} says MCP launched"),
                         finding(f"https://{platform}.example/only", f"{platform}-only claim")],
        }))

    async def stream_text(self, *, prompt, on_text=None, **_):
        self.synth_prompt = prompt
        for chunk in ["MCP is a protocol ", "[1][2]."]:
            on_text and on_text(chunk)
        return "MCP is a protocol [1][2]."


async def test_full_crew_offline(monkeypatch):
    async def fake(query, n=8):
        return [{"title": "MCP launch", "url": SHARED_URL, "snippet": "launched", "published": "", "meta": ""}]
    monkeypatch.setattr(sources, "web", fake)
    monkeypatch.setattr(sources, "hackernews", fake)

    llm, events, chunks = FakeLLM(), [], []
    result = await research(
        "what is mcp", llm=llm, mcp_server=server,
        on_event=lambda k, d: events.append(k), on_text=chunks.append,
    )

    # Planner: the made-up platform was filtered out.
    assert [t.platform for t in result.plan.tasks] == ["web", "hackernews"]

    # Scouts: the MCP search result actually reached the model as a tool_result.
    web_result = llm.seen_tool_results["web"][0][0]
    assert web_result["type"] == "tool_result" and SHARED_URL in web_result["content"]

    # The invalid hackernews submission got an error back, then a retry succeeded.
    retry = llm.seen_tool_results["hackernews"][1][0]
    assert retry["is_error"] and "invalid" in retry["content"]
    assert all(len(r.findings) == 2 for r in result.reports)

    # Merging: the URL both scouts found is ONE source tagged with both platforms.
    shared = [s for s in result.sources if s["url"] == SHARED_URL]
    assert len(shared) == 1 and shared[0]["platforms"] == ["web", "hackernews"]
    assert len(result.sources) == 3

    # Report: streamed text + code-built sources list.
    assert "".join(chunks) == "MCP is a protocol [1][2]."
    assert "## Sources" in result.report_md and SHARED_URL in result.report_md
    assert "[1]" in llm.synth_prompt
    assert events.count("scout_done") == 2 and "synthesizing" in events


async def test_scout_that_never_submits_returns_honest_empty_report(monkeypatch):
    class Chatty(FakeLLM):
        async def step(self, **_):
            return reply(NS(type="text", text="I think I'm done."), stop="end_turn")

    async def fake(query, n=8):
        return []
    monkeypatch.setattr(sources, "web", fake)
    result = await research("q", llm=Chatty(), mcp_server=server)
    assert all(r.findings == [] and r.gaps for r in result.reports)
    assert "## Sources" in result.report_md
