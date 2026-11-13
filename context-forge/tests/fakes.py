"""A stand-in for `anthropic.Anthropic()` so every test runs offline, for free.

`FakeClient` answers `client.beta.messages.create(...)` by looking at the
system prompt to tell which stage is calling (planner, judge, web scout,
synthesizer) and returning a canned response shaped like the real SDK's.
Every request is recorded in `client.calls` so tests can inspect it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

from forge import planner, ranker, synthesizer
from forge.sources import web


def text_block(text, citations=None):
    return NS(type="text", text=text, citations=citations)


def response(blocks, stop="end_turn"):
    return NS(content=blocks, stop_reason=stop)


def web_citation(url, title, quote):
    return NS(type="web_search_result_location", url=url, title=title,
              cited_text=quote, encrypted_index="x")


class FakeClient:
    def __init__(self, plan=None, answer="Context engineering picks what enters the window [S1].",
                 judge=None, web_blocks=None, fail_on=()):
        self.plan = plan or {"doc_query": "context window", "web_query": "context engineering",
                             "paper_query": "context engineering llm", "use_web": True,
                             "use_papers": True, "rationale": "test plan"}
        self.answer, self.judge, self.fail_on = answer, judge, fail_on
        self.web_blocks = web_blocks
        self.calls: list[dict] = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        system = kw["system"]
        stage = {planner.SYSTEM: "plan", ranker.JUDGE_SYSTEM: "judge",
                 web.SYSTEM: "web", synthesizer.SYSTEM: "answer"}[system]
        if stage in self.fail_on:
            raise ConnectionError(f"{stage} is down")
        if stage == "plan":
            return response([text_block(json.dumps(self.plan))])
        if stage == "judge":
            return response([text_block(json.dumps({"grades": self.judge or []}))])
        if stage == "web":
            return response(self.web_blocks or [
                NS(type="server_tool_use", name="web_search", input={"query": "context engineering"}),
                text_block("Notes: ", None),
                text_block("Context engineering curates the window.", [
                    web_citation("https://example.com/a", "Example A",
                                 "Context engineering is the art of filling the context window")]),
            ])
        return response([text_block(self.answer)])
