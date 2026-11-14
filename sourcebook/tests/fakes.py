"""A scripted stand-in for the Anthropic client, so tests run offline and free.

It records every request and answers from a queue, routing by call type:
structured-output calls (follow-up rewrite, memory summary, podcast script)
get JSON; plain calls (answers) get text blocks with citations.
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS


def text_block(text, citations=None):
    return NS(type="text", text=text, citations=citations)


def cite(result_index, start, end, cited_text):
    return NS(type="search_result_location", search_result_index=result_index,
              start_block_index=start, end_block_index=end, cited_text=cited_text,
              source="", title="")


class FakeClient:
    def __init__(self, answers=(), json_replies=()):
        self.answers = list(answers)          # each: list of content blocks
        self.json_replies = list(json_replies)  # each: dict
        self.requests: list[dict] = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        if "format" in kw.get("output_config", {}):
            payload = self.json_replies.pop(0)
            return NS(stop_reason="end_turn", content=[text_block(json.dumps(payload))])
        return NS(stop_reason="end_turn", content=self.answers.pop(0))

    @property
    def answer_requests(self):
        return [r for r in self.requests if "format" not in r["output_config"]]
