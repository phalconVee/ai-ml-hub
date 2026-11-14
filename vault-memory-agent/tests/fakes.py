"""A scripted stand-in for the Anthropic client, so tests run offline and free.

Each queued reply is a list of content blocks. A reply containing a tool_use
block gets stop_reason "tool_use", like the real API. Every request is kept
(as a deep copy of its messages) so tests can check exactly what was sent.
"""

from __future__ import annotations

import copy
import itertools
from types import SimpleNamespace as NS

_ids = itertools.count(1)


def text(t):
    return NS(type="text", text=t)


def thinking():
    # Opus 5.5 always thinks; by default the thinking text comes back empty.
    return NS(type="thinking", thinking="", signature="sig")


def tool(name, **inputs):
    return NS(type="tool_use", id=f"toolu_{next(_ids)}", name=name, input=inputs)


class FakeClient:
    def __init__(self, *replies, stop_reason=None):
        self.replies = list(replies)
        self.stop_reason = stop_reason   # force one, e.g. "refusal"
        self.requests: list[dict] = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append({**kw, "messages": copy.deepcopy(kw["messages"])})
        content = self.replies.pop(0)
        stop = self.stop_reason or ("tool_use" if any(b.type == "tool_use" for b in content) else "end_turn")
        return NS(stop_reason=stop, content=content)
