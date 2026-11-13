"""A scripted stand-in for Claude, plus helpers that build response blocks.

`FakeClient` mimics `client.beta.messages.create(...)`: each call pops the
next canned response and records the request, so tests can assert both what
the desk *did* and what it *sent*. The blocks are SimpleNamespaces with the
same attribute names as the SDK's types, which is all our code reads.
"""

import json
from types import SimpleNamespace as NS


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        # Snapshot `messages`: the reporter appends to the same list object.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def msg(stop_reason, *blocks):
    return NS(stop_reason=stop_reason, content=list(blocks))


def text(t, citations=None):
    return NS(type="text", text=t, citations=citations)


def cite(url, quote, title=None):
    return NS(type="web_search_result_location", url=url, cited_text=quote, title=title)


def search(query, *results):
    """A web_search call plus its results. results = (url, title, page_age) tuples."""
    return [
        NS(type="server_tool_use", name="web_search", input={"query": query}),
        NS(type="web_search_tool_result",
           content=[NS(type="web_search_result", url=u, title=t, page_age=a) for u, t, a in results]),
    ]


def fetch(url, title):
    return [
        NS(type="server_tool_use", name="web_fetch", input={"url": url}),
        NS(type="web_fetch_tool_result",
           content=NS(type="web_fetch_result", url=url, content=NS(type="document", title=title))),
    ]


def search_error(query):
    """Server-tool errors arrive as content, not exceptions."""
    return [
        NS(type="server_tool_use", name="web_search", input={"query": query}),
        NS(type="web_search_tool_result",
           content=NS(type="web_search_tool_result_error", error_code="max_uses_exceeded")),
    ]


def json_reply(obj):
    return msg("end_turn", text(json.dumps(obj)))


def article(headline="Rates held", dek="The bank paused.", body="The bank held rates at 4.25% [S1]."):
    return json_reply({"headline": headline, "dek": dek, "body": body})


def verdict(v, *notes):
    return json_reply({"verdict": v, "notes": list(notes)})


# A small, realistic reporter turn: two searches (one paused), one fetch, notes with citations.
BANK = "https://www.centralbank.example/news/rates"
WIRE = "https://wire.example/markets/bank-holds"
BLOG = "https://blog.example/opinion"


def reporter_turns():
    first = msg("pause_turn",
                *search("central bank rate decision", (WIRE, "Bank holds rates", "1 day ago"),
                        (BLOG, "My take", "2 days ago")))
    second = msg("end_turn",
                 *fetch(BANK, "Monetary policy statement"),
                 text("- The bank held its key rate at 4.25% on Thursday.",
                      citations=[cite(BANK, "The key rate remains at 4.25%.")]),
                 text("\n- Two members voted for a cut.",
                      citations=[cite(WIRE, "two of nine members voted to cut")]))
    return [first, second]
