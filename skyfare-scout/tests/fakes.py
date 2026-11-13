"""A scripted stand-in for `anthropic.Anthropic()`.

`FakeClient` mimics `client.beta.messages.create(**kwargs)`. Each agent has its
own system prompt, so the fake routes on it: give it a dict
{agent name: [responses…]} and every call pops the next response for the
agent that's calling. It also records every request so tests can assert on
what was *sent* (model, effort, schema, tools).
"""

import json
from types import SimpleNamespace as NS

from scout import advisor_agent, parser_agent, sources


def agent_of(system: str) -> str:
    return {parser_agent.SYSTEM: "parser", sources.SCOUT_SYSTEM: "scout",
            sources.EXTRACT_SYSTEM: "extract", advisor_agent.SYSTEM: "advisor"}[system]


class FakeClient:
    def __init__(self, script: dict[str, list]):
        self.script = {k: list(v) for k, v in script.items()}
        self.requests = []  # (agent, kwargs)
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        agent = agent_of(kwargs["system"])
        self.requests.append((agent, {**kwargs, "messages": list(kwargs["messages"])}))
        return self.script[agent].pop(0)

    def calls(self, agent):
        return [kw for a, kw in self.requests if a == agent]


def msg(*blocks, stop_reason="end_turn"):
    return NS(stop_reason=stop_reason, content=list(blocks))


def text(t, citations=None):
    return NS(type="text", text=t, citations=citations)


def as_json(obj, **kw):
    return msg(text(json.dumps(obj)), **kw)


def trip_json(**over):
    base = {"status": "ok", "clarification_question": "", "origin": "LOS",
            "origin_name": "Lagos", "destination": "LON", "destination_name": "London",
            "depart_date": "2026-12-18", "return_date": "2027-01-03", "adults": 2,
            "cabin": "economy", "max_stops": 1, "priority": "cheapest", "budget": None,
            "currency": "USD", "notes": "no overnight layovers"}
    return as_json({**base, **over})


def advice_json(picks, **over):
    base = {"headline": "Take the cheap one.", "picks": picks,
            "tradeoffs": "Cheaper but slower.", "caveats": ["Check baggage."]}
    return as_json({**base, **over})
