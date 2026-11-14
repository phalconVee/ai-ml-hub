"""Scripted stand-ins for both brains, the Anthropic client and OSM.

`FakeBrain` implements the one-method brain contract (`json(...)`) and routes
on the system prompt: give it {agent: [answers…]} and each call pops the next
answer for the agent that's calling. An answer that is an Exception is raised
instead, to test failure paths. Every call is recorded for assertions.
"""

from types import SimpleNamespace as NS

from stay import advisor_agent, parser_agent, sources


def agent_of(system: str) -> str:
    return {parser_agent.SYSTEM: "parser", sources.SCOUT_SYSTEM: "scout",
            sources.EXTRACT_SYSTEM: "extract", advisor_agent.SYSTEM: "advisor"}[system]


class FakeBrain:
    name = "fake"
    label = "Fake brain"
    can_search_web = False

    def __init__(self, script: dict[str, list]):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []  # (agent, kwargs)

    def json(self, **kwargs):
        agent = agent_of(kwargs["system"])
        self.calls.append((agent, kwargs))
        answer = self.script[agent].pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class FakeAnthropic:
    """Mimics `anthropic.Anthropic().messages.create(**kwargs)`."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.requests = []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def msg(*blocks, stop_reason="end_turn"):
    return NS(stop_reason=stop_reason, content=list(blocks))


def text(t, citations=None):
    return NS(type="text", text=t, citations=citations)


def search_result(*urls):
    return NS(type="web_search_tool_result",
              content=[NS(url=u, title=f"title {u}") for u in urls])


class FakeGeocoder:
    def __init__(self, places: dict):
        self.places = {k.lower(): v for k, v in places.items()}
        self.asked = []

    def locate(self, place):
        self.asked.append(place)
        return self.places.get(place.lower())


def stay_json(**over):
    base = {"status": "ok", "clarification_question": "", "city": "Lisbon",
            "area": "Alfama", "check_in": "2026-12-12", "check_out": "2026-12-15",
            "guests": 2, "rooms": 1, "budget_per_night": 160, "currency": "EUR",
            "must_haves": ["wifi", "air_conditioning"], "nice_to_haves": ["breakfast"],
            "min_rating": 8, "priority": "balanced", "notes": "quiet room"}
    return {**base, **over}


def advice_json(picks, **over):
    base = {"headline": "Stay at the first one.", "picks": picks,
            "tradeoffs": "Closer but pricier.", "caveats": ["Check city tax."]}
    return {**base, **over}


# A trimmed Overpass response: one node, one building outline (way + center),
# one hostel, and one hotel that explicitly has no wifi.
OVERPASS = {"elements": [
    {"type": "node", "id": 1, "lat": 38.7110, "lon": -9.1300,
     "tags": {"tourism": "hotel", "name": "Casa Alfama", "stars": "4",
              "internet_access": "wlan", "air_conditioning": "yes",
              "addr:street": "Rua A", "addr:housenumber": "5",
              "website": "https://casa-alfama.example"}},
    {"type": "way", "id": 2, "center": {"lat": 38.7200, "lon": -9.1400},
     "tags": {"tourism": "hotel", "name": "Baixa Palace", "stars": "5S",
              "wheelchair": "yes"}},
    {"type": "node", "id": 3, "lat": 38.7120, "lon": -9.1310,
     "tags": {"tourism": "hostel", "name": "Backpack Hub", "internet_access": "yes"}},
    {"type": "node", "id": 4, "lat": 38.7300, "lon": -9.1500,
     "tags": {"tourism": "hotel", "name": "Offline Inn", "internet_access": "no"}},
]}
