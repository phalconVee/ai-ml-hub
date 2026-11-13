"""The LLM agents, with Claude replaced by a script (no API key needed)."""

from datetime import date
from types import SimpleNamespace as NS

import pytest

from fakes import FakeClient, advice_json, as_json, msg, text, trip_json
from scout.advisor_agent import Advice, verify
from scout.analyst import analyse
from scout.config import SETTINGS
from scout.llm import ClaudeRefused
from scout.models import FlightOption, Leg, TripRequest
from scout.parser_agent import parse_trip, validate
from scout.sources import (DemoSource, SerpApiSource, WebFareSource, parse_serpapi,
                           pick_source, stops_code)

TODAY = date(2026, 11, 13)


# ── Parser ───────────────────────────────────────────────────────────────────
def test_parser_builds_trip_and_sends_schema_and_today():
    client = FakeClient({"parser": [trip_json()]})
    result = parse_trip(client, "Lagos to London mid Dec, back after NY, 2 of us", TODAY)
    assert result.trip.origin == "LOS" and result.trip.adults == 2
    assert [l.name for l in result.trip.legs()] == ["outbound", "return"]
    req = client.calls("parser")[0]
    assert req["model"] == SETTINGS.model
    assert req["output_config"]["effort"] == SETTINGS.parser_effort
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert "TODAY: 2026-11-13" in req["messages"][0]["content"]
    assert req["fallbacks"] == "default" and SETTINGS.fallback_beta in req["betas"]


def test_parser_passes_through_clarification():
    client = FakeClient({"parser": [trip_json(status="needs_clarification",
                                              clarification_question="Where to?")]})
    assert parse_trip(client, "a holiday", TODAY).question == "Where to?"


def test_parser_rejects_invalid_json_values():
    client = FakeClient({"parser": [trip_json(destination="London",
                                              depart_date="2026-01-02")]})
    result = parse_trip(client, "x", TODAY)
    assert result.trip is None
    assert any("3-letter" in p for p in result.problems)
    assert any("past" in p for p in result.problems)


def test_validate_return_before_departure():
    t = TripRequest("LOS", "LHR", "2026-12-18", "2026-12-10")
    assert validate(t, TODAY) == ["return date is before the departure date"]


def test_refusal_is_raised_not_parsed():
    client = FakeClient({"parser": [msg(stop_reason="refusal")]})
    with pytest.raises(ClaudeRefused):
        parse_trip(client, "x", TODAY)


# ── Sources ──────────────────────────────────────────────────────────────────
SERP = {
    "search_metadata": {"google_flights_url": "https://www.google.com/travel/flights?x=1"},
    "best_flights": [{
        "flights": [
            {"departure_airport": {"id": "LOS", "time": "2026-12-18 22:10"},
             "arrival_airport": {"id": "CDG", "time": "2026-12-19 05:40"},
             "airline": "Air France", "flight_number": "AF 149", "duration": 390},
            {"departure_airport": {"id": "CDG", "time": "2026-12-19 07:15"},
             "arrival_airport": {"id": "LHR", "time": "2026-12-19 07:40"},
             "airline": "Air France", "flight_number": "AF 1680", "duration": 85}],
        "layovers": [{"id": "CDG", "duration": 95}],
        "total_duration": 570, "price": 612}],
    "other_flights": [{
        "flights": [{"departure_airport": {"id": "LOS", "time": "2026-12-18 23:00"},
                     "arrival_airport": {"id": "LHR", "time": "2026-12-19 05:30"},
                     "airline": "British Airways", "flight_number": "BA 74",
                     "duration": 390}],
        "layovers": [], "total_duration": 390, "price": 890}],
}


def test_parse_serpapi_maps_fields():
    t = TripRequest("LOS", "LHR", "2026-12-18", None)
    opts = parse_serpapi(SERP, t.legs()[0], t)
    assert [o.id for o in opts] == ["OUT-1", "OUT-2"]
    af = opts[0]
    assert af.airline == "Air France" and af.flight_numbers == ["AF 149", "AF 1680"]
    assert af.stops == 1 and af.layovers == ["CDG 1h 35m"] and af.duration_minutes == 570
    assert af.depart_time == "2026-12-18 22:10" and af.arrive_time == "2026-12-19 07:40"
    assert opts[1].stops == 0 and opts[1].price == 890


def test_serpapi_request_params():
    seen = {}
    src = SerpApiSource("k", fetch=lambda url: seen.setdefault("url", url) and SERP)
    t = TripRequest("LOS", "LHR", "2026-12-18", None, cabin="business", max_stops=0)
    out = src.search(t.legs()[0], t)
    for part in ("engine=google_flights", "type=2", "departure_id=LOS",
                 "travel_class=3", "stops=1", "outbound_date=2026-12-18"):
        assert part in seen["url"]
    assert len(out.options) == 2
    assert stops_code(None) == 0 and stops_code(1) == 2 and stops_code(5) == 3


def test_serpapi_error_body_becomes_a_note():
    src = SerpApiSource("k", fetch=lambda url: {"error": "Invalid API key."})
    t = TripRequest("LOS", "LHR", "2026-12-18", None)
    out = src.search(t.legs()[0], t)
    assert out.options == [] and "Invalid API key" in out.notes[0]


def test_web_source_keeps_only_rows_with_seen_urls_and_resumes_pause():
    real = "https://www.example-air.com/los-lhr"
    paused = msg(NS(type="server_tool_use", name="web_search", input={"query": "LOS LHR"}),
                 NS(type="web_search_tool_result",
                    content=[NS(url=real + "/", title="Example Air LOS-LHR")]),
                 stop_reason="pause_turn")
    finished = msg(text(f"Example Air EA1 $540, 1 stop, {real}"))
    rows = {"options": [
        {"airline": "Example Air", "flight_numbers": ["EA1"], "depart_time": "",
         "arrive_time": "", "duration_minutes": 600, "stops": 1, "layovers": [],
         "price": 540, "currency": "USD", "source_url": real, "caveat": ""},
        {"airline": "Ghost Air", "flight_numbers": [], "depart_time": "",
         "arrive_time": "", "duration_minutes": None, "stops": 0, "layovers": [],
         "price": 99, "currency": "USD", "source_url": "https://made-up.example/deal",
         "caveat": ""}]}
    client = FakeClient({"scout": [paused, finished], "extract": [as_json(rows)]})
    t = TripRequest("LOS", "LHR", "2026-12-18", None)
    out = WebFareSource(client).search(t.legs()[0], t)

    assert [o.airline for o in out.options] == ["Example Air"]      # Ghost Air dropped
    assert out.options[0].estimated and out.options[0].id == "OUT-1"
    assert any("dropped 1" in n for n in out.notes)
    scout_calls = client.calls("scout")
    assert len(scout_calls) == 2                                     # resumed once
    assert scout_calls[1]["messages"][-1]["role"] == "assistant"     # no extra "continue"
    assert scout_calls[0]["tools"][0]["type"] == "web_search_20260209"


def test_demo_source_is_deterministic_and_respects_max_stops():
    t = TripRequest("LOS", "LHR", "2026-12-18", None, max_stops=0)
    a = DemoSource().search(t.legs()[0], t).options
    b = DemoSource().search(t.legs()[0], t).options
    assert [(o.price, o.duration_minutes) for o in a] == [(o.price, o.duration_minutes) for o in b]
    assert all(o.stops == 0 for o in a)


def test_pick_source(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    assert pick_source("auto", None).name == "web"
    assert pick_source("demo", None).name == "demo"
    with pytest.raises(ValueError):
        pick_source("serpapi", None)
    monkeypatch.setenv("SERPAPI_API_KEY", "abc")
    assert pick_source("auto", None).name == "serpapi"


# ── Advisor verification ─────────────────────────────────────────────────────
def _analysis():
    t = TripRequest("LOS", "LHR", "2026-12-18", "2027-01-03")
    legs = {leg.name: DemoSource().search(leg, t).options for leg in t.legs()}
    return t, analyse(t, legs)


def test_verify_drops_unknown_ids_and_fills_missing_leg():
    _, analysis = _analysis()
    advice = Advice("h", [{"option_id": "OUT-2", "role": "best_overall", "reason": "r"},
                          {"option_id": "OUT-99", "role": "cheapest", "reason": "made up"},
                          {"option_id": "OUT-2", "role": "fastest", "reason": "dup"}],
                    "t", [])
    verify(advice, analysis)
    ids = [p["option_id"] for p in advice.picks]
    assert "OUT-99" not in ids and ids.count("OUT-2") == 1
    return_top = analysis.legs[1].options[0].id
    assert return_top in ids                                    # added for the empty leg
    assert any("OUT-99" in c for c in advice.corrections)
