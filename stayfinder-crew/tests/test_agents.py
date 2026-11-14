"""Parser, sources and advisor, each on its own with fakes."""

from datetime import date

import pytest

from stay import advisor_agent, sources
from stay.analyst import analyse
from stay.geo import Geocoder, km
from stay.llm import BrainError, ClaudeBrain
from stay.models import HotelOption
from stay.parser_agent import parse_stay, validate

from .fakes import (OVERPASS, FakeAnthropic, FakeBrain, FakeGeocoder, advice_json, msg,
                    search_result, stay_json, text)

TODAY = date(2026, 11, 13)
ALFAMA = (38.7115, -9.1305)


# ── Parser ───────────────────────────────────────────────────────────────────
def test_parser_builds_request_and_sends_today():
    brain = FakeBrain({"parser": [stay_json(must_haves=["wifi", "wifi", "jacuzzi"],
                                            nice_to_haves=["wifi", "breakfast"])]})
    req = parse_stay(brain, "Lisbon in Dec", TODAY).request
    assert req.city == "Lisbon" and req.nights == 3 and req.target == "Alfama, Lisbon"
    assert req.must_haves == ["wifi"]              # de-duped, unknown key dropped
    assert req.nice_to_haves == ["breakfast"]      # can't be both must and nice
    assert brain.calls[0][1]["user"].startswith("TODAY: 2026-11-13")


@pytest.mark.parametrize("over, fragment", [
    ({"check_out": "2026-12-10"}, "must be after"),
    ({"check_in": "2026-01-01", "check_out": "2026-01-03"}, "in the past"),
    ({"check_in": "soon"}, "check-in and check-out"),
    ({"city": ""}, "Which city"),
])
def test_validate_turns_bad_fields_into_questions(over, fragment):
    out = validate(stay_json(**over), TODAY)
    assert out.request is None and fragment in out.question


def test_parser_passes_through_model_question():
    brain = FakeBrain({"parser": [stay_json(status="needs_clarification",
                                            clarification_question="Which dates?")]})
    assert parse_stay(brain, "a hotel in Rome", TODAY).question == "Which dates?"


# ── Geo ──────────────────────────────────────────────────────────────────────
def test_km_lisbon_to_porto_is_about_274():
    assert km((38.7223, -9.1393), (41.1579, -8.6291)) == pytest.approx(274, abs=3)


def test_geocoder_caches_and_waits_politely():
    calls, sleeps = [], []
    g = Geocoder(get=lambda url: calls.append(url) or [{"lat": "1.5", "lon": "2.5"}],
                 sleep=sleeps.append)
    assert g.locate("Alfama, Lisbon") == (1.5, 2.5)
    assert g.locate("alfama, lisbon ") == (1.5, 2.5)  # cache hit: no second request
    assert len(calls) == 1 and "format=jsonv2" in calls[0]
    g.locate("Porto")
    assert sleeps and sleeps[0] > 0  # second real request waited for the rate limit


# ── OpenStreetMap source ─────────────────────────────────────────────────────
def test_osm_amenities_maps_yes_no_and_unknown():
    has, lacks = sources.osm_amenities({"internet_access": "wlan", "wheelchair": "no",
                                        "smoking": "no"})
    assert has == ["wifi", "non_smoking"] and lacks == ["wheelchair_accessible"]


def test_osm_source_parses_ranks_and_widens():
    posted = []

    def post(url, form):
        posted.append(form["data"])
        return OVERPASS

    out = sources.OsmSource(post=post).search(
        validate(stay_json(), TODAY).request, ALFAMA)
    names = [o.name for o in out.options]
    assert names[-1] == "Backpack Hub"               # hostels after hotels
    assert names[0] == "Casa Alfama"                 # nearest hotel first
    casa = out.options[0]
    assert casa.id == "H1" and casa.stars == 4 and casa.address == "5 Rua A"
    assert set(casa.amenities) == {"wifi", "air_conditioning"}
    assert casa.distance_km < 0.2 and casa.price_per_night is None
    assert next(o for o in out.options if o.name == "Baixa Palace").stars == 5  # "5S"
    assert len(posted) == 2 and "around:5000" in posted[1]  # 4 < 6 results → widened


def test_osm_source_tries_mirrors_then_gives_up_gracefully():
    hosts = []

    def post(url, form):
        hosts.append(url)
        if "overpass-api.de" in url:
            raise OSError("HTTP Error 502: Bad Gateway")
        return OVERPASS

    out = sources.OsmSource(post=post).search(validate(stay_json(), TODAY).request, ALFAMA)
    assert out.options and "502" in out.notes[0]
    assert "private.coffee" in hosts[1]

    def down(url, form):
        raise OSError("timed out")

    out = sources.OsmSource(post=down).search(validate(stay_json(), TODAY).request, ALFAMA)
    assert out.options == [] and len(out.notes) == 3


def test_osm_source_without_point_returns_nothing():
    req = validate(stay_json(), TODAY).request
    assert sources.OsmSource(post=lambda *a: 1 / 0).search(req, None).options == []


# ── Web source (Claude) ──────────────────────────────────────────────────────
def test_nightly_and_rating_normalisation():
    assert sources.nightly(450, "total_stay", 3) == 150
    assert sources.nightly(120, "per_night", 3) == 120
    assert sources.nightly(None, "per_night", 3) is None
    assert sources.rating10(4.5, 5) == 9.0
    assert sources.rating10(4.6, None) == 9.2      # bare small number → out of 5
    assert sources.rating10(8.7, 10) == 8.7


def test_web_source_keeps_only_urls_it_saw_and_resumes_pause_turn():
    row = {"name": "Memmo Alfama", "address": "Travessa X", "price": 480,
           "price_basis": "total_stay", "currency": "eur", "rating": 9.1,
           "rating_scale": 10, "review_count": 1200, "stars": 4,
           "amenities": ["wifi", "air_conditioning"], "lacks": [],
           "source_url": "https://hotels.example/memmo/", "caveat": ""}
    fake_row = {**row, "name": "Invented Hotel", "source_url": "https://nowhere.example/x"}
    client = FakeAnthropic([
        msg(search_result("https://hotels.example/memmo"), text("partial"),
            stop_reason="pause_turn"),
        msg(text("Memmo Alfama, €480 total, 9.1/10 https://hotels.example/memmo")),
        msg(text(__import__("json").dumps({"hotels": [row, fake_row]}))),
    ])
    geo = FakeGeocoder({"Memmo Alfama, Travessa X": (38.7112, -9.1302)})
    src = sources.WebSource(ClaudeBrain(client), geo)
    out = src.search(validate(stay_json(), TODAY).request, ALFAMA)

    assert [o.name for o in out.options] == ["Memmo Alfama"]  # invented one dropped
    h = out.options[0]
    assert h.price_per_night == 160 and h.currency == "EUR" and h.estimated
    assert h.distance_km is not None and h.distance_km < 0.1
    assert any("dropped 1" in n for n in out.notes)
    # pause_turn was resumed by echoing the assistant turn back.
    assert client.requests[1]["messages"][1]["role"] == "assistant"
    assert client.requests[0]["tools"][0]["type"] == "web_search_20260318"
    assert "tools" not in client.requests[2]  # extraction call has no tools


def test_pick_source_rules():
    claude = ClaudeBrain(FakeAnthropic([]))
    local = FakeBrain({})
    assert sources.pick_source("auto", claude).name == "web"
    assert sources.pick_source("auto", local).name == "osm"
    with pytest.raises(ValueError, match="Claude"):
        sources.pick_source("web", local)


def test_demo_source_is_deterministic():
    req = validate(stay_json(), TODAY).request
    a = sources.DemoSource().search(req, ALFAMA).options
    b = sources.DemoSource().search(req, ALFAMA).options
    assert [(o.name, o.price_per_night) for o in a] == [(o.name, o.price_per_night) for o in b]
    assert all("(demo)" in o.name for o in a)


# ── Advisor ──────────────────────────────────────────────────────────────────
def _analysis():
    req = validate(stay_json(must_haves=[], min_rating=None), TODAY).request
    hotels = [HotelOption(id="H1", name="A", source="demo", price_per_night=100,
                          currency="EUR", rating=9.0, distance_km=0.5),
              HotelOption(id="H2", name="B", source="demo", price_per_night=80,
                          currency="EUR", rating=7.0, distance_km=3.0)]
    return req, analyse(req, hotels)


def test_advisor_drops_unknown_ids_and_guarantees_best_overall():
    req, analysis = _analysis()
    brain = FakeBrain({"advisor": [advice_json([
        {"hotel_id": "H9", "role": "best_overall", "reason": "ghost"},
        {"hotel_id": "H2", "role": "best_value", "reason": "cheap"}])]})
    advice = advisor_agent.advise(brain, req, analysis)
    # H2 tops the ranking (cheaper, balanced priority) and was already picked
    # as best_value, so it is promoted (keeping its reason), not duplicated.
    assert analysis.ranked[0].id == "H2"
    assert advice.picks == [{"hotel_id": "H2", "role": "best_overall", "reason": "cheap"}]
    assert any("H9" in c for c in advice.corrections)


def test_advisor_falls_back_when_brain_fails():
    req, analysis = _analysis()
    brain = FakeBrain({"advisor": [BrainError("model not found")]})
    advice = advisor_agent.advise(brain, req, analysis)
    assert advice.picks[0]["role"] == "best_overall"
    assert "model not found" in advice.corrections[0]
