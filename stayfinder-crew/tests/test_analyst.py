"""The analyst is pure Python, so it gets exact tests."""

from datetime import date

from stay.analyst import analyse, location_score, quality_score
from stay.models import HotelOption
from stay.parser_agent import validate

from .fakes import stay_json


def req(**over):
    return validate(stay_json(**over), date(2026, 11, 13)).request


def hotel(id, **kw):
    return HotelOption(id=id, name=f"Hotel {id}", source="demo", currency="EUR", **kw)


def test_rules_out_only_on_known_facts():
    r = req(must_haves=["wifi"], min_rating=8)
    a = analyse(r, [hotel("H1", lacks=["wifi"]),           # known no → out
                    hotel("H2", rating=7.5),                # below minimum → out
                    hotel("H3"),                            # unknown everything → kept
                    hotel("H4", amenities=["wifi"], rating=9)])
    assert {o.id for o, _ in a.excluded} == {"H1", "H2"}
    assert [o.id for o in a.ranked] == ["H4", "H3"]
    assert a.ranked[0].must_have_status == {"wifi": "yes"}
    assert "all must-haves confirmed" in a.ranked[0].badges


def test_priority_changes_the_winner():
    cheap_far = dict(price_per_night=70, distance_km=6.0, rating=7.5)
    dear_near = dict(price_per_night=150, distance_km=0.3, rating=8.0)
    for priority, winner in [("value", "CHEAP"), ("location", "NEAR")]:
        a = analyse(req(priority=priority, must_haves=[], min_rating=None),
                    [hotel("CHEAP", **cheap_far), hotel("NEAR", **dear_near)])
        assert a.ranked[0].id == winner, priority


def test_over_budget_is_flagged_and_penalised():
    a = analyse(req(budget_per_night=100, must_haves=[], min_rating=None),
                [hotel("H1", price_per_night=100), hotel("H2", price_per_night=150)])
    h2 = next(o for o in a.ranked if o.id == "H2")
    assert h2.over_budget and h2.parts["price"] == 0.0   # 50% over → zero
    assert a.within_budget == 1


def test_unknowns_score_neutral_and_other_currencies_are_not_compared():
    a = analyse(req(must_haves=[], min_rating=None, nice_to_haves=[]),
                [hotel("H1"), HotelOption(id="H2", name="X", source="web",
                                          price_per_night=10, currency="USD")])
    for o in a.ranked:
        assert o.parts["price"] == 0.5  # USD price not compared against EUR request
    assert a.known["price"] == 0 and a.notes[0].startswith("no prices known in EUR")


def test_badges_and_component_scores():
    a = analyse(req(must_haves=[], min_rating=None),
                [hotel("H1", price_per_night=90, distance_km=2.0, rating=8.0),
                 hotel("H2", price_per_night=120, distance_km=0.4, rating=9.4)])
    badges = {o.id: o.badges for o in a.ranked}
    assert "cheapest" in badges["H1"]
    assert "closest" in badges["H2"] and "top rated" in badges["H2"]
    assert location_score(hotel("x", distance_km=2.0)) == 0.5
    assert quality_score(hotel("x", rating=9.0, stars=5)) == 0.92
