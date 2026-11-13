"""Worked examples of the comparison maths. No LLM anywhere in this file."""

from scout.analyst import analyse, analyse_leg, dominates, pareto_ids
from scout.models import FlightOption, TripRequest


def trip(**over):
    base = dict(origin="LOS", destination="LHR", depart_date="2026-12-18",
                return_date=None, priority="balanced")
    return TripRequest(**{**base, **over})


def opt(id, price, minutes, stops, currency="USD", leg="outbound"):
    return FlightOption(id=id, leg=leg, airline="X", flight_numbers=[], depart_time="",
                        arrive_time="", duration_minutes=minutes, stops=stops,
                        price=price, currency=currency, source="demo")


def test_dominates_needs_strictly_better_somewhere():
    assert dominates((100, 300, 0), (120, 300, 0))
    assert not dominates((100, 300, 0), (100, 300, 0))   # equal is not domination
    assert not dominates((100, 400, 0), (120, 300, 0))   # a trade-off


def test_pareto_drops_strictly_worse_options():
    options = [opt("A", 400, 360, 0),   # direct, pricey
               opt("B", 250, 600, 1),   # cheap, slow
               opt("C", 450, 700, 2)]   # worse than A on everything
    assert pareto_ids(options, trip()) == {"A", "B"}


def test_priority_changes_the_winner():
    def winner(priority):
        options = [opt("CHEAP", 250, 900, 1), opt("FAST", 600, 360, 0)]
        return analyse_leg("outbound", options, trip(priority=priority)).options[0].id

    assert winner("cheapest") == "CHEAP"
    assert winner("fastest") == "FAST"


def test_badges_scores_and_budget():
    la = analyse_leg("outbound",
                     [opt("A", 400, 360, 0), opt("B", 250, 600, 1), opt("C", 450, 700, 2)],
                     trip(priority="cheapest", budget=300))
    by = {o.id: o for o in la.options}
    assert "cheapest" in by["B"].badges and "best match" in by["B"].badges
    assert {"fastest", "fewest stops", "efficient"} <= set(by["A"].badges)
    assert "efficient" not in by["C"].badges
    assert by["A"].over_budget and not by["B"].over_budget
    # Scores live in 0..1 and the list is sorted best first.
    assert all(0 <= o.score <= 1 for o in la.options)
    assert [o.score for o in la.options] == sorted(o.score for o in la.options)


def test_max_stops_filter_and_unknowns_rank_last():
    la = analyse_leg("outbound",
                     [opt("A", 300, 400, 0), opt("B", None, None, 1), opt("C", 200, 500, 2)],
                     trip(max_stops=1))
    assert la.removed == 1 and [o.id for o in la.options][0] == "A"
    b = next(o for o in la.options if o.id == "B")
    assert b.score == 1.0 * 0.45 + 1.0 * 0.35 + 1.0 * 0.20   # worst on everything


def test_other_currency_is_not_compared_on_price():
    la = analyse_leg("outbound", [opt("USD", 500, 400, 0), opt("GBP", 100, 400, 0, "GBP")],
                     trip())
    assert "cheapest" in next(o for o in la.options if o.id == "USD").badges
    assert any("GBP" in n for n in la.notes)


def test_trip_total_and_budget():
    t = trip(return_date="2027-01-03", budget=700)
    result = analyse(t, {"outbound": [opt("O1", 300, 400, 0), opt("O2", 350, 380, 0)],
                         "return": [opt("R1", 380, 400, 0, leg="return")]})
    assert result.cheapest_total == 680 and result.within_budget is True
