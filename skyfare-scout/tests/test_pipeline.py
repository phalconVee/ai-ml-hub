"""The whole crew end to end, and the Streamlit app, with a scripted Claude."""

from datetime import date

from streamlit.testing.v1 import AppTest

from fakes import FakeClient, advice_json, trip_json
from scout.pipeline import run
from scout.sources import DemoSource

TODAY = date(2026, 11, 13)


def test_full_run_with_demo_flights():
    client = FakeClient({
        "parser": [trip_json()],
        "advisor": [advice_json([{"option_id": "OUT-1", "role": "best_overall", "reason": "x"},
                                 {"option_id": "RET-1", "role": "best_overall", "reason": "y"}])],
    })
    result = run("Lagos to London…", client=client, source=DemoSource(), today=TODAY)

    assert result.question == "" and result.trip.destination == "LON"
    assert set(result.searches) == {"outbound", "return"}
    assert all(o.stops <= 1 for la in result.analysis.legs for o in la.options)
    assert [p["leg"] for p in result.advice.picks] == ["outbound", "return"]
    # One trace line per step: parser, 2 searches, analyst ×2 legs, advisor.
    assert len(result.trace) == 6 and result.trace[-1].startswith("advisor")
    # The advisor saw the scored table, including badges.
    sent = client.calls("advisor")[0]["messages"][0]["content"]
    assert '"badges"' in sent and '"score"' in sent


def test_clarification_stops_before_searching():
    client = FakeClient({"parser": [trip_json(status="needs_clarification",
                                              clarification_question="From where?")]})
    result = run("somewhere warm", client=client, source=DemoSource(), today=TODAY)
    assert result.question == "From where?" and result.searches == {}
    assert client.calls("advisor") == []


def test_streamlit_app_renders_with_fake_claude(monkeypatch):
    from scout import llm

    client = FakeClient({
        "parser": [trip_json(return_date=None)],
        "advisor": [advice_json([{"option_id": "OUT-1", "role": "best_overall",
                                  "reason": "Best balance."}])],
    })
    monkeypatch.setattr(llm, "make_client", lambda: client)
    at = AppTest.from_file("../app.py", default_timeout=30).run()
    at.sidebar.radio[0].set_value("demo")
    at.button[0].click().run()
    assert not at.exception
    page = " ".join(m.value for m in at.markdown)
    assert "Take the cheap one." in page and "Best balance." in page
    assert len(at.dataframe) == 1  # one-way → one table
