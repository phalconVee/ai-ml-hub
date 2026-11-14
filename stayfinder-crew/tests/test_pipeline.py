"""End to end: the whole crew with fake brain + demo/OSM data, and the UI."""

from datetime import date

from streamlit.testing.v1 import AppTest

from stay.pipeline import run
from stay.sources import DemoSource, OsmSource

from .fakes import OVERPASS, FakeBrain, FakeGeocoder, advice_json, stay_json

TODAY = date(2026, 11, 13)
GEO = {"Alfama, Lisbon": (38.7115, -9.1305), "Lisbon": (38.7223, -9.1393)}


def test_full_run_with_osm():
    brain = FakeBrain({"parser": [stay_json()],
                       "advisor": [advice_json([{"hotel_id": "H1", "role": "best_overall",
                                                 "reason": "Closest, has wifi + AC."}])]})
    steps = []
    result = run("Lisbon…", brain=brain, source=OsmSource(post=lambda u, f: OVERPASS),
                 geocoder=FakeGeocoder(GEO), today=TODAY, on_step=steps.append)
    assert result.question == ""
    assert result.target_point == GEO["Alfama, Lisbon"]
    # "Offline Inn" has internet_access=no and wifi is a must-have.
    assert [o.name for o, _ in result.analysis.excluded] == ["Offline Inn"]
    assert result.advice.picks[0]["hotel_id"] == "H1"
    assert len(steps) == 5 and steps[0].startswith("parser:")
    # The advisor saw unknown prices as null, not 0.
    sent = brain.calls[1][1]["user"]
    assert '"price_per_night": null' in sent


def test_unknown_area_falls_back_to_city_point():
    brain = FakeBrain({"parser": [stay_json(area="Nowhereville")],
                       "advisor": [advice_json([])]})
    result = run("x", brain=brain, source=DemoSource(), geocoder=FakeGeocoder(GEO),
                 today=TODAY)
    assert result.target_point == GEO["Lisbon"]
    assert result.advice.picks[0]["role"] == "best_overall"  # verify() filled it in


def test_clarification_stops_before_searching():
    brain = FakeBrain({"parser": [stay_json(status="needs_clarification",
                                            clarification_question="Which city?")]})

    class Boom:
        name, label = "boom", "boom"

        def search(self, *a):
            raise AssertionError("must not search")

    result = run("a hotel please", brain=brain, source=Boom(), today=TODAY)
    assert result.question == "Which city?" and result.analysis is None


def test_streamlit_app_renders_demo_result():
    """Drive the real app.py with the Ollama brain swapped for a fake."""
    at = AppTest.from_string(_APP_WITH_FAKE_BRAIN, default_timeout=30)
    at.run()
    at.radio[0].set_value("ollama")
    at.radio[1].set_value("demo")
    at.run()
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert any("Lisbon" in s.value for s in at.subheader)
    assert len(at.dataframe) == 1
    assert any("Best Overall" in m.value for m in at.markdown)


# The script imports the test fakes, patches the brain factory, then execs the
# real app.py, so the UI code under test is the shipped file, unchanged.
_APP_WITH_FAKE_BRAIN = f'''
import pathlib, sys
sys.path.insert(0, {str(__import__("pathlib").Path(__file__).parents[1])!r})
from stay import llm
from tests.fakes import FakeBrain, stay_json, advice_json
llm.make_brain = lambda kind, **kw: FakeBrain({{
    "parser": [stay_json()],
    "advisor": [advice_json([{{"hotel_id": "H1", "role": "best_overall", "reason": "ok"}}])]}})
llm.OllamaBrain.installed_models = lambda self: ["qwen3:4b"]
exec(compile(pathlib.Path(sys.path[0], "app.py").read_text(), "app.py", "exec"))
'''
