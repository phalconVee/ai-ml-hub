"""Drive the real Streamlit script headless: fake Claude, fake page, fake voice."""

from streamlit.testing.v1 import AppTest

from .fakes import ARTICLE_HTML, OUTLINE, FakeClient, ToneEngine, draft, edited, reply


def test_app_produces_script_and_audio(monkeypatch):
    client = FakeClient([reply(OUTLINE), draft(), edited()])
    # app.py imports these names at start-up, so patch the modules they come from.
    monkeypatch.setattr("podcast.llm.make_client", lambda: client)
    monkeypatch.setattr("podcast.voices.available_engines", lambda: ["say"])
    monkeypatch.setattr("podcast.voices.pick_engine", lambda name: ToneEngine())
    monkeypatch.setattr("podcast.source.fetch_html", lambda url: ARTICLE_HTML)

    at = AppTest.from_file("../app.py", default_timeout=30).run()
    assert at.button[0].disabled                     # nothing pasted yet → button off
    at.text_area[0].input("https://blog.example/bees").run()
    at.button[0].click().run()

    assert not at.exception
    assert at.main.header[0].value == "Why Bees Dance"
    script = " ".join(m.value for m in at.markdown)
    assert "**Maya:** bee bee" in script and "**Theo:** bee bee" in script
    assert len(client.requests) == 3
    assert at.session_state.episode.seconds > 1      # the tone engine "recorded" it
