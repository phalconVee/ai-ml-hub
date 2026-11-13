"""Drive the real Streamlit script headless, with the fake Claude plugged in."""

from streamlit.testing.v1 import AppTest

from .fakes import FakeClient, article, reporter_turns, verdict


def test_app_files_a_story_end_to_end(monkeypatch):
    client = FakeClient([*reporter_turns(), article(), verdict("publish")])
    # app.py does `from newsroom.llm import make_client`, so patch the module
    # attribute before the script imports it.
    monkeypatch.setattr("newsroom.llm.make_client", lambda: client)

    at = AppTest.from_file("../app.py", default_timeout=30).run()
    assert at.sidebar.button[0].disabled                 # no topic yet → button off
    at.sidebar.text_input[0].input("central bank rates").run()
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert at.header[0].value == "Rates held"
    body = " ".join(m.value for m in at.markdown)
    assert "[[S1]](https://www.centralbank.example/news/rates)" in body  # clickable citation
    assert len(client.requests) == 4
