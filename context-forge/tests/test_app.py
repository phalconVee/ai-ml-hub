"""Click through the real Streamlit page headlessly (AppTest), with fake Claude
and fake arXiv injected, so no key and no network are needed."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parents[1] / "app.py")


def test_ask_a_question_and_see_the_context_tabs(tmp_path, monkeypatch):
    from forge import pipeline
    from forge.sources.memory import MemoryStore
    from tests.fakes import FakeClient
    from tests.test_pipeline import fake_arxiv

    # Swap the slow/real pieces before the page runs.
    monkeypatch.setattr(pipeline, "search_arxiv", fake_arxiv)
    monkeypatch.setattr(pipeline.run, "__kwdefaults__",
                        {**pipeline.run.__kwdefaults__, "arxiv": fake_arxiv})

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["client"] = FakeClient()
    at.session_state["memory"] = MemoryStore(tmp_path / "m.json")
    at.run()
    assert not at.exception

    at.button[0].click().run()                      # "Load sample notes"
    assert "chunks from 1 file" in at.sidebar.caption[0].value

    at.chat_input[0].set_value("What is context engineering?").run()
    assert not at.exception and not at.error
    assert "[S1]" in at.markdown[2].value or any("[S1]" in m.value for m in at.markdown)
    labels = [t.label for t in at.tabs]
    assert labels[0].startswith("Sources") and labels[2].startswith("Rejected")
    assert at.session_state["memory"].turns       # the turn was remembered
