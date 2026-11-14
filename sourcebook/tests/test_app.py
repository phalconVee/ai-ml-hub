"""Click through the Streamlit app headlessly (AppTest) with a fake Claude."""

from __future__ import annotations

from pathlib import Path

from fakes import FakeClient, cite, text_block
from streamlit.testing.v1 import AppTest

from sourcebook import Notebook, load_text

ROOT = Path(__file__).parent.parent


def test_chat_renders_clickable_citations(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCEBOOK_HOME", str(tmp_path))
    # Pre-build a notebook on disk so the app opens it.
    nb = Notebook("Bee notebook", home=tmp_path)
    nb.add_source(load_text((ROOT / "sample_sources" / "waggle_dance.md").read_text(), "waggle_dance.md"))
    nb.save()
    p = nb.retriever.search("radar tracking recruits")[0]

    import sourcebook.config as cfg
    old_home = cfg.SETTINGS.home
    object.__setattr__(cfg.SETTINGS, "home", tmp_path)   # settings were read at import time
    try:
        _drive_app(tmp_path, p)
    finally:
        object.__setattr__(cfg.SETTINGS, "home", old_home)


def _drive_app(tmp_path, p):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["_client"] = FakeClient(answers=[[
        text_block("Recruits flew toward the danced spot", [cite(0, 0, 1, p.sentences[0])]),
        text_block("."),
    ]])
    at.run()
    assert not at.exception
    assert at.title[0].value == "Bee notebook"

    at.chat_input[0].set_value("radar tracking recruits").run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert 'href="#t0-c1"' in html            # inline [1] links to its card
    assert 'id="t0-c1"' in html and "<mark>" in html
    saved = Notebook.load("Bee notebook", home=tmp_path)
    assert len(saved.memory.turns) == 1      # the turn was persisted
