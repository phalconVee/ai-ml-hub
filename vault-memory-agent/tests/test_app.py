"""Click through the Streamlit app with AppTest and a fake Claude."""

from dataclasses import replace
from pathlib import Path

import pytest
import streamlit as st
from fakes import FakeClient, text, tool
from streamlit.testing.v1 import AppTest

import vault.config
import vault.llm
from vault import HashEmbedder, Vault

APP = str(Path(__file__).parent.parent / "app.py")
DOCS = Path(__file__).parent.parent / "sample_docs"


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    db = tmp_path / "app.db"
    # The app reads SETTINGS and make_client at run time from these modules.
    monkeypatch.setattr(vault.config, "SETTINGS", replace(vault.config.SETTINGS, db_path=db, embedder="hash"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    st.cache_resource.clear()
    # Pre-load a document the way the upload button would.
    v = Vault.open(db, HashEmbedder())
    for p in DOCS.glob("*.md"):
        v.ingest(p.name, p.read_bytes())
    v.store.close()
    yield db
    st.cache_resource.clear()


def test_chat_shows_answer_trace_and_vault(app_env, monkeypatch):
    fake = FakeClient(
        [tool("search_knowledge_base", query="greenhouse pH range"),
         tool("remember", content="User runs the greenhouse night shift", kind="fact")],
        [text("Keep the pH between 5.8 and 6.2 (greenhouse-notes.md, §1).")],
    )
    monkeypatch.setattr(vault.llm, "make_client", lambda: fake)

    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [e.value for e in at.sidebar.selectbox] == [at.sidebar.selectbox[0].value]  # one thread made

    at.chat_input[0].set_value("I run the night shift. What pH should the trays be?").run()
    assert not at.exception
    chat = [m.markdown[0].value for m in at.chat_message]
    assert chat[-1].startswith("Keep the pH between 5.8 and 6.2")
    # The thread got named after its first question.
    assert at.sidebar.selectbox[0].format_func(at.sidebar.selectbox[0].value).startswith("I run the night shift")

    # Vault tab: the note and the retrieved chunks are listed.
    page = " ".join(m.value for m in at.markdown)
    assert "User runs the greenhouse night shift" in page
    assert "greenhouse-notes.md" in page
    assert at.metric[0].value == "2"      # documents
    assert at.metric[4].value == "1"      # notes


def test_new_thread_and_forget_note(app_env, monkeypatch):
    at = AppTest.from_file(APP, default_timeout=30).run()
    v = Vault.open(app_env, HashEmbedder())
    v.remember("User prefers short answers", "preference", None)
    v.store.close()
    at.run()
    assert any("User prefers short answers" in m.value for m in at.markdown)
    next(b for b in at.button if b.label == "Forget").click().run()
    assert not any("User prefers short answers" in m.value for m in at.markdown)

    next(b for b in at.sidebar.button if "New thread" in b.label).click().run()
    assert len(at.sidebar.selectbox[0].options) == 2
