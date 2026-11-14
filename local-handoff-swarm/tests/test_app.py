"""The Streamlit page, run headless with Streamlit's AppTest and a scripted backend."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from swarm import backends
from swarm.backends import Reply

from .conftest import ScriptedBackend, call


def test_chat_shows_handoffs_and_the_answer(monkeypatch, fake_search):
    fake = ScriptedBackend([call("transfer_to_researcher"),
                            call("web_search", query="ollama"),
                            Reply(text="Ollama 1.0 shipped [1].")])
    # The app builds its backend via OllamaBackend(model=...); hand it the fake instead.
    monkeypatch.setattr(backends, "OllamaBackend", lambda model: fake)

    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("What's new with Ollama?").run()
    assert not at.exception

    page = " ".join(m.value for m in at.markdown)
    assert "Handoff:" in page and "Researcher" in page
    assert "Ollama 1.0 shipped" in page
    assert "web_search" in " ".join(c.value for c in at.caption)
    assert "▶ 🔎 Researcher" in page                     # roster highlights the new active agent
    assert "example.com/ollama" in page                  # sources listed in the sidebar
