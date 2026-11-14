"""The Streamlit UI, headless: Streamlit's AppTest + a fake Claude + the real browser.

Checks the part that is easiest to get wrong in a rerun-based UI: a run that
pauses for approval, survives reruns, and resumes when "Approve" is clicked.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import webpilot.session
from tests.test_agent import ScriptedLLM, find_id, reply, say, tool_use

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def fake_claude(site_url, monkeypatch):
    script = [
        lambda m: reply(say("Opening the test site."), tool_use("goto", url=f"{site_url}/index.html")),
        lambda m: reply(tool_use("read_page")),
        lambda m: reply(tool_use("type_text", element_id=find_id(m, "Your email"), text="me@example.com"),
                        tool_use("submit_form", element_id=find_id(m, "Sign up"))),
        lambda m: reply(say("Signed you up: the site shows **Thanks!**")),
    ]
    llm = ScriptedLLM(script)
    # PilotSession builds LLM(settings) lazily: hand it our fake instead.
    monkeypatch.setattr(webpilot.session, "LLM", lambda settings: llm)
    monkeypatch.setenv("WEBPILOT_DENY", "localhost")
    return llm


def test_approval_flow_in_the_ui(fake_claude):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert "Read-only" in at.caption[0].value

    at.sidebar.toggle[0].set_value(False)  # read-only off: allow typing + submits
    at.run()
    at.chat_input[0].set_value("Sign me up to the newsletter").run()

    # The run paused: the approval box shows the form, and the chat input is disabled.
    assert any("Approve form submission" in s.value for s in at.subheader)
    assert "me@example.com" in at.code[0].value
    assert at.chat_input[0].disabled

    approve = next(b for b in at.button if b.label == "Approve")
    approve.click().run()

    md = " ".join(m.value for m in at.markdown)
    assert "Signed you up" in md
    assert not at.chat_input[0].disabled
    assert not any("Approve form submission" in s.value for s in at.subheader)
    at.session_state.pilot.close()
