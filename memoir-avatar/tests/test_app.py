"""Drive the real Streamlit script headless: fake Claude, browser voice, temp memory."""

from streamlit.testing.v1 import AppTest

from .fakes import TURN1, FakeClient, as_json, text


def test_chat_turn_updates_chat_avatar_and_graph(monkeypatch, tmp_path):
    client = FakeClient([text("Nice to meet you, Henry!"), as_json(TURN1)])
    monkeypatch.setenv("MEMOIR_DB", str(tmp_path / "memory.db"))
    # app.py imports these names at start-up, so patch the modules they come from.
    monkeypatch.setattr("memoir.llm.make_client", lambda: client)
    monkeypatch.setattr("memoir.voice.available_engines", lambda: ["browser"])

    at = AppTest.from_file("../app.py", default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("Hi, I'm Henry and I live in Lagos with my sister Ada.").run()

    assert not at.exception
    assert len(client.requests) == 2                         # reply + extraction
    log = at.session_state.log
    assert [m["role"] for m in log] == ["user", "assistant"]
    assert log[-1]["content"] == "Nice to meet you, Henry!"
    assert log[-1]["delta"].startswith("memory: +2 new")
    assert at.session_state.avatar["text"] == "Nice to meet you, Henry!"
    assert at.session_state.avatar["audio_b64"] == ""         # browser speech: no server audio
    assert at.session_state.pending is None
    # The sidebar metrics read the graph after the merge.
    metrics = {m.label: m.value for m in at.sidebar.metric}
    assert metrics == {"Things": "3", "Facts": "2", "Turns": "1"}
