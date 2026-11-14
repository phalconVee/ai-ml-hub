"""Click through the real Streamlit page headlessly (AppTest) with a fake
Claude and an in-memory store injected, so no key and no network are needed."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from recall.store import MemoryStore
from tests.fakes import FakeClient, fact

APP = str(Path(__file__).parents[1] / "app.py")


def extract(text, prompt):
    if "porto" in text.lower():
        return [fact("user", "lives_in", "Porto", "The user lives in Porto.", exclusive=True)]
    return []


def test_chat_inspect_edit_and_forget():
    store = MemoryStore(":memory:")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["store"] = store
    at.session_state["client"] = FakeClient(extract=extract)
    at.run()
    assert not at.exception

    at.sidebar.text_input[0].set_value("Ana").run()
    at.sidebar.button[0].click().run()                    # "Add profile"
    assert store.profiles() and not at.exception

    at.chat_input[0].set_value("I live in Porto.").run()
    assert not at.exception and not at.error
    assert any("learned" not in m.value and "Porto" in m.value for m in at.markdown)
    assert any("🆕 added" in m.value for m in at.markdown)  # "Learned from this exchange"

    # Second message: the memory is now *used*, and the inspector says so.
    at.chat_input[0].set_value("Where do I live?").run()
    assert not at.exception
    assert any("score" in m.value and "Porto" in m.value for m in at.markdown)

    # Edit the memory from the inspector → new version + history entry.
    fid = store.facts(store.profiles()[0][0])[0].id
    form_inputs = [t for t in at.text_input if t.label == "Value"]
    form_inputs[0].set_value("Lisbon")
    [t for t in at.text_input if t.label == "Statement"][0].set_value("The user lives in Lisbon.")
    next(b for b in at.button if b.label == "Save edit").click().run()
    assert not at.exception
    pid = store.profiles()[0][0]
    assert [f.object for f in store.facts(pid)] == ["Lisbon"]
    assert store.get_fact(fid).status == "superseded"

    # Forget it.
    next(b for b in at.button if b.label == "🗑 Forget").click().run()
    assert store.facts(pid) == []
