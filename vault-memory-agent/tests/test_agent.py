"""The tool loop, with a scripted fake Claude: no key, no network."""

import json
from pathlib import Path

from fakes import FakeClient, text, thinking, tool

from vault import run_turn
from vault.agent import SYSTEM
from vault.config import SETTINGS

DOCS = Path(__file__).parent.parent / "sample_docs"


def test_search_then_answer_saves_the_trace(vault):
    for p in DOCS.glob("*.md"):
        vault.ingest(p.name, p.read_bytes())
    thread = vault.store.create_thread("t")
    client = FakeClient(
        [thinking(), tool("search_knowledge_base", query="turbidity limit UV filter")],
        [thinking(), text("Above 4 NTU, replace the UV cartridge (kestrel-station-handbook.md, §1).")],
    )
    seen = []
    result = run_turn(vault, thread.id, "When must the UV filter be replaced?", client, on_tool=seen.append)

    assert result.rounds == 2 and result.stop_reason == "end_turn"
    assert result.text.startswith("Above 4 NTU")

    # Request 1: the system prompt, all three tools, and just the question.
    first = client.requests[0]
    assert first["system"] == SYSTEM
    assert [t["name"] for t in first["tools"]] == ["search_knowledge_base", "recall", "remember", "forget"]
    assert first["messages"] == [{"role": "user", "content": "When must the UV filter be replaced?"}]

    # Request 2: the assistant turn echoed unchanged (thinking included),
    # then one user message holding the tool_result for that tool_use id.
    second = client.requests[1]["messages"]
    assert [b.type for b in second[1]["content"]] == ["thinking", "tool_use"]
    result_block = second[2]["content"][0]
    assert result_block["type"] == "tool_result"
    assert result_block["tool_use_id"] == second[1]["content"][1].id
    passages = json.loads(result_block["content"])
    assert any("4 NTU" in p["text"] for p in passages)

    # The thread now holds both messages, with the retrieved chunks on the answer.
    msgs = vault.store.get_messages(thread.id)
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[1].trace[0]["tool"] == "search_knowledge_base"
    assert msgs[1].trace[0]["chunks"][0]["doc"] == "kestrel-station-handbook.md"
    assert seen == msgs[1].trace   # the live callback saw the same steps


def test_a_note_saved_in_one_thread_is_recalled_in_another(vault):
    one = vault.store.create_thread("one")
    run_turn(vault, one.id, "I'm allergic to peanuts, keep that in mind.", FakeClient(
        [tool("remember", content="User is allergic to peanuts", kind="fact")],
        [text("Noted.")],
    ))
    assert vault.store.list_notes()[0].content == "User is allergic to peanuts"

    two = vault.store.create_thread("two")
    client = FakeClient(
        [tool("recall", query="user food allergies")],
        [text("You told me you're allergic to peanuts, so skip the satay.")],
    )
    result = run_turn(vault, two.id, "Any snack ideas for me?", client)
    recalled = json.loads(client.requests[1]["messages"][-1]["content"][0]["content"])
    assert recalled[0]["content"] == "User is allergic to peanuts"
    assert "peanuts" in result.text
    # Thread two's history never contained the allergy: only the note carried it.
    assert all("peanut" not in str(m["content"]) for m in client.requests[0]["messages"])


def test_history_is_replayed_as_text_only(vault):
    t = vault.store.create_thread("t")
    run_turn(vault, t.id, "My name is Ada.", FakeClient(
        [tool("remember", content="User's name is Ada", kind="fact")], [text("Hi Ada!")]))
    client = FakeClient([text("You're Ada.")])
    run_turn(vault, t.id, "What's my name?", client)
    assert client.requests[0]["messages"] == [
        {"role": "user", "content": "My name is Ada."},
        {"role": "assistant", "content": "Hi Ada!"},
        {"role": "user", "content": "What's my name?"},
    ]


def test_history_window_starts_with_a_user_message(vault, monkeypatch):
    import vault.agent as agent_mod
    from dataclasses import replace
    monkeypatch.setattr(agent_mod, "SETTINGS", replace(SETTINGS, history_messages=3))
    t = vault.store.create_thread("t")
    for i in range(2):
        vault.store.add_message(t.id, "user", f"q{i}")
        vault.store.add_message(t.id, "assistant", f"a{i}")
    client = FakeClient([text("ok")])
    run_turn(vault, t.id, "q2", client)
    # The newest 3 are a0, q1, a1; the leading assistant message is dropped.
    assert [m["content"] for m in client.requests[0]["messages"]] == ["q1", "a1", "q2"]


def test_parallel_tool_calls_return_in_one_message(vault):
    t = vault.store.create_thread("t")
    client = FakeClient(
        [tool("recall", query="preferences"), tool("search_knowledge_base", query="power")],
        [text("done")],
    )
    run_turn(vault, t.id, "hi", client)
    results = client.requests[1]["messages"][-1]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]
    assert results[0]["content"] == "No notes saved yet."
    assert "knowledge base is empty" in results[1]["content"]


def test_duplicate_remember_is_reported_not_stored(vault):
    t = vault.store.create_thread("t")
    vault.remember("User works night shifts", "fact", t.id)
    client = FakeClient([tool("remember", content="User works night shifts", kind="fact")], [text("ok")])
    result = run_turn(vault, t.id, "I work nights", client)
    assert "Already known" in client.requests[1]["messages"][-1]["content"][0]["content"]
    assert "duplicate_of" in result.trace[0] and len(vault.store.list_notes()) == 1


def test_forget_replaces_an_outdated_note(vault):
    t = vault.store.create_thread("t")
    old, _ = vault.remember("Project deadline is March 3", "task", t.id)
    client = FakeClient(
        [tool("recall", query="project deadline")],
        [tool("forget", note_id=old.id), tool("remember", content="Project deadline is March 10", kind="task")],
        [text("Updated: the deadline is now March 10.")],
    )
    result = run_turn(vault, t.id, "The deadline moved to March 10.", client)
    assert [s["tool"] for s in result.trace] == ["recall", "forget", "remember"]
    assert [n.content for n in vault.store.list_notes()] == ["Project deadline is March 10"]

    # Forgetting a note that doesn't exist is an error the model can read.
    client = FakeClient([tool("forget", note_id=999)], [text("ok")])
    run_turn(vault, t.id, "forget 999", client)
    block = client.requests[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True and "No note with id 999" in block["content"]


def test_tool_errors_go_back_to_the_model(vault):
    t = vault.store.create_thread("t")
    client = FakeClient([tool("delete_everything")], [text("I can't do that.")])
    result = run_turn(vault, t.id, "wipe it", client)
    block = client.requests[1]["messages"][-1]["content"][0]
    assert block["is_error"] is True and "Unknown tool" in block["content"]
    assert result.trace[0]["error"].startswith("Unknown tool")


def test_the_loop_stops_after_max_rounds(vault):
    t = vault.store.create_thread("t")
    client = FakeClient(*[[tool("recall", query=f"q{i}")] for i in range(SETTINGS.max_tool_rounds)])
    result = run_turn(vault, t.id, "loop forever", client)
    assert result.rounds == SETTINGS.max_tool_rounds and len(client.requests) == SETTINGS.max_tool_rounds
    assert "too many tool calls" in result.text
    assert vault.store.get_messages(t.id)[-1].content == result.text


def test_refusal_is_saved_as_a_plain_answer(vault):
    t = vault.store.create_thread("t")
    result = run_turn(vault, t.id, "something declined", FakeClient([], stop_reason="refusal"))
    assert result.stop_reason == "refusal" and result.text == "I can't help with that request."
