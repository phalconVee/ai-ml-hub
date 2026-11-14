"""The whole loop with a fake Claude: extraction, supersession through the
extractor, retrieval across sessions, and the failure paths."""

from recall.companion import as_api_messages, chat_turn
from recall.extractor import parse_facts
from recall.retriever import retrieve
from tests.fakes import FakeClient, fact


def scripted(rules):
    """extract(text) → facts, from {substring: [facts]} rules."""
    def extract(text, prompt):
        return [f for key, facts in rules.items() if key in text.lower() for f in facts]
    return extract


def test_extraction_is_stored_with_its_source(store):
    client = FakeClient(extract=scripted({"vegetarian": [
        fact("user", "diet", "vegetarian", "The user is vegetarian.", importance=5,
             exclusive=True, category="health")]}))
    pid = store.get_or_create_profile("Ana")
    sid = store.new_session(pid)
    turn = chat_turn(store, client, pid, sid, "Hi! I'm vegetarian.")

    assert [w.action for w in turn.writes] == ["added"]
    f = store.facts(pid)[0]
    assert f.statement == "The user is vegetarian." and f.subject == "Ana"
    assert f.source_session_id == sid and f.source_message_id == turn.user_message_id
    # 3 calls per turn: plan recall, answer, extract.
    assert len(client.calls) == 3


def test_memory_carries_across_sessions(store, clock):
    client = FakeClient(
        extract=scripted({
            "vegetarian": [fact("user", "diet", "vegetarian", "The user is vegetarian.",
                                importance=5, exclusive=True)],
            "sister": [fact("user", "has_sister", "Maya", "The user's sister is Maya.",
                            object_is_entity=True, kind="person", importance=4),
                       fact("Maya", "lives_in", "Porto", "Maya lives in Porto.",
                            object_is_entity=True, kind="place", exclusive=True)],
        }),
        planner_keywords={"Any ideas for visiting her?": "travel trip visit city"},
        planner_entities={"Any ideas for visiting her?": ["Maya"]},
    )
    pid = store.get_or_create_profile("Ana")
    s1 = store.new_session(pid)
    chat_turn(store, client, pid, s1, "I'm vegetarian, and my sister Maya lives in Porto.")

    clock.advance(days=10)
    s2 = store.new_session(pid)                       # a brand-new conversation
    turn = chat_turn(store, client, pid, s2, "Any ideas for visiting her?")

    used = {r.fact.statement for r in turn.recalled}
    assert "Maya lives in Porto." in used             # found via the entity hint
    assert "The user is vegetarian." in used          # core fact: always offered
    answer_call = client.calls[-2]                    # [-1] is the extractor
    assert "Maya lives in Porto." in answer_call["system"]
    # The new session's transcript does NOT contain session 1's messages...
    assert len(answer_call["messages"]) == 1
    # ...and the recall was logged for the inspector.
    assert {f.statement for f, _ in store.recalled_for(turn.assistant_message_id)} == used


def test_a_change_supersedes_through_the_extractor(store, clock):
    pid = store.get_or_create_profile("Ana")
    sid = store.new_session(pid)

    def extract(text, prompt):
        if "berlin" in text.lower():
            return [fact("user", "lives_in", "Berlin", "The user lives in Berlin.", exclusive=True)]
        if "moved to lisbon" in text.lower():
            # The extractor saw "[<id>] The user lives in Berlin." in CURRENT MEMORIES.
            old_id = int(prompt.split("] The user lives in Berlin")[0].rsplit("[", 1)[1])
            return [fact("user", "resides_in", "Lisbon", "The user lives in Lisbon.",
                         replaces=[old_id])]
        return []

    client = FakeClient(extract=extract)
    chat_turn(store, client, pid, sid, "I live in Berlin.")
    clock.advance(days=30)
    turn = chat_turn(store, client, pid, sid, "Big news: I moved to Lisbon!")

    assert [o.statement for o in turn.writes[0].superseded] == ["The user lives in Berlin."]
    assert [f.statement for f in store.facts(pid)] == ["The user lives in Lisbon."]

    # Default retrieval sees only the present; a question about the past sees both.
    now = retrieve(store, pid, "where do I live")
    assert [r.fact.object for r in now] == ["Lisbon"]
    then = retrieve(store, pid, "where did I live before", include_history=True)
    assert {r.fact.object for r in then} == {"Lisbon", "Berlin"}
    assert then[0].fact.object == "Lisbon"            # the present outranks the past


def test_retrieval_needs_a_match_unless_the_fact_is_core(store):
    from recall.store import NewFact
    pid = store.get_or_create_profile("Ana")
    store.add_fact(pid, NewFact("user", "likes", "jazz", "The user likes jazz.", importance=2))
    store.add_fact(pid, NewFact("user", "allergic_to", "peanuts", "The user is allergic to peanuts.",
                                importance=5))
    got = [r.fact.object for r in retrieve(store, pid, "recommend a restaurant")]
    assert got == ["peanuts"]                         # jazz didn't match; the allergy is core
    got = [r.fact.object for r in retrieve(store, pid, "any jazz concerts?")]
    assert got[0] == "jazz"


def test_profiles_never_see_each_others_memories(store):
    client = FakeClient(extract=scripted({"cat": [fact("user", "has_pet", "Miso", "Has a cat called Miso.")]}))
    ana, ben = store.get_or_create_profile("Ana"), store.get_or_create_profile("Ben")
    chat_turn(store, client, ana, store.new_session(ana), "I have a cat named Miso.")
    turn = chat_turn(store, client, ben, store.new_session(ben), "What's my cat called?")
    assert turn.recalled == [] and "none" in turn.answer


def test_failures_degrade_gracefully(store):
    pid = store.get_or_create_profile("Ana")
    sid = store.new_session(pid)
    client = FakeClient()
    client.refuse = True
    turn = chat_turn(store, client, pid, sid, "hello")
    assert "can't help" in turn.answer

    # Malformed extractor output: bad items are skipped, good ones kept.
    good = fact("user", "likes", "tea")
    assert len(parse_facts({"facts": [good, {"subject": "x"}, {**good, "object": "  "}]})) == 1
    assert parse_facts("not a dict") == []


def test_api_messages_alternate_and_start_with_user():
    hist = [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "a"},
            {"role": "user", "content": "b"}, {"role": "assistant", "content": "c"}]
    msgs = as_api_messages(hist, "d")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "a\n\nb"


def test_planner_failure_falls_back_to_plain_search(store):
    from recall.recall_planner import plan_recall
    client = FakeClient()
    client.fail = True
    plan = plan_recall(client, "what do I like?", [])
    assert plan.query == "what do I like?" and not plan.by_llm
