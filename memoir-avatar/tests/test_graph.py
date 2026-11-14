"""The memory's bookkeeping: identity, de-duplication, history, forgetting."""

import pytest

from memoir.graph import KnowledgeGraph, norm_key, norm_predicate

from .fakes import TURN1, TURN2, TURN3, ent, rel


@pytest.fixture
def kg():
    return KnowledgeGraph(":memory:")


def test_normalisation():
    assert norm_key("  The  Eiffel Tower! ") == "eiffel tower"
    assert norm_key("me") == norm_key("I") == norm_key("User") == "user"
    assert norm_predicate("Works At") == norm_predicate("works-at") == "works_at"
    assert norm_predicate("!!") == "related_to"


def test_user_node_exists_from_the_start(kg):
    assert kg.find("User") is not None and kg.find("me")["name"] == "User"
    assert kg.stats() == {"entities": 1, "facts": 0, "retired": 0, "turns": 0}


def test_merge_creates_and_alias_resolves(kg):
    d = kg.merge(TURN1, turn=1)
    assert set(d.new_entities) == {"Lagos", "Ada"}          # User already existed
    assert "User lives in Lagos" in d.new_facts
    assert kg.find("Henry")["name"] == "User"                # the user's name is an alias
    assert kg.find("the lagos")["name"] == "Lagos"


def test_repeating_a_fact_reinforces_instead_of_duplicating(kg):
    kg.merge(TURN1, turn=1)
    d = kg.merge({"entities": [], "relations": [rel("Henry", "sister", "ada")]}, turn=2)
    assert d.reinforced == ["User sister Ada"] and not d.new_facts
    [f] = [f for f in kg.facts() if f.predicate == "sister"]
    assert (f.weight, f.first_turn, f.last_turn) == (2, 1, 2)
    assert kg.find("Ada")["mentions"] == 1                   # entities list didn't repeat her


def test_exclusive_relation_retires_the_old_one(kg):
    kg.merge(TURN1, turn=1)
    d = kg.merge(TURN3, turn=3)
    assert d.retired == ["User lives in Lagos"]
    by_dst = {f.dst: f for f in kg.facts() if f.predicate == "lives_in"}
    assert by_dst["Berlin"].active and not by_dst["Lagos"].active
    assert by_dst["Lagos"].ended_turn == 3
    assert "NO LONGER TRUE since turn 3" in by_dst["Lagos"].sentence()


def test_negation_retires_and_restating_revives(kg):
    kg.merge(TURN1, turn=1)
    d = kg.merge({"entities": [], "relations": [rel("User", "sister", "Ada", negated=True)]}, turn=2)
    assert d.retired == ["User sister Ada"]
    # Negating something we never knew does nothing.
    d = kg.merge({"entities": [], "relations": [rel("User", "likes", "Jazz", negated=True)]}, turn=2)
    assert d.empty() is False and not d.retired             # "Jazz" node gets created, no fact
    assert not any(f.dst == "Jazz" for f in kg.facts())
    d = kg.merge({"entities": [], "relations": [rel("User", "sister", "Ada")]}, turn=3)
    assert d.reinforced == ["User sister Ada"]
    assert all(f.active for f in kg.facts() if f.predicate == "sister")


def test_relation_to_unlisted_entity_creates_it_and_self_loops_are_dropped(kg):
    d = kg.merge({"entities": [], "relations": [rel("User", "learning", "Rust"),
                                                rel("User", "is", "me")]}, turn=1)
    assert "Rust" in d.new_entities and d.new_facts == ["User learning Rust"]


def test_alias_cannot_shadow_another_entity(kg):
    kg.merge(TURN1, turn=1)
    kg.merge({"entities": [ent("Adaeze", "person", "", ["Ada"])], "relations": []}, turn=2)
    assert kg.find("Ada")["name"] == "Ada"                   # still the original Ada


def test_description_and_type_update_rules(kg):
    kg.merge({"entities": [ent("Acme")], "relations": []}, turn=1)
    kg.merge({"entities": [ent("Acme", "organization", "Fintech startup")], "relations": []}, turn=2)
    kg.merge({"entities": [ent("Acme", "project", "")], "relations": []}, turn=3)
    row = kg.find("Acme")
    assert (row["type"], row["description"], row["mentions"]) == ("organization", "Fintech startup", 3)


def test_networkx_view(kg):
    kg.merge(TURN1, turn=1)
    kg.merge(TURN2, turn=2)
    g = kg.to_networkx()
    assert g.has_edge("Ada", "Pottery", key="likes")
    assert "henry" in g.nodes["User"]["aliases"]


def test_forget_cascades_and_protects_the_user(kg):
    kg.merge(TURN1, turn=1)
    kg.merge(TURN2, turn=2)
    assert kg.forget("Ada") is True
    assert kg.find("Ada") is None and all("Ada" not in (f.src, f.dst) for f in kg.facts())
    assert kg.forget("User") is False and kg.forget("Nobody") is False


def test_persists_across_reopen_and_reset(tmp_path):
    path = tmp_path / "m.db"
    kg = KnowledgeGraph(path)
    kg.add_turn("hi", "hello")
    kg.merge(TURN1, turn=1)
    again = KnowledgeGraph(path)
    assert again.stats()["facts"] == 2 and again.turn_count() == 1
    again.reset()
    assert again.stats() == {"entities": 1, "facts": 0, "retired": 0, "turns": 0}
