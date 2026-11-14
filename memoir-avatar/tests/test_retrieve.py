"""Recall: seeds, spreading through the graph, ranking, the context budget."""

from memoir.graph import KnowledgeGraph
from memoir.retrieve import match_score, recall

from .fakes import TURN1, TURN2, TURN3, rel


def memory():
    kg = KnowledgeGraph(":memory:")
    kg.merge(TURN1, turn=1)
    kg.merge(TURN2, turn=2)
    kg.merge(TURN3, turn=3)
    # A separate island, not connected to the User.
    kg.merge({"entities": [], "relations": [rel("Rust", "uses", "Borrow checker")]}, turn=4)
    return kg


def test_empty_memory_says_so():
    r = recall(KnowledgeGraph(":memory:"), "hello")
    assert r.facts == [] and r.as_context() == "(nothing relevant in memory yet)"


def test_match_score():
    assert match_score("I love the Eiffel Tower", "Eiffel Tower", [], "") == 3.0
    assert match_score("what about Henry?", "User", ["henry"], "") == 3.0
    assert match_score("the tower", "Eiffel Tower", [], "") == 1.0   # half the name's words
    assert match_score("weather today", "Lagos", [], "") == 0.0


def test_two_hops_reach_what_shares_no_words_with_the_question():
    # "pottery" appears nowhere in the question: User → Ada → Pottery.
    r = recall(memory(), "What should I get my sister for her birthday?")
    sentences = [f.sentence() for f in r.facts]
    assert any(s.startswith("Ada likes Pottery") for s in sentences)
    assert r.seeds["User"] == 2.5                                  # "I"/"my" → personal
    assert not any("Borrow checker" in s for s in sentences)       # unrelated island left out


def test_named_island_is_recalled():
    r = recall(memory(), "Can you explain Rust lifetimes?")
    assert r.seeds["Rust"] == 3.0
    assert r.facts[0].sentence().startswith("Rust uses Borrow checker")


def test_retired_facts_rank_below_current_ones():
    facts = recall(memory(), "where do I live?").facts
    order = [f.dst for f in facts if f.predicate == "lives_in"]
    assert order == ["Berlin", "Lagos"]


def test_budget_is_respected():
    assert len(recall(memory(), "tell me everything about me", max_facts=2).facts) == 2
