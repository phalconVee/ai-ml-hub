"""The temporal store on its own: confirmation, supersession, history, edits."""

from recall.store import NewFact
from recall.textsearch import BM25, tokenize


def nf(**kw):
    base = dict(subject="user", predicate="lives_in", object="Berlin",
                statement="The user lives in Berlin.", exclusive=True,
                object_is_entity=True, object_kind="place")
    return NewFact(**{**base, **kw})


def test_profile_has_a_user_node_and_aliases_resolve_to_it(store):
    pid = store.get_or_create_profile("Ana")
    assert store.get_or_create_profile("Ana") == pid            # idempotent
    me = store.user_entity(pid)
    assert store.resolve_entity(pid, "I") == me == store.resolve_entity(pid, "the user")
    assert store.resolve_entity(pid, " Maya ") == store.resolve_entity(pid, "maya")


def test_repeating_a_fact_confirms_instead_of_duplicating(store, clock):
    pid = store.get_or_create_profile("Ana")
    first = store.add_fact(pid, nf())
    clock.advance(days=3)
    again = store.add_fact(pid, nf(statement="Ana lives in berlin."))
    assert again.action == "confirmed" and again.fact.id == first.fact.id
    assert again.fact.last_confirmed_at > first.fact.last_confirmed_at
    assert len(store.facts(pid)) == 1


def test_exclusive_predicate_supersedes_the_old_value(store, clock):
    pid = store.get_or_create_profile("Ana")
    berlin = store.add_fact(pid, nf()).fact
    clock.advance(days=40)
    res = store.add_fact(pid, nf(object="Lisbon", statement="The user lives in Lisbon."))

    assert [f.id for f in res.superseded] == [berlin.id]
    old = store.get_fact(berlin.id)
    assert old.status == "superseded" and old.superseded_by == res.fact.id
    assert old.valid_to == res.fact.valid_from                 # windows meet exactly
    assert [f.object for f in store.facts(pid)] == ["Lisbon"]  # only the present is active
    assert [f.object for f in store.history(berlin.id)] == ["Berlin", "Lisbon"]


def test_non_exclusive_facts_accumulate(store):
    pid = store.get_or_create_profile("Ana")
    store.add_fact(pid, nf(predicate="likes", object="tea", exclusive=False, statement="Likes tea"))
    store.add_fact(pid, nf(predicate="likes", object="jazz", exclusive=False, statement="Likes jazz"))
    assert len(store.facts(pid)) == 2


def test_explicit_replaces_catches_what_rules_cannot(store):
    pid = store.get_or_create_profile("Ana")
    coffee = store.add_fact(pid, nf(predicate="drinks", object="coffee", exclusive=False,
                                    statement="Drinks coffee every morning")).fact
    res = store.add_fact(pid, nf(predicate="quit", object="coffee", exclusive=False,
                                 statement="Quit coffee", replaces=[coffee.id]))
    assert [f.id for f in res.superseded] == [coffee.id]


def test_profiles_are_isolated(store):
    ana, ben = store.get_or_create_profile("Ana"), store.get_or_create_profile("Ben")
    secret = store.add_fact(ana, nf()).fact
    # Ben's extractor can't supersede Ana's fact by id.
    res = store.add_fact(ben, nf(predicate="likes", exclusive=False, replaces=[secret.id]))
    assert res.superseded == [] and store.get_fact(secret.id).active
    assert [f.id for f in store.facts(ben)] == [res.fact.id]


def test_edit_makes_a_new_version_and_wording_fix_does_not(store):
    pid = store.get_or_create_profile("Ana")
    f = store.add_fact(pid, nf()).fact
    typo = store.edit_fact(f.id, "The user lives in Berlin, Germany.")
    assert typo.action == "edited" and typo.fact.id == f.id

    moved = store.edit_fact(f.id, "The user lives in Hamburg.", "Hamburg")
    assert moved.fact.origin == "user edit" and moved.fact.subject == "Ana"
    assert store.get_fact(f.id).superseded_by == moved.fact.id


def test_delete_forgets_and_keeps_links_consistent(store):
    pid = store.get_or_create_profile("Ana")
    old = store.add_fact(pid, nf()).fact
    new = store.add_fact(pid, nf(object="Lisbon", statement="Lives in Lisbon")).fact
    store.delete_fact(new.id)
    assert store.get_fact(new.id) is None
    assert store.get_fact(old.id).superseded_by is None


def test_bm25_prefers_rare_words_and_tokenizer_folds_plurals():
    assert tokenize("My peanuts and the dogs") == ["peanut", "dog"]
    bm = BM25(["user likes coffee", "user allergic to peanuts", "user likes tea"])
    scores = bm.normalized("peanut allergy")
    assert scores.index(1.0) == 1 and scores[0] == 0
