"""The whole crew, with Claude replaced by a script. No API key needed."""

import pytest

from newsroom.config import SETTINGS
from newsroom.desk import run_desk
from newsroom.editor import audit
from newsroom.llm import RefusedError
from newsroom.writer import Draft

from .fakes import FakeClient, article, msg, reporter_turns, search_error, text, verdict

TODAY = "Friday, 13 November 2026"


def test_happy_path_reporter_resumes_pause_then_writer_and_editor():
    client = FakeClient([*reporter_turns(), article(), verdict("publish")])
    story = run_desk("rates", client, today=TODAY)

    assert story.status == "published"
    assert story.draft.headline == "Rates held"
    assert [s.id for s in story.used_sources()] == ["S1"]
    assert story.notes.queries == ["central bank rate decision"]
    assert len(story.notes.fetched) == 1

    reporter_1, reporter_2, writer, editor = client.requests
    # pause_turn: the paused assistant turn is sent back as-is, no extra user message.
    assert reporter_2["messages"][-1]["role"] == "assistant"
    assert len(reporter_2["messages"]) == 2
    # Today's date travels in the user message, not the system prompt.
    assert TODAY in reporter_1["messages"][0]["content"] and TODAY not in reporter_1["system"]
    # The writer sees the ledger with ids and quotes, and has no tools.
    assert "[S1]" in writer["messages"][0]["content"]
    assert "The key rate remains at 4.25%." in writer["messages"][0]["content"]
    assert "tools" not in writer


def test_every_request_uses_the_configured_model_fallback_and_effort():
    client = FakeClient([*reporter_turns(), article(), verdict("publish")])
    run_desk("rates", client, today=TODAY)
    efforts = [r["output_config"]["effort"] for r in client.requests]
    assert efforts == [SETTINGS.reporter_effort] * 2 + [SETTINGS.writer_effort, SETTINGS.editor_effort]
    for r in client.requests:
        assert r["model"] == SETTINGS.model
        assert r["fallbacks"] == "default" and r["betas"] == [SETTINGS.fallback_beta]
    tool_types = [t["type"] for t in client.requests[0]["tools"]]
    assert tool_types == ["web_search_20260209", "web_fetch_20260209"]
    # Writer and editor answer in JSON matching a schema.
    assert client.requests[2]["output_config"]["format"]["type"] == "json_schema"
    assert client.requests[3]["output_config"]["format"]["type"] == "json_schema"


def test_editor_sends_it_back_and_the_writer_revises_with_the_notes():
    client = FakeClient([*reporter_turns(),
                         article(body="The bank cut rates [S1]."),
                         verdict("revise", "S1 says rates were HELD, not cut."),
                         article(body="The bank held rates at 4.25% [S1]."),
                         verdict("publish")])
    story = run_desk("rates", client, today=TODAY)

    assert story.status == "published"
    assert len(story.drafts) == 2 and story.draft.revision == 1
    revise_request = client.requests[4]["messages"][0]["content"]
    assert "YOUR PREVIOUS DRAFT" in revise_request and "The bank cut rates" in revise_request
    assert "S1 says rates were HELD, not cut." in revise_request


def test_broken_citations_override_a_publish_verdict():
    # Writer cites a source that doesn't exist; the LLM editor misses it twice.
    bad = article(body="The bank held rates [S9].")
    client = FakeClient([*reporter_turns(), bad, verdict("publish"), bad, verdict("publish")])
    story = run_desk("rates", client, today=TODAY)

    assert [r.verdict for r in story.reviews] == ["revise", "revise"]
    assert story.status == "published_with_notes"     # revision budget (1) used up
    assert "Citation [S9] does not exist" in story.reviews[0].notes[0]


def test_no_sources_means_the_story_is_spiked_and_nothing_is_written():
    client = FakeClient([msg("end_turn", *search_error("rates"), text("Nothing found."))])
    story = run_desk("rates", client, today=TODAY)
    assert story.status == "spiked" and story.draft is None
    assert len(client.requests) == 1                    # writer and editor never called


def test_refusal_stops_the_desk():
    client = FakeClient([msg("refusal")])
    with pytest.raises(RefusedError):
        run_desk("rates", client, today=TODAY)


def test_audit_flags_unknown_ids_and_uncited_paragraphs_but_skips_subheads():
    body = "Fact one [S1][S2].\n\n## Background\n\nA claim with no source.\n\nFact [S7]."
    a = audit(Draft("h", "d", body), {"S1", "S2"})
    assert a.cited == ["S1", "S2", "S7"]
    assert a.unknown == ["S7"]
    assert a.uncited == ["A claim with no source."]
    assert not a.ok


def test_markdown_export_lists_only_cited_sources():
    client = FakeClient([*reporter_turns(), article(), verdict("publish")])
    md = run_desk("rates", client, today=TODAY).markdown()
    assert md.startswith("# Rates held")
    assert "**[S1]**" in md and "**[S2]**" not in md
