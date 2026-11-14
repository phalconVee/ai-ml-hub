"""The whole crew offline: researcher → scriptwriter → lint + editor, with a fake Claude."""

import pytest

from podcast.crew import produce
from podcast.llm import RefusedError
from podcast.script import Line, Script, clean_text, lint

from .fakes import ARTICLE_HTML, OUTLINE, FakeClient, draft, edited, reply

URL = "https://blog.example/bees"


def run(client, **kw):
    return produce(URL, client, fetch=lambda u: ARTICLE_HTML, minutes=1, **kw)


def test_three_calls_in_order_with_the_right_settings():
    client = FakeClient([reply(OUTLINE), draft(), edited()])
    steps = []
    prod = run(client, names={"A": "Ada", "B": "Bo"}, style="playful", on_step=steps.append)

    researcher, writer, editor = client.requests
    # Every agent: the same model, structured JSON, server-side refusal fallback.
    for r in client.requests:
        assert r["model"] == "claude-opus-5-5"
        assert r["fallbacks"] == "default"
        assert r["output_config"]["format"]["type"] == "json_schema"
        assert "tools" not in r                      # nobody gets tools; the source is fetched in code
    assert [r["output_config"]["effort"] for r in client.requests] == ["medium", "high", "medium"]

    # The researcher reads the extracted page; the writer reads only the outline.
    assert "waggle dance & other bees" in researcher["messages"][0]["content"]
    w = writer["messages"][0]["content"]
    assert "Host A is Ada" in w and "playful" in w
    assert "Duration encodes distance" in w and "waggle dance & other" not in w
    assert "about 150 words" in w                    # 1 minute × 150 wpm

    # The editor sees the draft as "Name: text" lines and the lint to-do list.
    e = editor["messages"][0]["content"]
    assert "Ada: bee bee" in e and "AUTOMATIC CHECK" in e

    assert prod.outline.title == "Why Bees Dance"
    assert len(prod.script.lines) == 8 and not prod.edit.kept_draft
    assert prod.edit.changes == ["Tightened the opening."]
    assert len(steps) == 4 and steps[-1].startswith("Editor:")


def test_a_broken_edit_never_replaces_the_draft():
    # The editor returns 2 lines (below the 6-line minimum): keep the draft.
    client = FakeClient([reply(OUTLINE), draft(n=8), edited(n=2)])
    prod = run(client)
    assert prod.edit.kept_draft
    assert len(prod.script.lines) == 8


def test_final_script_is_cleaned_for_the_voice():
    rows = [{"speaker": "AB"[i % 2], "text": f"Line {i} [laughs] is *really* fine (pause)."}
            for i in range(6)] + [{"speaker": "A", "text": "[music]"}]
    client = FakeClient([reply(OUTLINE), draft(), reply({"title": "T", "lines": rows, "changes": []})])
    script = run(client).script
    assert script.lines[0].text == "Line 0 is really fine."
    assert len(script.lines) == 6                    # the "[music]"-only line vanished


def test_refusal_is_raised_before_any_further_call():
    client = FakeClient([reply({}, stop="refusal")])
    with pytest.raises(RefusedError, match="researcher"):
        run(client)
    assert len(client.requests) == 1


def test_truncated_json_is_reported_not_parsed():
    client = FakeClient([reply(OUTLINE), reply({}, stop="max_tokens")])
    with pytest.raises(RuntimeError, match="ran out of room"):
        run(client)


# ── lint and clean_text, the mechanical editor ──────────────────────────────

def test_lint_finds_mechanical_problems():
    long = " ".join(["word"] * 80)
    s = Script("t", [Line("A", long), Line("A", "Hi [laughs] there friend."), Line("A", "See https://x.y now."),
                     Line("B", "ok")])
    issues = " | ".join(lint(s, target_words=40))
    assert "at least 6" in issues
    assert "Line 1 is 80 words" in issues
    assert "Line 2 has a stage direction" in issues
    assert "Line 3 contains a URL" in issues
    assert "3+ lines in a row" in issues
    assert "Host B has only" in issues
    assert "target is about 40" in issues


def test_clean_text_keeps_words_and_drops_directions():
    assert clean_text("So (about 40 percent) of it, *really*.") == "So (about 40 percent) of it, really."
    assert clean_text("*sighs* Fine. (chuckles) [beat] Okay ,  sure") == "Fine. Okay, sure"
