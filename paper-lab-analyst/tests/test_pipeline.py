"""The whole workflow with a scripted fake Claude (no API key, no network)."""

import pytest

from conftest import FakeClient, reply, text_block, tool_use
from paperlab.config import SETTINGS
from paperlab.llm import AnalysisError
from paperlab.pipeline import analyse_pdf
from paperlab.report import to_markdown


def test_end_to_end(sample_pdf, fake_client):
    events = []
    result = analyse_pdf(sample_pdf, fake_client, on_event=events.append)

    # OCR ran on the scanned page and its text replaced the empty text layer.
    page3 = result.paper.pages[2]
    assert page3.ocr_done and page3.text.startswith("OCR TEXT")
    assert len(fake_client.calls_for("transcribe")) == 1

    # One vision call per figure, each with the image first.
    fig_calls = fake_client.calls_for("reading one figure")
    assert len(fig_calls) == 2
    assert fig_calls[0]["messages"][0]["content"][0]["type"] == "image"
    assert set(result.figure_notes) == {"F1", "F2"}

    # Analysts saw the OCR'd text and the figure notes.
    ctx = fake_client.calls_for("methods analyst")[0]["messages"][0]["content"]
    assert "OCR TEXT" in ctx and "Treatment C is highest" in ctx

    # Reviewer: tool results went back in one user message, matched by id,
    # and view_figure delivered an actual image.
    second = fake_client.calls_for("senior reviewer")[1]["messages"]
    assert second[1]["role"] == "assistant"
    results = second[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]
    assert results[1]["content"][0]["type"] == "image"

    # Grounding ran on the final report.
    assert (result.grounding["verified"], result.grounding["wrong_page"],
            result.grounding["not_found"]) == (1, 1, 1)
    assert any("reviewer → search_paper" in e for e in events)

    md = to_markdown(result)
    assert "## Key findings" in md and "✅" in md and "actually on p. 2" in md


def test_every_request_uses_the_configured_model_and_fallback(sample_pdf, fake_client):
    analyse_pdf(sample_pdf, fake_client)
    for call in fake_client.calls:
        assert call["model"] == SETTINGS.model == "claude-opus-5-5"
        assert call["output_config"]["effort"] == SETTINGS.effort
        assert call["fallbacks"] == "default" and call["betas"] == [SETTINGS.fallback_beta]
    # Everything except OCR asks for schema-shaped JSON.
    for call in fake_client.calls:
        has_schema = "format" in call["output_config"]
        assert has_schema == ("transcribe" not in call["system"])


def test_reviewer_is_cut_off_after_its_turn_budget(sample_pdf):
    import json
    from conftest import REPORT
    looping = reply(tool_use("tu_x", "search_paper", {"query": "x"}), stop="tool_use")
    script = [looping] * SETTINGS.max_reviewer_turns + [reply(text_block(json.dumps(REPORT)))]
    client = FakeClient(reviewer_script=script)
    result = analyse_pdf(sample_pdf, client)
    calls = client.calls_for("senior reviewer")
    assert len(calls) == SETTINGS.max_reviewer_turns + 1
    assert calls[-1]["tool_choice"] == {"type": "none"}            # no more tools allowed
    assert "budget used up" in calls[-1]["messages"][-1]["content"][-1]["text"]
    assert "tool_choice" not in calls[0]
    assert result.report["title"]


def test_refusal_becomes_a_readable_error(sample_pdf):
    client = FakeClient(reviewer_script=[reply(stop="refusal")])
    with pytest.raises(AnalysisError, match="declined"):
        analyse_pdf(sample_pdf, client)
