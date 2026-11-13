"""The whole crew end to end, with Claude and YouTube replaced by fakes. No keys, no network."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from anthropic.lib._parse._transform import transform_schema

from trendwatch.config import Settings
from trendwatch.llm import AgentError, ClaudeLLM
from trendwatch.models import Trend, TrendReport, Video, VideoDigest
from trendwatch.pipeline import run, write_brief

NOW = datetime.now(timezone.utc)


def make_video(vid, channel):
    return Video(video_id=vid, channel_id="UC" + channel, channel_title=channel, title=f"Video {vid}",
                 published_at=NOW - timedelta(days=1), views=500, transcript=f"talk about agents in {vid}")


class FakeCollector:
    def collect(self, refs, on_progress):
        return [make_video("v1", "A"), make_video("v2", "B"), make_video("boom", "C")], []


class FakeLLM:
    """Same interface as ClaudeLLM. Records every call so tests can inspect the prompts."""

    def __init__(self):
        self.calls = []

    def structured(self, *, model, system, prompt, schema, effort="medium", **_):
        self.calls.append(SimpleNamespace(model=model, prompt=prompt, schema=schema, effort=effort))
        if schema is VideoDigest:
            if "video_id: boom" in prompt:
                raise AgentError("simulated refusal")
            return VideoDigest(video_id="WRONG", summary="s", topics=["AI Agents"], key_claims=["c"],
                               named_things=[], tone="excited", novelty="new")
        return TrendReport(
            headline="Agents everywhere",
            trends=[Trend(name="Agents", explanation="e", why_now="w", momentum="rising",
                          evidence_video_ids=["v1", "v2", "made-up-id"], channels=["A", "B"], confidence=0.7)],
            outliers=[], caveats=["tiny sample"],
        )

    def stream_text(self, *, prompt, **_):
        self.calls.append(SimpleNamespace(prompt=prompt, schema=None))
        yield "**Agents everywhere.** "
        yield "More soon."


def test_full_run_with_fakes():
    llm = FakeLLM()
    settings = Settings(lead_model="lead", worker_model="worker")
    result = run(["@a"], settings=settings, llm=llm, collector=FakeCollector(), on_progress=lambda _: None)

    # One analyst failed: the run continues and says so.
    assert [d.video_id for d in result.digests] == ["v1", "v2"]          # ids repaired, video order kept
    assert any("boom" in w for w in result.warnings)

    # Analysts use the worker model at low effort; the trend spotter the lead at high effort.
    analyst_calls = [c for c in llm.calls if c.schema is VideoDigest]
    trend_call = next(c for c in llm.calls if c.schema is TrendReport)
    assert {c.model for c in analyst_calls} == {"worker"} and {c.effort for c in analyst_calls} == {"low"}
    assert (trend_call.model, trend_call.effort) == ("lead", "high")

    # The trend spotter saw the counted signals, not raw transcripts.
    assert "<topic_signals>" in trend_call.prompt and "ai agents" in trend_call.prompt
    assert "talk about agents" not in trend_call.prompt

    # Hallucinated evidence id was dropped.
    assert result.report.trends[0].evidence_video_ids == ["v1", "v2"]

    brief = "".join(write_brief(result, "devs", settings=settings, llm=llm))
    assert brief.startswith("**Agents everywhere.**")
    assert "https://www.youtube.com/watch?v=v1" in llm.calls[-1].prompt   # editor gets real links


def test_no_videos_is_a_clear_error():
    class Empty:
        def collect(self, refs, on_progress):
            return [], ["Skipped '@x': 404"]

    with pytest.raises(AgentError, match="No recent videos"):
        run(["@x"], settings=Settings(), llm=FakeLLM(), collector=Empty(), on_progress=lambda _: None)


# ---- ClaudeLLM itself, against a fake Anthropic client --------------------------------------
class FakeMessages:
    def __init__(self, stop_reason="end_turn"):
        self.kwargs, self.stop_reason = None, stop_reason

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(stop_reason=self.stop_reason, stop_details=SimpleNamespace(category="cyber"),
                               parsed_output=kwargs["output_format"](headline="h", trends=[], outliers=[], caveats=[]))


def fake_client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def test_claude_llm_sends_effort_and_fallbacks_only_where_supported():
    msgs = FakeMessages()
    llm = ClaudeLLM(client=fake_client(msgs))
    llm.structured(model="claude-opus-5-5", system="s", prompt="p", schema=TrendReport, effort="high")
    assert msgs.kwargs["output_config"] == {"effort": "high"}
    assert msgs.kwargs["fallbacks"] == "default" and msgs.kwargs["betas"] == ["server-side-fallback-2026-07-01"]

    llm.structured(model="claude-haiku-5-5", system="s", prompt="p", schema=TrendReport)
    assert "fallbacks" not in msgs.kwargs and "betas" not in msgs.kwargs


def test_claude_llm_raises_on_refusal():
    llm = ClaudeLLM(client=fake_client(FakeMessages(stop_reason="refusal")))
    with pytest.raises(AgentError, match="declined"):
        llm.structured(model="claude-opus-5-5", system="s", prompt="p", schema=TrendReport)


@pytest.mark.parametrize("schema", [VideoDigest, TrendReport])
def test_schemas_are_accepted_by_the_sdk_transform(schema):
    # The SDK rewrites Pydantic schemas into the subset structured outputs supports.
    # If that rewrite chokes, we'd only find out with a paid call, so check it here.
    out = transform_schema(schema)
    assert out["type"] == "object" and out["additionalProperties"] is False
