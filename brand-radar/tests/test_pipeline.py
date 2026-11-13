"""The whole radar end to end, offline: fake sources + a scripted fake Claude.

Nothing touches the network or needs a key, but every real piece runs: the
collectors' fan-out and error isolation, prep, the parallel analysts and their
id checks, metrics, the strategist prompt and the citation check.
"""

from radar import collectors, pipeline
from radar.schemas import AnalystOutput, BrandQuery, Mention, Verdict

Q = BrandQuery("Arc", context="web browser")


async def fake_news(q, n):
    return [Mention("news", "Arc raises prices", "https://n.test/1", "Arc raises its subscription price"),
            Mention("news", "Arc welding tips", "https://n.test/2", "Arc is a welding method")]


async def fake_bluesky(q, n):
    return [Mention("bluesky", "", "https://b.test/1", "Arc crashed three times today, unusable", engagement=120)]


async def broken_source(q, n):
    raise RuntimeError("HTTP 503")


class FakeLLM:
    """Plays Claude: judges by keyword, and records what it was asked."""

    def __init__(self):
        self.analyst_prompts: list[str] = []
        self.strategist_prompt = ""

    async def structured(self, *, prompt, schema, **_):
        assert schema is AnalystOutput
        self.analyst_prompts.append(prompt)
        verdicts = []
        for id_ in ("news-1", "news-2", "bluesky-1"):
            if f'"{id_}"' not in prompt:
                continue
            if id_ == "news-2":
                verdicts.append(Verdict(id=id_, relevant=False, sentiment="neutral", themes=[],
                                        action="none", urgency=0, summary="welding, not the browser"))
            else:
                verdicts.append(Verdict(id=id_, relevant=True, sentiment="negative",
                                        themes=["pricing"] if id_ == "news-1" else ["stability"],
                                        action="fix" if id_ == "bluesky-1" else "respond",
                                        urgency=3 if id_ == "bluesky-1" else 2, summary="complaint"))
        verdicts.append(Verdict(id="made-up-7", relevant=True, sentiment="positive", themes=[],
                                action="none", urgency=0, summary="hallucinated"))
        return AnalystOutput(verdicts=verdicts, source_summary="Mostly complaints.")

    async def stream_text(self, *, prompt, on_text=None, **_):
        self.strategist_prompt = prompt
        text = "## Verdict\nFix crashes first [bluesky-1], then pricing [news-1]. Also [news-42]."
        for chunk in text.split(" "):
            on_text and on_text(chunk + " ")
        return text

    async def web_search(self, **_):
        raise AssertionError("sweep is off; web_search must not be called")


async def test_full_run_offline(monkeypatch):
    monkeypatch.setattr(collectors, "COLLECTORS",
                        {"news": fake_news, "bluesky": fake_bluesky, "reddit": broken_source})
    llm, events, streamed = FakeLLM(), [], []

    run = await pipeline.run_radar(Q, ["news", "bluesky", "reddit"], llm=llm,
                                   on_event=lambda s, msg: events.append(s), on_report_text=streamed.append)

    # One analyst per source that had mentions; the broken source got none.
    assert len(llm.analyst_prompts) == 2
    assert any("Bluesky" in p for p in llm.analyst_prompts)
    # The failure is isolated and reported, not fatal.
    assert run.sources["reddit"].error.startswith("RuntimeError")
    # The namesake (welding) is collected but not counted.
    assert run.metrics["collected"] == 3 and run.metrics["relevant"] == 2
    assert run.metrics["net_sentiment"] == -100
    assert run.metrics["actions"] == {"fix": 1, "respond": 1}
    # The hallucinated verdict id never makes it into the results.
    assert all("made-up-7" not in r.verdicts for r in run.sources.values())
    # The strategist saw the numbers, the flagged mentions (most urgent first) and the failure note.
    p = llm.strategist_prompt
    assert '"net_sentiment": -100' in p and "reddit: RuntimeError" in p
    assert p.index('"bluesky-1"') < p.index('"news-1"')
    # The report streamed, and its bogus citation was caught.
    assert "".join(streamed).strip() == run.report_md
    assert any("news-42" in w for w in run.warnings)
    assert any("made-up-7" in w for w in run.warnings)
    assert events[-1] == "done"


async def test_no_relevant_mentions_skips_the_report(monkeypatch):
    async def nothing(q, n):
        return [Mention("news", "", "https://x.test", "the notion of arc lengths")]  # passes the word filter; only the analyst can tell it is maths
    monkeypatch.setattr(collectors, "COLLECTORS", {"news": nothing})

    class NeverRelevant(FakeLLM):
        async def structured(self, *, prompt, schema, **_):
            return AnalystOutput(verdicts=[Verdict(id="news-1", relevant=False, sentiment="neutral", themes=[],
                                                   action="none", urgency=0, summary="math")], source_summary="")

        async def stream_text(self, **_):
            raise AssertionError("no report should be written when nothing is relevant")

    run = await pipeline.run_radar(Q, ["news"], llm=NeverRelevant())
    assert run.metrics["relevant"] == 0 and "No relevant mentions" in run.report_md


async def test_sweep_failure_does_not_sink_the_run(monkeypatch):
    monkeypatch.setattr(collectors, "COLLECTORS", {"news": fake_news})
    run = await pipeline.run_radar(Q, ["news"], llm=FakeLLM(), use_sweep=True)  # FakeLLM.web_search raises
    assert "claude_sweep" in run.sources and run.sources["claude_sweep"].error
    assert run.metrics["relevant"] == 1


async def test_all_analysts_failing_is_reported_not_hidden(monkeypatch):
    monkeypatch.setattr(collectors, "COLLECTORS", {"news": fake_news, "bluesky": fake_bluesky})

    class NoKey(FakeLLM):
        async def structured(self, **_):
            raise TypeError("Could not resolve authentication method")

    run = await pipeline.run_radar(Q, ["news", "bluesky"], llm=NoKey())
    assert "Analysis failed for every source" in run.report_md and "authentication" in run.report_md
