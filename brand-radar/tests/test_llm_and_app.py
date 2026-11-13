"""The Claude wrapper's request shapes, and a smoke run of the Streamlit app.

The wrapper test uses a fake SDK client that records each request, so we can
check what WOULD be sent (model, effort, fallback beta, web-search tool) and
that a `pause_turn` gets resumed. No network, no key.
"""

from types import SimpleNamespace as NS

import pytest
from streamlit.testing.v1 import AppTest

from radar import llm as llm_mod
from radar.pipeline import RadarRun
from radar.schemas import BrandQuery, Mention, SourceReport, Verdict


class FakeMessages:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def fake_client(*replies):
    msgs = FakeMessages(replies)
    return NS(beta=NS(messages=msgs)), msgs


async def test_web_search_resumes_after_pause_turn():
    first = NS(stop_reason="pause_turn", content=[NS(type="server_tool_use")])
    second = NS(stop_reason="end_turn", content=[NS(type="text", text="done", citations=[])])
    client, msgs = fake_client(first, second)
    blocks = await llm_mod.LLM(client).web_search(model="m", effort="low", system="s", prompt="p", max_searches=3)

    assert len(blocks) == 2 and len(msgs.calls) == 2
    req = msgs.calls[1]
    assert req["tools"][0]["type"].startswith("web_search_") and req["tools"][0]["max_uses"] == 3
    # Resuming = the paused assistant turn sent back as-is, no extra user message.
    assert req["messages"][-1] == {"role": "assistant", "content": first.content}
    assert req["output_config"] == {"effort": "low"} and req["fallbacks"] == "default"


async def test_structured_returns_parsed_output_and_raises_on_refusal():
    client, msgs = fake_client(NS(stop_reason="end_turn", parsed_output="OK"), NS(stop_reason="refusal"))
    llm = llm_mod.LLM(client)
    assert await llm.structured(model="m", effort="low", system="s", prompt="p", schema=object) == "OK"
    assert msgs.calls[0]["output_format"] is object
    with pytest.raises(llm_mod.RefusalError):
        await llm.structured(model="m", effort="low", system="s", prompt="p", schema=object)


def test_app_renders_a_finished_run(monkeypatch):
    """Run the real app.py with run_radar swapped for a canned result."""
    mention = Mention("bluesky", "t", "https://b.test/1", "Arc crashed", engagement=5, id="bluesky-1")
    verdict = Verdict(id="bluesky-1", relevant=True, sentiment="negative", themes=["stability"],
                      action="fix", urgency=3, summary="Crash report")
    canned = RadarRun(
        BrandQuery("Arc"),
        {"bluesky": SourceReport("bluesky", [mention], {"bluesky-1": verdict}, "Crashes."),
         "reddit": SourceReport("reddit", [], {}, error="RuntimeError: 503")},
        {"collected": 1, "relevant": 1, "sentiment": {"negative": 1}, "net_sentiment": -100,
         "per_source": {"bluesky": {"collected": 1, "relevant": 1, "sentiment": {"negative": 1}, "net_sentiment": -100, "error": None},
                        "reddit": {"collected": 0, "relevant": 0, "sentiment": {}, "net_sentiment": None, "error": "503"}},
         "top_themes": [{"theme": "stability", "count": 1, "sentiment": {"negative": 1}}], "actions": {"fix": 1}},
        "## Verdict\nFix crashes [bluesky-1].",
        ["reddit: collection failed"],
    )

    async def fake_run(q, sources, **kw):
        kw["on_event"]("collect", "searching")
        return canned

    import radar.pipeline
    monkeypatch.setattr(radar.pipeline, "run_radar", fake_run)

    at = AppTest.from_file("../app.py", default_timeout=30).run()
    assert not at.exception
    at.sidebar.text_input[0].set_value("Arc").run()   # typing reruns the script, enabling the button
    at.sidebar.button[0].click().run()

    assert not at.exception
    assert [m.value for m in at.metric] == ["1", "1", "-100", "1"]
    assert any("Fix crashes" in md.value for md in at.markdown)
    assert any("reddit" in w.value for w in at.warning)
    assert len(at.tabs) == 2          # bluesky (has mentions) + reddit (has an error)
