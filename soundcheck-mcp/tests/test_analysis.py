"""Local analysis: sentiment curve, keywords, entities, summary, search."""

from soundcheck import analysis
from soundcheck.transcript import fmt_ts


def test_fmt_ts():
    assert fmt_ts(0) == "00:00"
    assert fmt_ts(75.4) == "01:15"
    assert fmt_ts(3725) == "1:02:05"


def test_sentiment_curve_dips_then_recovers(sample_transcript):
    s = analysis.sentiment_timeline(sample_transcript, window_seconds=15)
    scores = [p["score"] for p in s["points"]]
    assert len(scores) == 3
    # 0-15 s mixes praise with "disaster"; 15-30 s has the outage and the fix;
    # the story's low point must be one of the windows before the end.
    assert min(scores) < scores[-1] or s["most_negative"]
    assert any("outage" in m["text"] for m in s["most_negative"])
    assert any("fantastic" in m["text"] or "wonderful" in m["text"] for m in s["most_positive"])
    assert "by_speaker" not in s  # no labels yet


def test_sentiment_by_speaker_when_labelled(sample_transcript):
    for i, seg in enumerate(sample_transcript.segments):
        seg.speaker = f"Speaker {i % 2 + 1}"
    s = analysis.sentiment_timeline(sample_transcript)
    assert set(s["by_speaker"]) == {"Speaker 1", "Speaker 2"}


def test_keywords_and_entities(sample_transcript):
    terms = [k["term"] for k in analysis.keywords(sample_transcript)]
    assert terms[0] == "kubernetes"  # said four times
    assert "deployment pipeline" in terms or "platform team" in terms
    names = {e["name"] for e in analysis.entities(sample_transcript)}
    assert {"Kubernetes", "Acme Corp", "Maria Lopez"} <= names
    assert not {"Welcome", "Today", "Thanks", "Last"} & names  # sentence starts aren't names


def test_extractive_summary_is_in_time_order(sample_transcript):
    picks = analysis.extractive_summary(sample_transcript, sentences=3)
    assert len(picks) == 3
    assert [p["time"] for p in picks] == sorted(p["time"] for p in picks)


def test_search_exact_words_and_fuzzy(sample_transcript):
    exact = analysis.search(sample_transcript, "deployment pipeline")
    assert exact["hits"][0]["match"] == "exact" and exact["hits"][0]["time"] == "00:21"
    words = analysis.search(sample_transcript, "outage customers")
    assert words["hits"][0]["match"] == "all words"
    fuzzy = analysis.search(sample_transcript, "Kubernetis")  # typical transcription misspelling
    assert fuzzy["total_hits"] == 4 and fuzzy["hits"][0]["match"] == "fuzzy"
    assert "[00:" in fuzzy["hits"][0]["context"]
    assert analysis.search(sample_transcript, "giraffe")["hits"] == []
