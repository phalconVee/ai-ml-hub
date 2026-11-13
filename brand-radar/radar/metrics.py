"""Turn verdicts into numbers. Plain Python, deliberately.

The LLM judges each mention; arithmetic is done here, never by the LLM. That way
"62% positive" in the report is a count you can check, not a vibe Claude
produced. The strategist receives these numbers and is told to use them as-is.

Also here: `check_citations`, which verifies the strategist only cited mention
ids that exist.
"""

import re
from collections import Counter

from radar.schemas import SourceReport

SENTIMENT_WEIGHT = {"positive": 1, "negative": -1, "neutral": 0, "mixed": 0}


def net_sentiment(sentiments: list[str]) -> int | None:
    """(positives - negatives) / all, as a whole number from -100 to +100.

    Like a Net Promoter Score for mentions: +100 = all positive, -100 = all
    negative, 0 = balanced or all neutral. None when there's nothing to score.
    """
    if not sentiments:
        return None
    return round(100 * sum(SENTIMENT_WEIGHT[s] for s in sentiments) / len(sentiments))


def compute(sources: dict[str, SourceReport]) -> dict:
    """Headline numbers, per-source breakdown, top themes and action counts."""
    per_source: dict[str, dict] = {}
    all_sent: list[str] = []
    themes: Counter = Counter()
    theme_sent: dict[str, Counter] = {}
    actions: Counter = Counter()
    collected = 0

    for name, rep in sources.items():
        collected += len(rep.mentions)
        relevant = [v for v in rep.verdicts.values() if v.relevant]
        sents = [v.sentiment for v in relevant]
        all_sent += sents
        for v in relevant:
            for t in v.themes:
                t = t.strip().lower()
                themes[t] += 1
                theme_sent.setdefault(t, Counter())[v.sentiment] += 1
            if v.action != "none":
                actions[v.action] += 1
        per_source[name] = {
            "collected": len(rep.mentions),
            "relevant": len(relevant),
            "sentiment": dict(Counter(sents)),
            "net_sentiment": net_sentiment(sents),
            "error": rep.error or None,
        }

    return {
        "collected": collected,
        "relevant": len(all_sent),
        "sentiment": dict(Counter(all_sent)),
        "net_sentiment": net_sentiment(all_sent),
        "per_source": per_source,
        "top_themes": [
            {"theme": t, "count": n, "sentiment": dict(theme_sent[t])} for t, n in themes.most_common(12)
        ],
        "actions": dict(actions),
    }


def flagged_mentions(sources: dict[str, SourceReport], limit: int = 40) -> list[dict]:
    """The mentions the strategist should read in full: relevant ones that need
    action, or carry clear sentiment. Ranked by urgency, then reach."""
    rows = []
    for rep in sources.values():
        for m in rep.mentions:
            v = rep.verdicts.get(m.id)
            if not v or not v.relevant:
                continue
            if v.action == "none" and v.sentiment == "neutral":
                continue
            rows.append({
                "id": m.id, "source": m.source, "url": m.url, "title": m.title[:120],
                "sentiment": v.sentiment, "action": v.action, "urgency": v.urgency,
                "themes": v.themes, "summary": v.summary, "engagement": m.engagement,
            })
    rows.sort(key=lambda r: (-r["urgency"], -r["engagement"]))
    return rows[:limit]


def check_citations(report_md: str, known_ids: set[str]) -> list[str]:
    """Ids the report cites in [brackets] that don't exist. Empty list = all good."""
    cited = set(re.findall(r"\[([a-z_]+-\d+)\]", report_md))
    return sorted(cited - known_ids)
