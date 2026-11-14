"""Text analysis that runs locally: no model download, no key, milliseconds.

    sentiment_timeline   VADER scores per time window -> a mood curve
    keywords             frequent content words and two-word phrases
    entities             capitalised names (people, places, products)
    extractive_summary   the most "central" sentences, kept in order
    search               exact, all-words, and fuzzy (typo-tolerant) matching

These are the *fallback* for summary/topics (Claude does those better when a
key is present) and the *only* implementation for sentiment and search, where
a deterministic, explainable method is actually what you want: a sentiment
curve should give the same numbers every time you ask.
"""

import difflib
import re
from collections import Counter

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from soundcheck.transcript import Segment, Transcript, fmt_ts

# VADER: a hand-built lexicon + rules (negation, "very", ALL CAPS, "!!").
# Built for short social text, which suits speech segments well. Load once.
_vader = SentimentIntensityAnalyzer()

# Words that carry no topic. Short, hand-picked list tuned for *speech*, which
# is full of fillers ("yeah", "gonna", "kind of") that written-text lists miss.
STOPWORDS = set("""
a about above after again against all also am an and any are aren't as at be because been before being below
between both but by can can't could couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs them themselves then there
there's these they they'd they'll they're they've this those through to too under until up very was wasn't we
we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why
why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
yeah yes okay ok oh uh um uhm hmm like just really actually basically literally gonna wanna gotta kind sort
thing things stuff lot lots know think mean right well get got go going went say said says see one two also
much many way even still back now get want need make made take come really something anything everything
people time little bit maybe sure good great will
""".split())

_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]+")


def _words(text: str) -> list[str]:
    return [w.lower().strip("'-") for w in _WORD.findall(text)]


# --- Sentiment over time ------------------------------------------------------


def _label(score: float) -> str:
    # VADER's own recommended cut-offs on the "compound" score (-1..+1).
    return "positive" if score >= 0.05 else "negative" if score <= -0.05 else "neutral"


def sentiment_timeline(t: Transcript, window_seconds: float = 30.0) -> dict:
    """Bucket the recording into fixed windows and score each one.

    Each segment gets a VADER compound score; a window's score is the average
    of its segments weighted by word count (a 40-word rant should outweigh a
    one-word "great"). Fixed windows make the curve easy to plot and compare.
    """
    window_seconds = max(5.0, float(window_seconds))
    scored = []
    for s in t.segments:
        n_words = max(1, len(s.text.split()))
        scored.append((s, _vader.polarity_scores(s.text)["compound"], n_words))

    points = []
    n_windows = max(1, int(t.duration // window_seconds) + (1 if t.duration % window_seconds else 0))
    for i in range(n_windows):
        lo, hi = i * window_seconds, (i + 1) * window_seconds
        # Assign a segment to the window its midpoint falls in: no double counting.
        inside = [(s, c, w) for s, c, w in scored if lo <= (s.start + s.end) / 2 < hi]
        if not inside:
            continue  # silence: no point, rather than a misleading 0.0 "neutral"
        total_w = sum(w for _, _, w in inside)
        score = sum(c * w for _, c, w in inside) / total_w
        points.append({
            "start": round(lo, 1), "end": round(min(hi, t.duration), 1),
            "label_time": fmt_ts(lo), "score": round(score, 3), "mood": _label(score),
            "excerpt": " ".join(s.text for s, _, _ in inside)[:160],
            "speakers": sorted({s.speaker for s, _, _ in inside if s.speaker}),
        })

    overall = (sum(c * w for _, c, w in scored) / sum(w for _, _, w in scored)) if scored else 0.0
    by_seg = sorted(scored, key=lambda x: x[1])
    moments = lambda items: [{"time": fmt_ts(s.start), "score": round(c, 3), "text": s.text} for s, c, _ in items]
    result = {
        "method": "VADER lexicon, word-weighted per window",
        "window_seconds": window_seconds,
        "overall": {"score": round(overall, 3), "mood": _label(overall)},
        "points": points,
        "most_negative": moments([x for x in by_seg[:3] if x[1] < 0]),
        "most_positive": moments([x for x in by_seg[::-1][:3] if x[1] > 0]),
    }
    # Per-speaker averages are a cheap, surprisingly useful extra.
    if any(s.speaker for s in t.segments):
        per: dict[str, list[tuple[float, int]]] = {}
        for s, c, w in scored:
            per.setdefault(s.speaker or "unknown", []).append((c, w))
        result["by_speaker"] = {
            k: round(sum(c * w for c, w in v) / sum(w for _, w in v), 3) for k, v in per.items()
        }
    return result


# --- Keywords and entities ------------------------------------------------------


def keywords(t: Transcript, top: int = 15) -> list[dict]:
    """Frequent content words and two-word phrases, with where they first appear.

    A phrase counts only if both words are content words ("machine learning",
    not "the machine"). Phrases get a small boost because "interest rates" says
    more than "rates" on its own.
    """
    counts: Counter = Counter()
    first_seen: dict[str, float] = {}
    for s in t.segments:
        ws = _words(s.text)
        content = [w for w in ws if w not in STOPWORDS and len(w) > 2]
        grams = content + [f"{a} {b}" for a, b in zip(ws, ws[1:])
                           if a not in STOPWORDS and b not in STOPWORDS and len(a) > 2 and len(b) > 2]
        for g in grams:
            counts[g] += 1
            first_seen.setdefault(g, s.start)
    # Something said once is noise, not a theme: require two mentions.
    scored = {g: c * (1.5 if " " in g else 1.0) for g, c in counts.items() if c >= 2}
    if not scored:  # very short clips: fall back to anything that appeared
        scored = dict(counts)
    best = sorted(scored, key=lambda g: (-scored[g], first_seen[g]))[:top]
    return [{"term": g, "count": counts[g], "first_at": fmt_ts(first_seen[g])} for g in best]


_NAME = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+(?:of|de|van|von|the)?\s*[A-Z][a-zA-Z]+)*)\b")


def entities(t: Transcript, top: int = 20) -> list[dict]:
    """Capitalised word runs that are not just sentence starts.

    Whisper capitalises proper nouns and sentence starts. To tell them apart:
    a single word that *also* appears lower-case somewhere ("So", "Today") is a
    sentence start; one that never does ("Lagos", "Kubernetes") is a name.
    A single word that *only* ever shows up at the start of a sentence
    ("Welcome", "Thanks") is dropped too: capitalisation there proves nothing.
    Multi-word runs ("Bank of England") are kept as they are.
    """
    lower_vocab = set(w for w in _WORD.findall(t.text) if w.islower())
    counts: Counter = Counter()
    first_seen: dict[str, float] = {}
    mid_sentence: set[str] = set()  # names seen at least once NOT starting a sentence
    for s in t.segments:
        for m in _NAME.finditer(s.text):
            before = s.text[: m.start()].rstrip()
            starts_sentence = not before or before[-1] in ".!?"
            # "Then Maria Lopez" -> "Maria Lopez": peel off a capitalised
            # sentence-start word glued to the front of a real name.
            words = m.group(1).split()
            while len(words) > 1 and (words[0].lower() in STOPWORDS or words[0].lower() in lower_vocab):
                words.pop(0)
            name = " ".join(words)
            if " " not in name and (name.lower() in lower_vocab or name.lower() in STOPWORDS or name == "I"):
                continue
            counts[name] += 1
            first_seen.setdefault(name, s.start)
            if not starts_sentence or len(words) < len(m.group(1).split()):
                mid_sentence.add(name)
    keep = [(n, c) for n, c in counts.most_common() if " " in n or n in mid_sentence][:top]
    return [{"name": n, "mentions": c, "first_at": fmt_ts(first_seen[n])} for n, c in keep]


# --- Extractive summary -----------------------------------------------------------


def extractive_summary(t: Transcript, sentences: int = 6) -> list[dict]:
    """Pick the segments that mention the recording's main words most densely.

    Score = sum of keyword frequencies in the segment / sqrt(length). The sqrt
    stops long segments winning just by being long. The winners are returned
    in *time order* so the summary still reads like the conversation did.
    """
    freq = Counter(w for w in _words(t.text) if w not in STOPWORDS and len(w) > 2)
    if not freq:
        return []

    def score(s: Segment) -> float:
        ws = [w for w in _words(s.text) if w not in STOPWORDS and len(w) > 2]
        return sum(freq[w] for w in ws) / (len(_words(s.text)) ** 0.5 or 1)

    candidates = [s for s in t.segments if len(s.text.split()) >= 5] or t.segments
    best = sorted(candidates, key=score, reverse=True)[:sentences]
    return [{"time": fmt_ts(s.start), "speaker": s.speaker, "text": s.text}
            for s in sorted(best, key=lambda s: s.start)]


# --- Search -----------------------------------------------------------------------


def search(t: Transcript, query: str, max_results: int = 10, context: int = 1) -> dict:
    """Find where something was said. Three passes, strictest first:

      exact    the phrase appears verbatim (case-insensitive)
      words    every query word appears in the segment, in any order
      fuzzy    a word close in spelling (Whisper writes "Kubernetis" for
               "Kubernetes"; difflib's ratio catches that)

    Each hit carries its timestamp, speaker and the neighbouring segments, so
    the agent can quote it in context without fetching the whole transcript.
    """
    q = query.strip().lower()
    if not q:
        return {"query": query, "hits": []}
    q_words = [w for w in _words(q) if w not in STOPWORDS] or _words(q)
    segs = t.segments
    hits: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(i: int, how: str):
        if i not in seen:
            seen.add(i)
            hits.append((i, how))

    for i, s in enumerate(segs):
        if q in s.text.lower():
            add(i, "exact")
    for i, s in enumerate(segs):
        if q_words and all(w in _words(s.text) for w in q_words):
            add(i, "all words")
    for i, s in enumerate(segs):
        seg_words = set(_words(s.text))
        if q_words and all(difflib.get_close_matches(w, seg_words, n=1, cutoff=0.8) for w in q_words):
            add(i, "fuzzy")

    out = []
    for i, how in hits[:max_results]:
        s = segs[i]
        lo, hi = max(0, i - context), min(len(segs), i + context + 1)
        out.append({
            "time": fmt_ts(s.start), "start": s.start, "end": s.end, "speaker": s.speaker,
            "match": how, "text": s.text,
            "context": t.render(segs[lo:hi]),
        })
    return {"query": query, "total_hits": len(hits), "hits": out}
