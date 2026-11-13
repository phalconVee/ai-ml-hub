"""The three kinds of agent. Each is one prompt + one function.

  analyst     one per source: reads that source's mentions, returns a Verdict per mention
  sweeper     optional: Claude web-searches for mentions the free sources missed
  strategist  reads the numbers + every analyst's findings, writes the report

Why one analyst PER SOURCE instead of one big call? A tweet-length Bluesky post,
a Hacker News comment and a news headline need different reading habits
("sarcasm is common here", "a headline is not the writer's opinion"). Giving
each analyst a source-specific briefing, and only that source's mentions, keeps
every prompt small and focused, and lets the sources run in parallel.

The agents never trust Claude blindly. `check_verdicts` throws out any verdict
for an id we didn't send (hallucinated) and reports which mentions got no
verdict (skipped). The strategist's citations are checked the same way later.
"""

import asyncio
import json

from radar.config import CLAUDE_SWEEP, SOURCES, settings
from radar.schemas import AnalystOutput, BrandQuery, Mention, Verdict

# --- Analyst -------------------------------------------------------------------

# Reading tips per source. This is the "specialist" part of each analyst.
SOURCE_TIPS: dict[str, str] = {
    "news": "Headlines report events; judge sentiment by what the event means for the brand, not the writer's tone.",
    "web": "Mixed bag: vendor pages, blogs, listicles, docs. Marketing copy by the brand itself is neutral, action 'none'.",
    "reddit": "You see only the thread title and a search snippet. Users vent and joke; questions often hide complaints.",
    "hackernews": "Technical crowd; terse and often critical. Sarcasm is common. A comment may mention the brand only in passing.",
    "bluesky": "Short social posts. Sarcasm and jokes are common. Many posts mention a namesake, so check relevance hard.",
    "mastodon": "Posts tagged with the brand's hashtag; the tag is sometimes used loosely or for an unrelated meaning.",
    "youtube": "Video titles and descriptions; reviews, tutorials and comparisons. Tutorials are usually neutral-to-positive.",
    CLAUDE_SWEEP: "Quoted passages Claude found via web search; each is a direct excerpt from the cited page.",
}

ANALYST_SYSTEM = """You are a brand-monitoring analyst. You read mentions of one brand \
from one source and judge each one, for a team that has to decide what to act on.

Rules:
- Return exactly one verdict per mention, using its id exactly as given.
- relevant=false if the text is about a namesake (another company, a common word, a person). \
For irrelevant mentions use sentiment "neutral", action "none", urgency 0, themes [].
- Sentiment is toward the BRAND, not the topic. "X is down again" is negative even if phrased as a joke.
- Urgency: 3 = act today (outage, security, viral complaint, legal/PR risk); 2 = this week; \
1 = worth knowing; 0 = ignore.
- Themes: short lowercase labels reused across mentions where they fit (e.g. "pricing", "outage", \
"customer support", "ai features", "competitor comparison").
- Judge only from the text given. Do not invent facts."""


def analyst_prompt(q: BrandQuery, source: str, batch: list[Mention]) -> str:
    """Everything one analyst call needs: who the brand is, how to read this
    source, and the mentions as compact JSON."""
    items = [{"id": m.id, "title": m.title, "text": m.text, "author": m.author,
              "published": m.published, "engagement": m.engagement} for m in batch]
    return (
        f"Brand: {q.brand}\n"
        f"What the brand is: {q.context or '(not given: infer from the texts, and be strict about namesakes)'}\n"
        f"Also known as: {', '.join(q.aliases) or '(none)'}\n\n"
        f"Source: {SOURCES.get(source, source)}\n"
        f"How to read this source: {SOURCE_TIPS.get(source, '')}\n\n"
        f"Mentions ({len(batch)}):\n{json.dumps(items, ensure_ascii=False, indent=1)}"
    )


def check_verdicts(batch: list[Mention], output: AnalystOutput) -> tuple[dict[str, Verdict], list[str]]:
    """Keep verdicts whose id we actually sent; list the ids that got none.

    Returns (verdicts by id, problems). A verdict for an unknown id is dropped:
    if we kept it, the report could cite a mention that doesn't exist.
    """
    sent = {m.id for m in batch}
    verdicts: dict[str, Verdict] = {}
    problems: list[str] = []
    for v in output.verdicts:
        if v.id not in sent:
            problems.append(f"dropped verdict for unknown id {v.id!r}")
        elif v.id not in verdicts:            # first verdict wins on duplicates
            v.urgency = max(0, min(3, v.urgency))   # clamp: the schema can't enforce a range
            verdicts[v.id] = v
    missing = sorted(sent - verdicts.keys())
    if missing:
        problems.append(f"no verdict for {', '.join(missing)}")
    return verdicts, problems


async def run_analyst(llm, q: BrandQuery, source: str, mentions: list[Mention],
                      gate: asyncio.Semaphore) -> tuple[dict[str, Verdict], str, list[str]]:
    """Analyse one source. Big sources are split into batches that run in parallel;
    `gate` (a semaphore shared by all analysts) caps how many calls are in flight."""
    batches = [mentions[i:i + settings.batch_size] for i in range(0, len(mentions), settings.batch_size)]

    async def one(batch: list[Mention]) -> AnalystOutput:
        async with gate:
            return await llm.structured(
                model=settings.analyst_model, effort=settings.analyst_effort,
                system=ANALYST_SYSTEM, prompt=analyst_prompt(q, source, batch), schema=AnalystOutput,
            )

    outputs = await asyncio.gather(*(one(b) for b in batches))
    verdicts: dict[str, Verdict] = {}
    problems: list[str] = []
    for batch, out in zip(batches, outputs):
        v, p = check_verdicts(batch, out)
        verdicts.update(v)
        problems += p
    # Several batches -> several summaries; joining them is good enough here
    # because the strategist reads them all anyway.
    summary = " ".join(o.source_summary.strip() for o in outputs)
    return verdicts, summary, problems


# --- Sweeper (optional) ---------------------------------------------------------

SWEEP_SYSTEM = """You find recent public mentions of a brand on the web: reviews, complaints, \
forum threads, news, social posts. Search several angles (complaints, reviews, outage, pricing, \
alternatives). Then list each distinct mention you found as a short bullet that quotes or \
closely paraphrases it, citing the page it came from. Skip the brand's own marketing pages."""


def mentions_from_search(blocks: list) -> list[Mention]:
    """Turn the sweep's cited answer into Mentions.

    Each citation on the answer text carries the exact passage Claude relied on
    (`cited_text`) plus the page's URL and title, which is exactly the shape of
    a mention. Pages that were searched but never cited are ignored: we only
    have their titles, which is too little to judge sentiment on.
    """
    by_url: dict[str, Mention] = {}
    for b in blocks:
        if getattr(b, "type", "") != "text":
            continue
        for c in getattr(b, "citations", None) or []:
            url = getattr(c, "url", None)
            cited = (getattr(c, "cited_text", "") or "").strip()
            if not url or not cited:
                continue
            if url in by_url:
                if cited not in by_url[url].text:
                    by_url[url].text += " ... " + cited      # several quotes, same page
            else:
                by_url[url] = Mention(CLAUDE_SWEEP, getattr(c, "title", "") or url, url, cited)
    return list(by_url.values())


async def run_sweep(llm, q: BrandQuery) -> list[Mention]:
    prompt = (f"Brand: {q.brand} ({q.context or 'infer what it is'}). "
              f"Find mentions from the last {q.days} days, newest first.")
    blocks = await llm.web_search(model=settings.sweep_model, effort="low", system=SWEEP_SYSTEM,
                                  prompt=prompt, max_searches=settings.sweep_max_searches)
    return mentions_from_search(blocks)


# --- Strategist -----------------------------------------------------------------

STRATEGIST_SYSTEM = """You are the head of brand insights. You turn a monitoring run into a \
short, actionable report in Markdown for a busy marketing + product + support team.

Structure, with these exact headings:
## Verdict           (2-3 sentences: overall health, the single biggest story)
## What people are saying   (key themes with sentiment, which sources they come from)
## Action queue      (a numbered list, most urgent first. Each item: what to do, who should own it \
(Support / Product / Comms / Legal / Marketing), and the mention ids that justify it)
## Source by source  (one line per source)
## Watch list        (weak signals worth checking again next run)

Rules:
- Cite evidence with mention ids in square brackets, e.g. [reddit-2][news-5]. Use ONLY ids that \
appear in the data. Never invent ids, numbers, quotes or facts.
- Use the computed metrics as given; do not recompute or round differently.
- If the data is thin (few relevant mentions), say so plainly instead of over-interpreting."""


def strategist_prompt(q: BrandQuery, metrics: dict, summaries: dict[str, str],
                      flagged: list[dict], notes: list[str]) -> str:
    return (
        f"Brand: {q.brand} ({q.context or 'context not given'}), last {q.days} days.\n\n"
        f"Computed metrics (JSON):\n{json.dumps(metrics, indent=1)}\n\n"
        f"Analyst summaries per source:\n" + "\n".join(f"- {s}: {t}" for s, t in summaries.items()) + "\n\n"
        f"Relevant mentions that need attention or show strong sentiment (JSON):\n"
        f"{json.dumps(flagged, ensure_ascii=False, indent=1)}\n\n"
        f"Run notes (sources that failed or returned nothing):\n" + ("\n".join(notes) or "(none)")
    )


async def run_strategist(llm, q: BrandQuery, metrics: dict, summaries: dict[str, str],
                         flagged: list[dict], notes: list[str], on_text=None) -> str:
    return await llm.stream_text(
        model=settings.strategist_model, effort=settings.strategist_effort,
        system=STRATEGIST_SYSTEM, prompt=strategist_prompt(q, metrics, summaries, flagged, notes),
        on_text=on_text,
    )
