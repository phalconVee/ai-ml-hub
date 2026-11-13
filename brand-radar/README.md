# Brand Radar

A multi-agent **brand-monitoring** app. Type a brand name and it sweeps seven free sources for recent mentions: news, the web, Reddit, Hacker News, Bluesky, Mastodon and YouTube. A specialist Claude **analyst per source** judges every mention: is it really about this brand, what's the sentiment, what themes, does someone need to act? A **strategist** then turns that into a short report with a prioritised action queue. Every recommendation cites the mentions behind it.

The only key it needs is `ANTHROPIC_API_KEY`. No scraping service and no social-media API keys. An optional extra source uses Claude's own server-side web search, billed on the same key.

## The idea in one paragraph

Brand monitoring is mostly a **data problem with a judgment step in the middle**. So the shape is: plain Python fetches and cleans (cheap, deterministic), Claude judges (one focused analyst per source, all running in parallel), plain Python counts (so "net sentiment −40" is arithmetic you can check, not a number the model made up), and Claude writes (the strategist). Two checks keep the LLM honest: analyst verdicts for mention ids we never sent are thrown away, and any `[id]` the report cites that doesn't exist is flagged.

## Architecture

```
  Streamlit (app.py)   brand · "what is it?" · aliases · exclusions · days · sources
        │
        ▼
  pipeline.run_radar()
        │
        ├─ 1. COLLECT (concurrently; a failing source can't sink the run)        collectors.py
        │     news · web · reddit · youtube ── DuckDuckGo (ddgs, keyless)
        │     hackernews ── Algolia API     bluesky ── public AppView search
        │     mastodon ── public hashtag timeline
        │     [optional] claude_sweep ── Claude + web_search server tool         agents.run_sweep
        │
        ├─ 2. CLEAN: brand must appear as a whole word, no excluded phrase,       prep.py
        │           de-dupe across sources (URL + opening words), id → "news-3"
        │
        ├─ 3. ANALYSE: one analyst agent per source, in parallel (semaphore)      agents.run_analyst
        │     Sonnet 5.5, structured output → Verdict per mention
        │     check_verdicts(): drop unknown ids, list skipped ones
        │
        ├─ 4. COUNT: net sentiment, per-source split, themes, action counts      metrics.py
        │
        ├─ 5. REPORT: strategist (Opus 5.5, high effort), streamed into the UI   agents.run_strategist
        │
        └─ 6. VERIFY: every [id] cited in the report must exist                  metrics.check_citations
```

## Code map: read in this order

| File | Role |
|------|------|
| [`radar/config.py`](radar/config.py) | Every knob: the sources, the model and effort per agent, the limits |
| [`radar/schemas.py`](radar/schemas.py) | The data's journey: `BrandQuery` → `Mention` → `Verdict` → `SourceReport` → `RadarRun` |
| [`radar/collectors.py`](radar/collectors.py) | Seven keyless fetchers, each normalising its source into `Mention`s |
| [`radar/prep.py`](radar/prep.py) | Whole-word brand filter, cross-source de-dupe, citable ids |
| [`radar/llm.py`](radar/llm.py) | The only file that calls Claude: structured output, streaming, web search + `pause_turn` |
| [`radar/agents.py`](radar/agents.py) | **The agents:** per-source analyst prompts, verdict checks, sweep, strategist |
| [`radar/metrics.py`](radar/metrics.py) | The numbers, the "needs attention" shortlist, the citation check |
| [`radar/pipeline.py`](radar/pipeline.py) | The conductor: collect → clean → analyse → count → report → verify |
| [`app.py`](app.py) | Streamlit UI: inputs, live progress, metrics, charts, report, evidence tabs |
| [`radar/__main__.py`](radar/__main__.py) | CLI, including a key-free `--collect-only` mode |
| [`tests/`](tests/) | 15 offline tests: worked examples, the whole pipeline with a fake Claude, the UI |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your Anthropic key in `.env`. (If you've run `ant auth login`, the SDK finds that profile instead.)

Run the tests. No key needed, takes a few seconds:

```bash
uv run pytest
```

See what the free sources return, with no AI and no key:

```bash
uv run python -m radar "Figma" --collect-only
```

Start the app:

```bash
uv run streamlit run app.py
```

Or run the whole thing from the terminal and save everything:

```bash
uv run python -m radar "Notion" --context "notes and docs app" --exclude "notion of" --json notion.json
```

**Tip:** for brands with ordinary-word names (Notion, Linear, Arc, Apple), always fill in *What is it?* and a few *Ignore* phrases. The word filter can't tell "Linear the issue tracker" from "linear algebra"; the analysts can, but only if they know which one you mean.

## Limits, honestly

- **DuckDuckGo is throttled.** News, web, Reddit and YouTube go through the keyless `ddgs` package, which sometimes returns nothing after a burst. Bluesky, Hacker News and Mastodon are direct APIs and much steadier. The UI shows per-source counts, so an empty source is visible rather than silent.
- **Reddit** blocks anonymous API and RSS access (HTTP 403), so Reddit is "reddit.com pages DuckDuckGo has indexed". It looks back at least 30 days, since new threads take time to get indexed.
- **YouTube's** video endpoint is often empty. The fallback (web search restricted to youtube.com) has no view counts.
- **Mastodon** full-text search needs a login, so only posts under `#brand` on mastodon.social are seen.
- **X/Twitter, Instagram, LinkedIn, TikTok** have no keyless access. Turn on the Claude sweep to get some coverage of public pages from them.
- Sentiment comes from snippets, not full pages. Treat it as triage, not a measurement.

## What I learned

- **Let code do the counting.** LLMs are good judges of one mention and bad accountants of fifty. The analyst returns a verdict per item; Python does the sums; the strategist is told to use the numbers as given.
- **Verify the model's references.** With structured output, the JSON always *parses*, but it can still refer to things that don't exist. `check_verdicts` and `check_citations` are about ten lines each and catch exactly that.
- **One specialist per source beats one generalist.** A Hacker News comment, a headline and a Bluesky joke need different reading habits. Small, source-specific prompts also run in parallel and fail independently.
- **Filter before you pay.** In live runs for Notion and Linear, 12 or 13 of DuckDuckGo News's 15 results didn't name the brand at all (the engine ignores the quotes). The free word filter drops them before any tokens are spent.
- **Read the API, not the docs page.** Live probing turned up four quirks: Algolia's typo tolerance matched "notion" to "national"; `public.api.bsky.app` refuses search while `api.bsky.app` allows it; Reddit's RSS is blocked too; and stripping query strings for de-duplication collapsed every Hacker News link (`item?id=…`) into one. Each one is now a commented line in the code.
- **Make failure loud.** With no API key, the first version reported "no relevant mentions", which was technically true and completely misleading. Now, if every analyst fails, the report says so and shows the error.
