# context-forge

A research assistant that **shows its context engineering**. You ask a question;
it gathers candidate context from four sources (your documents, the web,
arXiv papers, and its own conversation memory), decides what deserves a place
in a fixed token budget, and answers with citations it has checked.

The point is the middle step. Retrieval is easy; choosing what goes into the
window is the hard part. Every candidate is kept or dropped with a written
reason, and the UI shows all of it.

## What it does

- **Plans** with one small Claude call: a keyword query per source, plus whether
  the web and papers are worth searching for this question.
- **Gathers in parallel** from:
  - 📄 **documents** you upload (PDF / Markdown / text), chunked and BM25-searched locally
  - 🌐 **web**, via Claude's server-side web search tool (no search API key)
  - 🎓 **arXiv**, the free public API (no key)
  - 🧠 **memory**: past Q&A turns in a local JSON file, retrieved by relevance rather than replayed in full
- **Ranks** every candidate on one scale: relevance → optional LLM judge → source
  trust → off-topic threshold → near-duplicate removal → per-source cap → token budget.
- **Answers** from the selected window only, cites `[S#]` per claim, and strips
  any citation that points at a block that wasn't in the window.
- **Shows its work** in four tabs under each answer: Sources · Window · Rejected · Trace.

Only key needed: `ANTHROPIC_API_KEY`.

## Architecture

```
                         question
                            │
                 ┌──────────▼──────────┐
                 │ 1 Planner (Claude)  │  per-source queries, skip web/papers?
                 └──────────┬──────────┘
        ┌─────────────┬─────┴──────┬──────────────┐      parallel threads;
        ▼             ▼            ▼              ▼      one failing source
   📄 documents   🌐 web search  🎓 arXiv     🧠 memory   doesn't stop the run
   (BM25, local)  (server tool)  (Atom API)   (JSON+BM25)
        └─────────────┴─────┬──────┴──────────────┘
                            │  ContextItem × N  (one shape for every source)
                 ┌──────────▼──────────┐
                 │ 3 Ranker            │  relevance · judge · trust · threshold
                 │   (the gauntlet)    │  · dedupe · per-source cap · token budget
                 └──────────┬──────────┘
                            │  kept items → [S1] [S2] …
                 ┌──────────▼──────────┐
                 │ 4 Synthesizer       │  answer from window only; citations checked
                 └──────────┬──────────┘
                            ▼
                 5 save Q+A to memory  →  Streamlit renders the whole Run
```

Model: `claude-opus-5-5` throughout. Planner, judge and web scout run at
effort `low`; the final answer at `medium`. All calls opt into the
server-side refusal fallback (`fallbacks="default"`).

## Code map (read in this order)

| # | File | What to look for |
|---|------|------------------|
| 1 | [forge/config.py](forge/config.py) | Every budget, weight and threshold. Context engineering is mostly these numbers. |
| 2 | [forge/items.py](forge/items.py) | `ContextItem`: the one shape all four sources produce, plus the ranker's verdict fields. |
| 3 | [forge/textsearch.py](forge/textsearch.py) | Tokenizer, chunker, Jaccard, and BM25 written from scratch. |
| 4 | [forge/llm.py](forge/llm.py) | The single door to Claude: model, effort, structured outputs, fallback. |
| 5 | [forge/planner.py](forge/planner.py) | Structured-output call that routes the question; safe fallback plan. |
| 6 | [forge/sources/documents.py](forge/sources/documents.py) | Upload → chunk → BM25. |
| 7 | [forge/sources/papers.py](forge/sources/papers.py) | arXiv Atom feed parsed with the standard library. |
| 8 | [forge/sources/web.py](forge/sources/web.py) | Server-side web search; turns citation quotes into readable passages; `pause_turn`. |
| 9 | [forge/sources/memory.py](forge/sources/memory.py) | Persistent memory retrieved by relevance, not replayed. |
| 10 | [forge/ranker.py](forge/ranker.py) | **The core.** The seven-step gauntlet and the optional LLM judge. |
| 11 | [forge/synthesizer.py](forge/synthesizer.py) | Renders the window, writes the answer, checks citations. |
| 12 | [forge/pipeline.py](forge/pipeline.py) | Wires it all together and records everything on a `Run`. |
| 13 | [app.py](app.py) | Streamlit UI: sidebar controls, chat, the four tabs. |
| 14 | [tests/](tests/) | Offline tests: fake Claude, real SDK with a mock transport, headless UI. |

## How to run

```bash
cd context-forge
```

```bash
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env
```

```bash
uv sync
```

```bash
uv run streamlit run app.py
```

Click **Load sample notes** (or upload your own files), then ask, e.g.
*"What is context engineering, and is there research on models losing
information in the middle of long contexts?"* Then try this: drag the
**Token budget** slider down to 500, ask again, and watch the Rejected tab fill
up with `over budget` reasons.

Tests (no key, no network):

```bash
uv run pytest -q
```

## Verified vs. not verified

- ✅ 22 offline tests pass: ranker rules, BM25, chunking, arXiv parsing,
  web-citation harvesting, `pause_turn` resume, planner fallback, citation
  checking, memory persistence, full pipeline, LLM-judge re-ranking, failing sources.
- ✅ The real `anthropic` 2.8 SDK with a mock HTTP transport accepts our requests
  (structured outputs, `web_search_20260209`, `fallbacks`, beta header) and our
  code reads its real citation types.
- ✅ Live arXiv API call, run through the real ranker (fake Claude only).
- ✅ Headless Streamlit click-through (AppTest) and `streamlit run` health check.
- ❌ **Not verified:** a live run against the Claude API (no key was used), real
  web search results, and PDF upload with a real PDF.

## What I learned

- **One shape beats four pipelines.** Once every source emits a `ContextItem`,
  ranking, deduping, budgeting and display are written once. Adding a fifth
  source is one file.
- **Score on a shared scale.** Scoring each source with its own retriever gives
  numbers that can't be compared. Re-scoring every candidate with one BM25 index
  over the pooled set fixes that, and source weights then express trust on top.
- **The budget is a knapsack, not a cut-off.** "Take the top N" wastes space when
  one item is huge. Greedy packing by score, skipping what doesn't fit, lets
  smaller relevant items fill the gap.
- **Diversity has to be enforced.** Without a per-source cap, whichever source
  returns the most text takes over the window.
- **Memory is just another source.** Replaying the whole chat is expensive and
  noisy. Retrieving relevant past turns and making them compete (at a lower
  trust weight) keeps the useful part.
- **Server-side web search hides page text.** Result blocks are encrypted; the
  readable passages are the `cited_text` on citations. So the web scout asks for
  cited notes and harvests the quotes.
- **Verify citations mechanically.** A regex check against the ids in the
  window catches fabricated references for free.
- **Explain every decision.** Writing a one-line `reason` for each kept or dropped
  item made the ranker far easier to debug, and it is the main thing the UI shows.
