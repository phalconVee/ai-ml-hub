# Skyfare Scout

A multi-agent flight-search assistant. You describe a trip in plain words ("Lagos to London around 18 December, back after New Year, 2 of us, cheap but at most one stop") and four agents take over: one **parses** it into a search, one **finds** flight options, one **compares** them on price, duration and stops, and one **recommends** the best picks and explains why. Streamlit UI. **Information only:** it never books, holds or pays for anything.

Only one key is required (`ANTHROPIC_API_KEY`). Flight data comes from Claude's built-in web search by default. If you add a free-plan SerpApi key you get live Google Flights prices instead. A **demo** source with made-up flights lets you click through the UI without any network calls for flight data.

## The idea in one paragraph

Use an LLM where language or judgement is needed, and plain code where exactness is. Turning "after New Year, 2 of us" into `return_date=2027-01-03, adults=2` is language: Claude does it. Ranking flights on price, duration and stops is arithmetic: Python does it, exactly and for free. Explaining whether a $40 saving is worth five extra hours and an overnight layover is judgement: Claude again. Between the steps, code **checks** each LLM output: dates must be in the future, airport codes must be 3 letters, a web fare is kept only if its link actually came back from a search, and a recommended flight must exist in the table.

## Architecture

```
 "Lagos to London ~18 Dec, back after New Year, 2 adults, cheap, ≤1 stop"
                                   │
                                   ▼
 ┌─ 1. Trip Parser (Claude, structured outputs) ──────────────── parser_agent.py ─┐
 │  JSON: LOS→LON, 2026-12-18 ↩ 2027-01-03, 2 adults, economy, max 1 stop, cheapest│
 │  validate(): 3-letter codes? dates not past? return after departure?            │
 │  too vague? → ask the traveller ONE question and stop                           │
 └──────────────────────────────────┬──────────────────────────────────────────────┘
                                    │ TripRequest → legs: outbound, return
                                    ▼
 ┌─ 2. Fare Scout, once per leg ───────────────────────────────────── sources.py ─┐
 │  SerpApiSource  SERPAPI_API_KEY set → Google Flights JSON (no LLM)              │
 │  WebFareSource  default → Claude + web_search (resume pause_turn)               │
 │                 → 2nd call extracts rows (structured outputs)                   │
 │                 → keep a row only if its URL was in a search result            │
 │  DemoSource     made-up, seeded, offline                                        │
 └──────────────────────────────────┬──────────────────────────────────────────────┘
                                    │ FlightOptions (OUT-1…, RET-1…)
                                    ▼
 ┌─ 3. Analyst (pure Python) ───────────────────────────────────────── analyst.py ─┐
 │  max-stops filter → weighted score by priority → Pareto-efficient set           │
 │  badges: cheapest · fastest · fewest stops · best match · efficient · over budget│
 └──────────────────────────────────┬──────────────────────────────────────────────┘
                                    │ TripAnalysis (sorted tables + cheapest total)
                                    ▼
 ┌─ 4. Advisor (Claude, structured outputs) ──────────────────── advisor_agent.py ─┐
 │  picks by option id + reasons + trade-offs + caveats                            │
 │  verify(): unknown ids dropped; any leg without a pick gets the top-scored one  │
 └──────────────────────────────────┬──────────────────────────────────────────────┘
                                    ▼
                    app.py: recommendation cards, per-leg tables, agent trace
```

`pipeline.py` runs these four in a fixed order. The order lives in code, not in a prompt, so no agent can skip a step.

## Code map: read in this order

| File | Role |
|------|------|
| [`scout/config.py`](scout/config.py) | Every knob: model, per-agent effort, search limits, scoring weights |
| [`scout/models.py`](scout/models.py) | The three data shapes: `TripRequest`, `Leg`, `FlightOption` |
| [`scout/llm.py`](scout/llm.py) | The one call to Claude: model, effort, structured outputs, refusal fallback |
| [`scout/parser_agent.py`](scout/parser_agent.py) | **Agent 1:** sentence → JSON schema → `validate()` |
| [`scout/sources.py`](scout/sources.py) | **Agent 2:** SerpApi, Claude web search (with URL check), demo data |
| [`scout/analyst.py`](scout/analyst.py) | **Agent 3:** normalised weighted scores, Pareto set, badges, budget |
| [`scout/advisor_agent.py`](scout/advisor_agent.py) | **Agent 4:** recommendation JSON → `verify()` against the table |
| [`scout/pipeline.py`](scout/pipeline.py) | The crew in order, with a timed trace line per step |
| [`app.py`](app.py) | Streamlit: trip box, source picker, live progress, results |
| [`tests/fakes.py`](tests/fakes.py) | A scripted fake Claude client, routed by system prompt |
| [`tests/test_analyst.py`](tests/test_analyst.py) | Worked examples of the comparison maths |
| [`tests/test_agents.py`](tests/test_agents.py) | Parser, the three sources, advisor verification |
| [`tests/test_pipeline.py`](tests/test_pipeline.py) | End-to-end run and the Streamlit app, no keys |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your Anthropic key in `.env` (or run `ant auth login`; the SDK finds that profile). Optionally add `SERPAPI_API_KEY` for live Google Flights prices.

Run the tests (no keys needed, about 3 seconds):

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Try the example trip with **Demo data** first (only the parser and advisor call Claude), then switch to **Claude web search** for real, indicative fares.

## Design choices, and why

| Choice | Why |
|--------|-----|
| Four agents in a fixed Python pipeline, not a free-roaming agent | Each step's input and output is a typed object you can print and test. Nothing is skipped or done twice. |
| Comparison in Python, not in the LLM | Ranking numbers is maths. Code does it exactly, for free, and the advisor then explains facts instead of making up numbers. |
| Pareto-efficient ("efficient") badge | An option that is beaten on price, duration *and* stops by another is never the right answer, whatever your priority. This removes those options cleanly, with no weights to tune. |
| Round trips searched as two one-way legs | Simple and the same for every source. The cost: round-trip fares are sometimes cheaper than two one-ways (see Ideas). |
| Web search in two calls (research, then extract) | The research call writes free notes with citations. A separate tool-free call converts them to JSON under a schema. Each call stays simple and each failure is visible in the trace. |
| Keep a web fare only if its URL came back from search | It's cheap to check, and it catches made-up or garbled fares before they reach the table. |
| Per-agent effort (`low` for parsing/extraction, `medium` for scouting/advice) | Spend thinking where it pays. Parsing a sentence doesn't need deep reasoning. |
| `TODAY` in the user message, not the system prompt | Relative dates need today's date, and keeping it out of the system prompt keeps that prompt identical every day (better for prompt caching). |
| SerpApi optional, web search default | One required key. A free-plan SerpApi key upgrades the data to live quotes without changing any other code. |
| `fallbacks="default"` on every call | If a safety classifier declines a request, the API retries on a fallback model in the same call. Remove it in `llm.py` if you don't want that. |

## What was verified, and what wasn't

- ✅ **22 offline tests pass** (`uv run pytest`, ~3 s, no keys). They cover: the scoring maths and Pareto set; priority changing the winner; max-stops, budget and other-currency handling; the parser's schema, `TODAY` injection and validation (bad codes, past dates, return before departure); refusal handling; SerpApi request parameters, response mapping and error bodies (from a recorded-shape fixture); the web source's `pause_turn` resume and its URL check; the demo source's determinism; advisor `verify()`; a full four-agent run; and the Streamlit app rendering a result (Streamlit `AppTest` with a fake Claude).
- ✅ The installed `anthropic` SDK (2.8.0) accepts every request parameter the app sends (`output_config`, `betas`, `fallbacks`, tools).
- ✅ The real Streamlit server starts headless and answers its health check.
- ⬜ **Not run against the live Claude API** (no key was used). The calls follow the current API docs; the first real run is the true test, especially the web-search prompt's fare quality.
- ⬜ **Not run against live SerpApi** (no key). The parser was written from SerpApi's documented response shape and tested on a fixture in that shape.

## What I learned

- **Split the work by kind, not just by step.** "Multi-agent" doesn't mean every agent is an LLM. Two of the four steps here (SerpApi search and the analyst) are plain code, and they're the most reliable parts.
- **Structured outputs make LLM hand-offs boring, which is good.** Each LLM step returns JSON that's guaranteed to parse. What's left to check is whether the *values* make sense, and that's a short function per agent.
- **A schema can't check everything.** "Date in the future", "return after departure", "this URL was really returned by search", "this flight id exists" all need code. Each check is a few lines and prevents a believable but wrong answer.
- **Pareto before weights.** Weighted scores depend on numbers someone picked. The Pareto set needs no weights at all, so it's a useful filter before the weighted ranking.
- **Server tools have their own quirks.** Web search can stop with `pause_turn` (send the turn back as-is to resume), and its errors arrive as content, not exceptions.
- **Make the LLM swappable.** Routing a fake client by system prompt let one small class script all four agents, in any order, for free.

## Ideas for next steps

- Real round-trip pricing (SerpApi's `departure_token` flow) and comparing it with two one-ways.
- Flexible dates: search ±2 days and show a price calendar.
- Currency conversion so prices in other currencies can be ranked too.
- A follow-up chat ("what about Gatwick only?") that edits the `TripRequest` instead of starting over.
