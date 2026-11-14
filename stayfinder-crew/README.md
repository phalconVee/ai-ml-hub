# stayfinder-crew

A multi-agent hotel finder. You describe a stay in plain words:

> Lisbon, 3 nights from 12 December, two of us. Under €160 a night, near Alfama
> or Baixa. Must have wifi and air conditioning, breakfast would be nice.
> Rated 8+ please, and a quiet room.

A crew of agents parses that into a typed request, finds hotels, compares them
on price, location, reviews and amenities, and recommends the best picks with
reasons. You get a ranked table, a map, and a short list of picks.

**Information only.** Nothing is booked, reserved or paid. Prices from the web
are what a page said, not live quotes.

**Two brains, one switch.** Pick **Claude** (cloud, default) or **Ollama** (a
local model such as `qwen3:4b`) in the sidebar. With Ollama + OpenStreetMap the
whole thing runs without any API key, and the language models run 100% on
your machine.

## Architecture

```
            ┌───────────────────────── brain: ClaudeBrain | OllamaBrain ─────────────────────────┐
            │                         (one method: json(system, user, schema))                    │
            ▼                                                                                     ▼
 text ──► 1 Parser ──► StayRequest ──► 2 Geocoder ──► target point ──► 3 Hotel scout ──► [HotelOption]
          (LLM)        city, dates,     (Nominatim,                     web  : Claude web search (LLM)
                       budget, must-     no LLM)                        osm  : Overpass API (no LLM)
                       haves, priority                                  demo : made-up, offline
                                                                                │
                      Advice ◄── 5 Advisor ◄── Analysis ◄── 4 Analyst ◄─────────┘
                      picks +     (LLM, then    ranked +     (pure Python: rule out on
                      reasons     verify() /    excluded     known facts, score 4 factors,
                                  fallback)                  weight by priority, badges)
```

| Step | Who | Why this tool |
|---|---|---|
| Parser | LLM + `validate()` | Language (relative dates, "two of us", €) needs a model; dates and enums are then checked in code. |
| Geocoder | OpenStreetMap Nominatim | Free, keyless coordinates for "Alfama, Lisbon" so distance is a number, not a vibe. |
| Hotel scout | Claude web search **or** OSM Overpass **or** demo | Web search finds prices and review scores; OSM gives real hotels and tagged amenities with no key; demo works offline. |
| Analyst | Plain Python | Comparing 12 hotels on 4 factors is arithmetic. Deterministic, testable, free. |
| Advisor | LLM + `verify()` | Judgement and explanation, but every cited hotel id is checked against the table, and a no-LLM fallback means a run never dead-ends. |

The orchestration is ordinary Python in `pipeline.py`: the order never
changes, so it lives in code, not in a prompt.

### What each source can and can't tell you

| | price | reviews | stars | amenities | location |
|---|---|---|---|---|---|
| Claude web search | indicative (`estimated`) | yes, normalised to /10 | often | what pages mention | geocoded by name |
| OpenStreetMap | **no** | **no** | sometimes | what mappers tagged (yes / no / unknown) | exact |
| Demo | fake | fake | fake | fake | fake |

Missing facts score a neutral 0.5 in the analyst, so an OSM hotel isn't
punished for having no price, and the table shows `None` instead of guessing.

## Code map (reading order)

1. [`stay/config.py`](stay/config.py): every knob: models, effort per agent, Ollama settings, OSM endpoints, scoring weights.
2. [`stay/models.py`](stay/models.py): `StayRequest`, `HotelOption`, and the fixed amenity vocabulary.
3. [`stay/llm.py`](stay/llm.py): the two brains. `ClaudeBrain` (structured outputs, effort) and `OllamaBrain` (`/api/chat` with `format=<schema>`, `think`, lenient parsing + one retry).
4. [`stay/parser_agent.py`](stay/parser_agent.py): Agent 1. Schema, prompt, and `validate()`, which turns bad dates into a question.
5. [`stay/geo.py`](stay/geo.py): Nominatim geocoder (cache + 1 req/s), haversine `km()`.
6. [`stay/sources.py`](stay/sources.py): Agent 2. `OsmSource`, `WebSource` (search → extract → keep only URLs actually seen), `DemoSource`, `pick_source()`.
7. [`stay/analyst.py`](stay/analyst.py): Agent 3. Rule-outs, four factor scores, weights, badges.
8. [`stay/advisor_agent.py`](stay/advisor_agent.py): Agent 4. Prompt, `verify()`, `fallback()`.
9. [`stay/pipeline.py`](stay/pipeline.py): runs the crew and records a timed trace.
10. [`app.py`](app.py): Streamlit UI.
11. [`tests/`](tests): offline tests with fake brains, a fake Anthropic client and canned OSM data.

## Run it

```bash
cd stayfinder-crew
uv sync
cp .env.example .env        # add ANTHROPIC_API_KEY for the Claude brain
uv run streamlit run app.py
```

**100% local (no keys):** install [Ollama](https://ollama.com), then

```bash
ollama pull qwen3:4b
```

and choose *Ollama* in the sidebar with *OpenStreetMap* (or *Demo*) as the
source. The language model runs locally; OSM lookups go to the free public
OpenStreetMap services, and *Demo* needs no network at all. Any pulled chat
model works (the sidebar lists them); bigger models parse and advise better.

**Tests (offline, no keys):**

```bash
uv run pytest -q
```

## What I learned

- **Pick the brain per job, not per app.** Only three of the five steps use a
  model. Geocoding, distance and scoring are code: exact, free, and the same
  on Claude or a 4B local model. That's what made a local backend realistic.
- **A one-method interface is enough to swap LLMs.** Every agent calls
  `brain.json(system, user, schema)`. Claude enforces the schema server-side
  (structured outputs); Ollama turns the schema into a grammar via `format`.
  The local side still needs a lenient parser and a retry: small models can
  wrap JSON in prose or stop early.
- **"Unknown" is a third value.** OSM tags amenities as yes, no, or not at
  all. Ruling a hotel out only on a *known* "no" (and scoring unknowns as
  neutral) keeps the ranking honest instead of silently treating "untagged"
  as "missing".
- **Normalise before you compare.** Total-stay prices become per-night,
  4.5/5 becomes 9/10, and prices in other currencies aren't compared at all
  rather than FX-guessed.
- **Verify what the model cites.** The web scout keeps only hotels whose URL
  appeared in a real search result, and the advisor's picks must be ids in the
  table. Both checks are a few lines and catch confident inventions.
- **Free public APIs are shared.** The main Overpass server returned 502
  during development; trying mirrors in order fixed it. Nominatim asks for a
  real User-Agent and ≤1 request per second, so the geocoder caches and waits.
