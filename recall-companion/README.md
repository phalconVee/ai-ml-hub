# recall-companion

A chat assistant with **long-term memory that works the way people remember**.
It doesn't replay old transcripts. After each exchange it keeps the facts
worth keeping ("Ana is vegetarian", "her sister Maya lives in Porto",
"marathon on 2027-04-11"). When something changes ("I moved to Lisbon"), it
closes the old fact's time window instead of overwriting it. Before every
answer it recalls only the memories that matter for the message at hand.

The memory layer is written from scratch: one SQLite file, a hand-written
BM25, and two small Claude calls. No memory service and no embedding API, so
you can read every step.

## What it does

- **Remembers across sessions.** Each profile has many chat sessions. A
  session's transcript is short-term memory. Facts are long-term memory and
  are shared by every session of that profile.
- **Multiple profiles.** Each person gets their own memory. Profiles never
  see each other's facts. The store enforces this, so it doesn't rely on the prompt.
- **Temporal facts.** Every fact has `valid_from` / `valid_to` (when it was
  true) and `recorded_at` (when we learned it). New information
  *supersedes* old facts, and the history stays queryable ("where did I live
  before?").
- **Memory inspector** next to the chat:
  - **Used now**: which memories went into the latest answer, with a
    per-signal score breakdown, plus what was learned or replaced in that exchange
  - **Memories**: everything currently believed. Edit or 🗑 forget any of it.
  - **History**: facts that are no longer true, with their date ranges
  - **Entities**: the nodes (people, places, pets…) and current relations

Only key needed: `ANTHROPIC_API_KEY`.

## Architecture

```
 user message
     │
     ├─1─► messages table (this session's transcript = short-term memory)
     │
     ├─2─► Recall planner (Claude, effort low, JSON schema)
     │       "dinner ideas?" → keywords "food diet allergy vegetarian…",
     │                         entities [..], about_the_past? 
     │
     ├─3─► Retriever (local, no LLM)
     │       every fact scored: 0.55·BM25 + 0.25·entity hit
     │                         + 0.12·importance + 0.08·recency
     │       core facts (importance 5) always offered
     │
     ├─4─► Answer (Claude, effort medium)
     │       system prompt = persona + [M12] memories with dates
     │       messages      = last 12 messages of THIS session
     │
     ├─5─► recalls table: which facts were used for which answer
     │
     └─6─► Extractor (Claude, effort low, JSON schema)
             sees: the exchange + known entities + current related facts (with ids)
             returns: edges  (subject) —predicate→ (object)
                      + statement, category, importance, exclusive, date, replaces[ids]
                  │
                  ▼
           store.add_fact():  same fact again?   → confirm (refresh timestamp)
                              replaces [ids]?    → supersede those
                              exclusive predicate? → supersede same subject+predicate
```

```
SQLite (.recall/memory.db)
 profiles ─┬─ sessions ── messages ── recalls ──┐
           ├─ entities  (nodes: user, Maya, Porto, Miso…)
           └─ facts     (edges + statement + valid_from/valid_to/recorded_at
                         + superseded_by → facts.id)
```

Model: `claude-opus-5-5` for all three calls. The two memory calls run at
effort `low` and the answer at `medium`. Every call opts into the server-side
refusal fallback (`fallbacks="default"`). The memory calls use structured
outputs (`output_config.format`), so their JSON always parses.

## Code map (read in this order)

| # | File | What to look for |
|---|------|------------------|
| 1 | [recall/config.py](recall/config.py) | Every number: top-k, score weights, recency half-life, the "core" threshold. |
| 2 | [recall/store.py](recall/store.py) | **The core.** Schema, the two-clock time model, `add_fact` (confirm / supersede), `history`, edit and forget. |
| 3 | [recall/textsearch.py](recall/textsearch.py) | Tokenizer and Okapi BM25 from scratch. |
| 4 | [recall/retriever.py](recall/retriever.py) | Four-signal scoring, the relevance gate, core facts, history mode. |
| 5 | [recall/llm.py](recall/llm.py) | The single door to Claude: model, effort, structured outputs, fallback. |
| 6 | [recall/recall_planner.py](recall/recall_planner.py) | Rewrites the message into a better memory search. Falls back to plain search on any failure. |
| 7 | [recall/extractor.py](recall/extractor.py) | The memory step: prompt, JSON schema, which existing facts it is shown, parsing. |
| 8 | [recall/companion.py](recall/companion.py) | One turn: recall → answer → log → remember. |
| 9 | [app.py](app.py) | Streamlit: profiles, sessions, chat, the four inspector tabs, edit/forget forms. |
| 10 | [tests/](tests/) | Fake Claude, controllable clock, real SDK over a mock transport, headless UI click-through. |

## How to run

```bash
cd recall-companion
uv sync
cp .env.example .env          # then put your ANTHROPIC_API_KEY in .env
uv run streamlit run app.py
```

Create a profile in the sidebar and tell it something about yourself. Then
click **➕ New session** and ask a question where that fact matters. Watch the
**Used now** tab. Try "I moved to …" to see a fact superseded, and check the
**History** tab.

Memory lives in `.recall/memory.db` (git-ignored). Delete that file to start
fresh.

Tests (offline, no key, ~5 s):

```bash
uv run pytest -q
```

## What I learned

- **Memory is mostly about writing, not reading.** Retrieval is easy once
  facts are clean. The hard part is deciding what's worth keeping and in what
  shape. Storing graph edges plus a readable sentence gives the retriever
  something to match and gives the inspector something to show.
- **To detect a change, the writer has to see the old belief.** My first
  version showed the extractor only keyword matches, and "I moved to Lisbon"
  never met "lives in Berlin" (no shared words). A test caught it. The fix:
  show keyword matches, then top up with exclusive and important facts.
- **Two supersession paths are better than one.** The LLM's `replaces` list
  catches contradictions no rule can see ("I quit coffee"). The
  deterministic `exclusive`-predicate rule catches the common case even when
  the LLM forgets to link them.
- **Never overwrite; close the window.** Keeping `valid_to` and
  `superseded_by` costs nothing. In exchange the assistant can answer "what
  changed?", the inspector can show history, and a wrong supersession can be
  undone by hand.
- **The raw message is a bad query.** "What should I cook?" shares no words
  with "vegetarian". A cheap rewrite step (keywords + entities) helps, and
  "core" facts (importance 5) work as a safety net for hard constraints.
- **Separate who from where.** Profiles isolate memory, and sessions only
  isolate transcripts. Getting that boundary right in the schema
  (`profile_id` on every fact, enforced even for `replaces` ids) is what
  makes "carries across sessions, never across people" true by construction.
- **Make time injectable.** A clock passed into the store lets tests jump
  "30 days later" to check recency decay and validity windows without sleeping.

Ideas not built: embeddings next to BM25 for paraphrase matching,
consolidating many small facts into summaries, and decaying (not just
ranking down) memories that are never recalled.
