# Newsroom Crew

A small **multi-agent news desk**. Give it a topic and three Claude agents work it like a newsroom:

1. a **Reporter** searches the live web, opens the pages that matter and files field notes,
2. a **Writer** turns those notes into a neutral news article where every fact carries a source marker like `[S2]`,
3. an **Editor** checks the draft against the sources and either publishes it or sends it back with notes.

The Streamlit UI shows the article with clickable citations, the sources with the exact quoted evidence, every draft with the editor's notes, and a trace of what was searched and read. You can download the result as Markdown.

Only the LLM calls are borrowed. The pieces that decide what counts as a source and when a draft is good enough are plain Python you can read: the source ledger, the citation audit, and the write → review → revise loop. Every file is heavily commented.

**One key:** `ANTHROPIC_API_KEY`. Web search and page fetching are Claude's server-side tools, so there's no Tavily, Serper or news-API key.

## The idea in one paragraph

Language models write fluent news even when they have no facts. So in this desk the writer never touches the web: it can only use a **numbered source ledger**. Plain code builds that ledger from what the API returned: the pages the search actually found, the pages the reporter actually opened, and the **citations the API attached** to the reporter's notes (exact quoted text, with the URL it came from). The writer must tag each factual sentence with ledger ids. Before the LLM editor reads the draft, a **mechanical audit** checks those ids: a citation to a source that doesn't exist, or a paragraph that cites nothing, means "revise", whatever the editor thinks. No sources at all? The story is **spiked**. The desk won't publish news from nothing.

## Architecture

```
                 ┌──────────────────────────── Streamlit (app.py) ────────────────────────────┐
  topic, angle,  │  st.status live log  ·  article with [S#] links  ·  sources + quotes  ·    │
  length ───────►│  editor's desk (drafts + notes)  ·  reporter's notes  ·  trace  ·  .md ⬇   │
                 └───────────────────────────────────┬────────────────────────────────────────┘
                                                     │ run_desk()
                                                     ▼
 ┌─ reporter.py ─────────── Agent 1 ─┐      desk.py decides the order of work, in code
 │ Claude + server tools:            │
 │   web_search_20260209  (≤6)       │
 │   web_fetch_20260209   (≤4 pages) │
 │ resumes pause_turn                │
 │ → field notes (text + citations)  │
 └───────────────┬───────────────────┘
                 │ all response blocks
                 ▼
 ┌─ ledger.py ──────────── plain code ─┐
 │ search results → sources            │
 │ fetch results  → "read in full"     │
 │ citations      → quoted evidence    │
 │ rank, cap at 12, number S1..Sn      │──── no sources? ──► SPIKED (nothing written)
 └───────────────┬─────────────────────┘
                 ▼
 ┌─ writer.py ──────────── Agent 2 ─┐         ┌─ editor.py ─────────────────────────────┐
 │ no tools: notes + ledger only    │ draft   │ audit()  (code): unknown [S#]?          │
 │ JSON: headline, dek, body [S#]   │───────► │          uncited paragraph?             │
 └──────────────────────────────────┘         │ Editor (Agent 3): claims vs. quotes     │
                 ▲                            │ JSON: publish | revise + notes          │
                 │  revise (≤1 round)         │ broken citations ⇒ always "revise"      │
                 └────────────────────────────┴───────────────┬─────────────────────────┘
                                                              │ publish
                                                              ▼
                                         Story: final draft, cited sources, drafts, reviews, trace
```

## Code map: read in this order

| File | Role |
|------|------|
| [`newsroom/config.py`](newsroom/config.py) | Every knob: model, per-agent effort, search/fetch budgets, ledger size, revision rounds |
| [`newsroom/llm.py`](newsroom/llm.py) | The single door to Claude: model, effort, structured outputs, refusal fallback |
| [`newsroom/reporter.py`](newsroom/reporter.py) | **Agent 1:** server-side web search + fetch, `pause_turn` resume, the search/read trail |
| [`newsroom/ledger.py`](newsroom/ledger.py) | **The core idea:** response blocks → numbered sources with API-quoted evidence |
| [`newsroom/writer.py`](newsroom/writer.py) | **Agent 2:** tool-less writer, JSON article, revision prompt |
| [`newsroom/editor.py`](newsroom/editor.py) | Mechanical citation `audit()` + **Agent 3**, the LLM editor, and the hard override rule |
| [`newsroom/desk.py`](newsroom/desk.py) | The orchestrator: report → ledger → write → review → revise, spike, `Story` |
| [`app.py`](app.py) | Streamlit: the assignment form, live progress, article, sources, editor's desk, download |
| [`tests/fakes.py`](tests/fakes.py) | A scripted fake Claude and helpers that build realistic response blocks |
| [`tests/test_ledger.py`](tests/test_ledger.py) | Worked examples: ranking, de-duplication, tool errors, rendering |
| [`tests/test_desk.py`](tests/test_desk.py) | The whole crew offline: pause/resume, revision loop, audit override, spike, refusal |
| [`tests/test_sdk_wire.py`](tests/test_sdk_wire.py) | The real `anthropic` SDK over a mock HTTP transport: wire JSON + real block types |
| [`tests/test_app.py`](tests/test_app.py) | The Streamlit script driven headless with the fake Claude |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your Anthropic key in `.env`. That's the only key. (If you've run `ant auth login`, the SDK finds that profile and `.env` is optional.)

Run the tests (no key, no network, a few seconds):

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Type a topic in the sidebar (try something from this week), optionally an angle, pick a length and click **File the story**. A run makes about 4 to 7 Claude requests: 1 or more for the reporter, then a writer and an editor call per draft.

## Design choices, and why

| Choice | Why |
|--------|-----|
| Three agents with different powers | Only the reporter can reach the web. The writer can't search, so it can't quietly add facts of its own. The editor can't rewrite, so review and authorship stay separate. |
| The ledger is built by code, not by an LLM | A model saying "source: Reuters" is a claim. A `web_search_tool_result` block from the API is a record. Quotes come from the API's own citations, so they are text the page really contains. |
| A mechanical audit before the LLM editor | Unknown ids and uncited paragraphs are facts code can check perfectly and for free. The LLM editor then spends its effort on what code can't judge: does the source actually say this? |
| Hard rule beats soft judgement | If the audit fails, the verdict is "revise" even when the editor says "publish". Tested in `test_broken_citations_override_a_publish_verdict`. |
| Spike instead of improvising | No sources means no story. An honest "nothing found" is better than fluent fiction. |
| Structured outputs for writer and editor | Headline, dek, body and verdict arrive as validated JSON fields, so there's no prose parsing to break. |
| Fresh single-turn calls for revisions | The revision prompt carries the old draft and the notes. No conversation history to replay, so each call is simple to read and test. |
| Effort per agent (`medium` / `high` / `medium`) | Reading the web is breadth work. Writing clean, fully cited prose is where extra thinking pays off. |
| Date in the user message, not the system prompt | Keeps system prompts byte-identical across runs (prompt-cache friendly) while the agents still know what "recent" means. |
| `fallbacks="default"` on every call | If a safety classifier declines, the API retries on a fallback model inside the same call. Remove it in `llm.py` if you don't want that. |

## What was verified, and what wasn't

- ✅ **15 offline tests pass** (`uv run pytest`, no key, no network):
  - ledger ranking, URL de-duplication, search-error blocks skipped, fetch titles, rendering;
  - the full desk with a scripted Claude: `pause_turn` resumed by sending the paused turn back with no extra user message, writer sees ids + quotes and has no tools, editor → revise → rewrite carrying the notes, audit overriding a "publish", revision budget → `published_with_notes`, spike on zero sources (writer never called), refusal raised;
  - the model, effort, tool versions, fallback beta and JSON schemas on every request.
- ✅ **Real SDK on the wire:** `test_sdk_wire.py` runs the genuine `anthropic` 2.8.0 client over a mock HTTP transport. It confirms the SDK accepts every parameter, the JSON body matches the documented shapes (`web_search_20260209`, `web_fetch_20260209`, `fallbacks`, `output_config`, beta header), and real `web_search_result` / citation JSON parses into SDK types that the ledger reads correctly.
- ✅ **The Streamlit app runs headless** (`AppTest`): type a topic, click, get the headline, linked `[[S1]](url)` citations, and 4 requests to the fake Claude.
- ✅ **`streamlit run app.py` starts** and its health endpoint answers (checked locally, no key needed to load the page).
- ⬜ **Not run against the live Claude API** (needs a paid key). The requests follow the current API docs and pass the SDK's own validation, but the first real run is the true test of prompt quality, citation density and search behaviour.
- ⬜ **Citations from fetched pages:** the ledger takes quotes from web-search citations (which carry a URL). Pages the reporter fetched are marked "read in full", but citations into fetched documents aren't turned into quotes. See "Ideas".

## What I learned

- **Separate the powers, not just the prompts.** "Don't make things up" in a prompt is a request. Taking the web tools away from the writer makes it a fact of the code.
- **Server tools leave a paper trail.** Every search, every fetched page and every citation comes back as a typed block. Read those blocks instead of asking the model what it read, and the source list builds itself honestly.
- **Cheap checks first, smart checks second.** A 10-line regex audit catches the citation errors an LLM editor can miss, and it never gets tired. The LLM is for what needs judgement.
- **Server-tool quirks are part of the design.** `pause_turn` means "send the turn back as-is", and tool errors arrive as content, not exceptions. Both need a line of code and a test.
- **A spike is a feature.** Deciding when *not* to publish is as much a part of a news desk as writing well.
- **Swap the client, test the whole crew.** Passing `client` in let a 20-line fake run every agent loop offline. A mock transport under the real SDK checked what fakes can't: the actual wire format.

## Ideas for next steps

- Turn on `citations` for `web_fetch` and map fetched-document citations into ledger quotes too.
- Stream the writer's draft into the page as it's written (`client.beta.messages.stream`).
- A "fact-check" agent that re-searches the 2 or 3 most important claims independently before publishing.
- Domain allow/block lists in the sidebar (`allowed_domains` / `blocked_domains` on the search tool), e.g. "primary sources only".
- Run the reporter several times with different angles in parallel and merge the ledgers.
