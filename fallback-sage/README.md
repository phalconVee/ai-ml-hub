# Fallback Sage

An agentic RAG app that **answers from your own documents first** and searches the web **only when the documents don't have the answer**. Upload PDFs, text or Markdown, ask a question in a Streamlit chat, and every answer is labelled with where it came from (📚 your documents or 🌐 the web), with its sources and a trace of how it got there.

Everything except the LLM is written **from scratch** so it can be read line by line: the chunker, the BM25 search engine (~80 lines with comments, no vector DB, no embeddings key), the agent loop, and the router. Every file is heavily commented.

## The idea in one paragraph

Two agents, one strict order. A **Librarian** gets your question and the best-matching passages from your files. It may search again with better keywords, then must give a JSON verdict: *answered* (with the passage ids it relied on) or *not found*. Plain Python then **checks the claim**: an "answered" counts only if it cites passages that search actually returned. If the claim fails the check, a **Scout** answers with Claude's built-in web search. The order is set in code, not in a prompt. The Scout never runs until the Librarian has said no, so the model cannot skip your documents.

## Architecture

```
                       ┌────────────────────────── Streamlit (app.py) ──────────────────────────┐
  upload PDF/TXT/MD ──►│ documents.py: pages → 180-word chunks (40 overlap) → bm25.py: BM25Index│
                       └──────────────────────────────────┬─────────────────────────────────────┘
                                                          │ question
                                                          ▼
                                                   router.py: answer()
                                                          │
                         no documents? ───────────────────┼───────────────────────┐
                                                          ▼                       │
                ┌──────────── librarian.py (Stage 1) ────────────┐                │
                │ seed search with the raw question              │                │
                │ loop ≤4: Claude ⇄ search_documents (BM25)      │                │
                │ final JSON verdict (structured outputs)        │                │
                │ verify(): cited ids ⊆ passages it was shown?   │                │
                └───────────────┬───────────────────┬────────────┘                │
                        found ✔ │                   │ not found / failed check     │
                                ▼                   ▼                              ▼
                     📚 answer + passages   ┌──── scout.py (Stage 2) ─────────────────┐
                                            │ Claude + web_search server tool          │
                                            │ resume on pause_turn, collect citations │
                                            └──────────────────┬───────────────────────┘
                                                               ▼
                                                    🌐 answer + links
```

## Code map: read in this order

| File | Role |
|------|------|
| [`sage/config.py`](sage/config.py) | Every knob: model, effort, chunk size, top-k, loop limits |
| [`sage/documents.py`](sage/documents.py) | Files → pages → overlapping word-window `Chunk`s with citable ids |
| [`sage/bm25.py`](sage/bm25.py) | **Search from scratch:** tokenizer, IDF, the BM25 formula, ranking |
| [`sage/llm.py`](sage/llm.py) | The single call to Claude (model, effort, refusal fallback in one place) |
| [`sage/librarian.py`](sage/librarian.py) | **The core agent:** tool loop over your docs, JSON verdict, `verify()` |
| [`sage/scout.py`](sage/scout.py) | Web fallback: server-side web search, `pause_turn`, source collection |
| [`sage/router.py`](sage/router.py) | Docs first, web second: the decision, in plain Python |
| [`app.py`](app.py) | Streamlit: upload, index caching, chat, source + trace panels |
| [`tests/test_retrieval.py`](tests/test_retrieval.py) | Worked examples of chunking and BM25 ranking |
| [`tests/test_routing.py`](tests/test_routing.py) | The whole agent flow with a scripted fake Claude (no key needed) |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your Anthropic key in `.env`. That's the only key: web search runs inside the Claude API, so there's no separate search-API key. (If you've run `ant auth login`, the SDK finds that profile and `.env` is optional.)

Run the tests (no key needed, takes a second):

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Try it: upload a document, ask something it covers (📚), then something it doesn't (🌐). Turn off **Fall back to web search** in the sidebar to see the "not found" path.

## Design choices, and why

| Choice | Why |
|--------|-----|
| Two separate agents, not one agent with both tools | With both tools in one conversation, "search the docs first" is only a request in the prompt. Splitting them makes the order a fact of the code. |
| BM25 instead of embeddings | Needs no second API key or vector DB, and you can check every score by hand. Its weakness (synonyms) is covered by letting Claude rewrite the query and search again. |
| Seed search before the agent starts | Claude always sees your documents, even if it never calls the tool. |
| `verify()` after the verdict | An LLM's "I found it" is a claim. Citing passages it was actually shown is evidence. A failed check sends the question to the web instead of showing a possibly made-up answer. |
| Structured outputs for the verdict | The final message must match a JSON schema, so parsing can't break on extra prose. |
| Claude's server-side `web_search` tool | No Tavily/Serper key or glue code. Results arrive with citations attached. |
| `fallbacks="default"` on every call | If a safety classifier declines a request, the API retries it on a fallback model within the same call. Remove it in `llm.py` if you don't want that. |
| Each question is answered on its own | Keeps the flow easy to follow. Follow-ups like "and the year before?" won't see earlier turns (see "Ideas"). |

## What was verified, and what wasn't

- ✅ 12 unit tests pass. They cover chunking, BM25 ranking, the librarian's tool loop (tool results tied back to their `tool_use_id`), the docs → web fallback, rejection of answers that cite passages never retrieved, the web toggle, `pause_turn` resumption, and the model/effort/fallback settings on every request.
- ✅ The Streamlit app runs headless (Streamlit `AppTest`) with a fake Claude, from question to rendered answer, sources and trace.
- ✅ The installed `anthropic` SDK (2.8.0) accepts every request parameter used.
- ⬜ **Not yet run against the live API** (needs a paid key). The Claude calls follow the current API docs, but the first real run is the true test.
- ⬜ PDF upload isn't tested with a real PDF. Text/Markdown extraction is covered; the PDF path is a thin `pypdf` call.

## What I learned

- **"Agentic" doesn't have to mean "let the model decide everything".** The useful split is: the model decides *how* to search (which keywords, how many tries); code decides *what is allowed* (docs before web) and *what counts* (cited evidence).
- **The model's verdict is a claim; check it.** One set check (`cited ⊆ seen`) catches the most damaging failure of RAG: an answer that sounds grounded but isn't.
- **BM25 is still strong.** It fits in one short file, needs no keys and is fully explainable. Add embeddings when synonym misses become a real problem, not by default.
- **Client tools vs server tools.** `search_documents` runs on my machine: I execute it and send back a `tool_result`. `web_search` runs on Anthropic's servers: I only declare it and read the results, and I must handle `pause_turn` and error objects that arrive as content instead of exceptions.
- **Make the LLM swappable for tests.** Passing `client` in, instead of creating it deep inside, let a 20-line fake check the whole agent flow for free.

## Ideas for next steps

- Conversation memory: have Claude rewrite a follow-up into a standalone question before routing.
- Hybrid retrieval: add embeddings next to BM25 and merge the rankings (reciprocal-rank fusion).
- Partial answers: when the docs answer half the question, search the web for only the missing half.
- Stream tokens to the UI instead of waiting for the full answer.
