# Vault Memory Agent: an Agent Whose Memory Lives in a Database

A chat agent with three kinds of memory, each stored in a local database and each read or written on purpose:

1. **Knowledge base.** Upload PDFs, text or Markdown. They are chunked, embedded and stored with their vectors. The agent searches them with a tool.
2. **Conversation history.** Every thread's messages are saved. Pick any old thread and carry on where you left off.
3. **Notes the agent writes itself.** When you mention something durable ("I'm the duty engineer this winter", "the deadline moved to March 10"), the agent decides to `remember` it. In a later conversation, in any thread, it can `recall` it. Outdated notes it can `forget`.

The Streamlit UI shows all of it. Under each answer you see the tool calls behind it. A **Vault** tab lists every saved note and the exact chunks each answer retrieved.

Everything is built **from scratch** and small enough to read in one sitting: the storage layer, the chunker, the tool loop. No agent framework, no vector-database service. Every file is commented for a line-by-line walkthrough.

**Keys:** `ANTHROPIC_API_KEY` is the only one. The database is a single SQLite file with the [sqlite-vec](https://github.com/asg017/sqlite-vec) extension, and embeddings run locally with [fastembed](https://github.com/qdrant/fastembed) (bge-small, CPU, about 70 MB downloaded once). Nothing else needs an account.

## The idea in one paragraph

All three memories sit in one SQLite file. Ordinary tables hold documents, chunks, threads, messages and notes; two `vec0` virtual tables (from sqlite-vec) hold the vectors for chunks and notes, keyed by the same ids. Each turn, the agent loop replays the thread's recent messages (memory 2), then calls Claude with four tools: `search_knowledge_base` (memory 1), and `recall`, `remember` and `forget` (memory 3). Claude decides which to call. Our code runs each call against the database and hands back the result, until Claude answers. The answer is saved to the thread together with a **trace** of every tool call, which is what the UI displays. The agent, the tools and the UI talk only to a small `MemoryStore` interface, so a hosted database could replace SQLite without touching them.

## Architecture

```
                        ┌──────────────────────────── app.py (Streamlit) ──────────────────────────────┐
                        │ sidebar: threads, upload      chat + "memory used"      vault: notes, chunks  │
                        └──────┬───────────────────────────────┬──────────────────────────┬────────────┘
             upload            │                     question  │                          │ reads
                               ▼                               ▼                          │
  ingest.py ◄────────── memory.py: Vault ◄───────── agent.py: run_turn ──► llm.py ──► Claude (Opus 5.5)
  file → pages →        text in, vectors out           1. replay thread history          ▲      │ tool_use
  overlapping chunks     embeds via embeddings.py      2. call Claude                    │      ▼
                         (fastembed or hash)           3. run tools ◄── tools.py: ToolBox ┘  tool_result
                               │                       4. save answer + trace
                               ▼                               │
                   store.py: MemoryStore (interface) ◄─────────┘
                               │
                   sqlite_store.py: SqliteVecStore  ──►  .vault/vault.db (one file)
                     documents ──< chunks ──── chunk_vectors (vec0, cosine)     memory 1
                     threads   ──< messages (text + trace JSON)                 memory 2
                     notes ─────────────────── note_vectors  (vec0, cosine)     memory 3
```

## Code map: read in this order

| File | Role |
|------|------|
| [`vault/config.py`](vault/config.py) | Every knob: model, effort, chunk size, top-k, history window, duplicate threshold, DB path |
| [`vault/store.py`](vault/store.py) | **The interface.** What the agent needs from a database, in three groups of methods |
| [`vault/sqlite_store.py`](vault/sqlite_store.py) | **The database.** Schema, sqlite-vec KNN queries, dedupe, embedder pinning |
| [`vault/embeddings.py`](vault/embeddings.py) | Local embeddings (fastembed) and a no-download hash embedder for tests |
| [`vault/ingest.py`](vault/ingest.py) | File to pages to overlapping word-window chunks |
| [`vault/memory.py`](vault/memory.py) | `Vault`: store + embedder, so callers work in text, not vectors |
| [`vault/tools.py`](vault/tools.py) | The four tool schemas (`strict: true`) and the code that runs them, recording a trace |
| [`vault/llm.py`](vault/llm.py) | The one Claude call: model, effort, tools, refusal fallbacks |
| [`vault/agent.py`](vault/agent.py) | **The core:** the hand-written tool loop and the system prompt |
| [`app.py`](app.py) | Streamlit UI: thread picker, upload, chat with live tool steps, Vault tab |
| [`tests/`](tests/) | Offline tests: storage, retrieval, the tool loop with a fake Claude, a real-SDK wire test, an AppTest click-through |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Then put your `ANTHROPIC_API_KEY` in `.env`. Run the tests (seconds, no key, no model download):

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Upload the two files in `sample_docs/` (a made-up research station's handbook and greenhouse notes), then try:

1. *"I'm the duty engineer this winter. What do I check on Mondays?"* The agent searches the documents and saves a note about you.
2. Click **New thread** and ask *"What are my duties?"* This thread has no history, so the answer can only come from `recall`.
3. Open **Vault** to see the note, and the chunks each answer used.
4. Say *"Actually I've moved to the greenhouse team."* The agent should `forget` the old note and `remember` the new one.

The first start downloads the embedding model. To skip that (for a quick look), set `VAULT_EMBEDDER=hash`: it only matches shared words, not meaning. A database remembers which embedder filled it and refuses to open with another, so use a fresh `VAULT_DB` when switching.

Use it from Python without the UI:

```python
from pathlib import Path
from vault import Vault, run_turn

vault = Vault.open()                       # .vault/vault.db + fastembed
vault.ingest("handbook.md", Path("sample_docs/kestrel-station-handbook.md").read_bytes())
thread = vault.store.create_thread("demo")
result = run_turn(vault, thread.id, "How long can visitors stay?")
print(result.text)
for step in result.trace:
    print(step["tool"], step["input"])
```

### What it costs

- **Uploading and searching:** free. Embeddings and the database run locally.
- **Each message:** one Claude call (Opus 5.5, effort `medium`) per loop round. A typical turn is two rounds: one that calls tools, one that answers. The loop stops after 8 rounds.
- **Request size stays bounded:** only the last 20 messages of a thread are replayed, as plain text. Older context comes back only if the agent recalls it.

## Swapping the database

Only `sqlite_store.py` knows about SQLite. A hosted store is one new class implementing [`MemoryStore`](vault/store.py), wired in `Vault.open`. For MongoDB Atlas, for example:

| `MemoryStore` method | SQLite + sqlite-vec (here) | MongoDB Atlas |
|---|---|---|
| `add_document` | rows in `documents`, `chunks`, `chunk_vectors` in one transaction | one `documents` doc + `chunks` docs with an `embedding` array field |
| `search_chunks` | `WHERE embedding MATCH ? AND k = ?` on `vec0` | `$vectorSearch` aggregation stage on a vector index over `chunks.embedding` |
| `add_message` / `get_messages` | `messages` table ordered by id | `messages` collection, index on `(thread_id, created_at)` |
| `add_note` / `search_notes` | `notes` + `note_vectors`, nearest-neighbour check before insert | `notes` collection, `$vectorSearch` with `limit: 1` for the duplicate check |
| `delete_document` / `delete_note` | delete vectors by hand (virtual tables skip foreign keys) | `deleteMany` on the chunk ids |

The connection string would be an optional setting; SQLite stays the default.

## Where this differs from the reference project, and why

| Change | Why |
|---|---|
| One SQLite file with sqlite-vec, behind a `MemoryStore` interface | No hosted database or account. Relational rows and vectors commit in the same transaction. The interface keeps a hosted store one class away. |
| Local embeddings (fastembed) | No embedding API key, and documents never leave the machine before you ask a question. |
| Notes are a separate, agent-controlled memory (`remember` / `recall` / `forget`) | Chat history is what was *said*; notes are what is worth *knowing*. Keeping them apart lets a fact cross threads without replaying whole conversations. |
| Hand-written tool loop with a saved trace | Every step is visible, and the trace powers the "memory used" and Vault views. |
| Near-duplicate check on `remember` | Agents repeat themselves. One nearest-neighbour query before the insert keeps the notes list clean. |

## Limitations (honest list)

- **Exact search.** sqlite-vec compares the query with every vector. That's fast into the hundreds of thousands of chunks, not millions. Beyond that, move to a store with an approximate index (the interface allows it).
- **The duplicate threshold is a judgement call.** At 0.04 cosine distance, close paraphrases are merged, but a reworded note can still slip through as a second copy. A tighter merge risks swallowing real updates ("March 3" vs "March 10" are only 0.07 apart with bge-small). Duplicates are harmless; you can delete them in the Vault tab.
- **Notes are only as good as the agent's judgement.** Whether a remark is worth remembering is the model's call, steered by the system prompt. It may save too little or too much.
- **Only the text of past turns is replayed.** The tool results behind an old answer aren't resent; if a follow-up needs them, the agent searches again.
- **Scanned PDFs** (no text layer) are rejected; they need OCR first. Chunking is by words, so tables and code blocks can be split mid-row.
- **One user.** Notes are global to the database file. A multi-user version would add a `user_id` column to threads and notes and filter on it.

## What I learned

- **A vector database can be a table.** sqlite-vec's `vec0` virtual table sits next to ordinary tables and shares their ids, so "search, then join back to text" is a single SQL statement with a CTE. Vectors and the rows they describe commit together, which a separate vector service can't give you.
- **Pin a database to its embedding model.** Vectors from two models live in different spaces, and comparing them doesn't fail, it just returns nonsense. Storing the embedder's name and size in a `meta` table turns a silent bug into a clear error.
- **Three memories, three access patterns.** The knowledge base is searched by meaning; history is read in order by recency; notes are searched by meaning *and* written by the agent. One generic "memory" abstraction would blur these; three groups of methods keep each honest.
- **Measure thresholds with the real model.** I first set the duplicate cut-off at 0.08. Checking pairs with bge-small showed "deadline is March 3" vs "March 10" at 0.074: an update would have been thrown away as a duplicate. So the threshold dropped to 0.04, and a `forget` tool was added so the agent replaces stale facts instead of piling up contradictions.
- **Return tool errors, don't raise them.** An unknown tool or a missing note id goes back to Claude as an `is_error` result. The model reads it and recovers (searches again, recalls the right id) instead of the whole turn crashing.
- **Echo the assistant turn back exactly.** Inside the tool loop, the response content (thinking blocks included) is appended unchanged before the tool results. Opus 5.5 always thinks, and editing those blocks would break the next request.
- **A trace is cheap observability.** Saving the list of tool calls next to each answer cost one JSON column and gave the UI its two most useful views.

## What was verified

- **Offline (25 pytest tests, no key, no download):** chunking and PDF page locations (with a hand-built PDF); knowledge-base search, content-hash dedupe and document deletion (vectors included); thread ordering, history window, cascade delete; note dedupe, recall, forget, notes surviving their thread; persistence across reopening the file; refusing a mismatched embedder. The tool loop with a scripted fake Claude: search-then-answer, a note saved in one thread and recalled in another, text-only history replay, parallel tool calls in one result message, duplicate and unknown-tool handling, forget-then-remember, the max-rounds stop, refusals. A wire test sends a full tool round trip through the real Anthropic SDK (2.8) with mocked HTTP and checks the request body (strict tools, fallbacks, effort, beta header, thinking block echoed with its signature). A Streamlit AppTest drives a chat, the Vault tab, Forget and New thread.
- **Live, keyless:** fastembed (bge-small) loaded and used on the sample documents: paraphrased questions ("can I stay for a month?") ranked the right chunk first, and a paraphrased note was merged as a duplicate. The app ran in a browser with that database: thread picker, chat history with its trace, Vault tab (notes, retrieved chunks per answer, direct search).
- **Not verified:** live Claude calls (no key was used), so whether the model chooses to `remember`, `recall` and `forget` at the right moments is untested. Uploading through the browser's file picker (ingest itself is tested in code). Hosted databases (the Atlas mapping above is a design, not code).
