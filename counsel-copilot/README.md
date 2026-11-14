# counsel-copilot

A paralegal-style crew of three AI agents for your legal documents. Upload
contracts, policies or case documents, ask a question in plain English, and get
an answer back in plain English. Every claim in the answer is tied to the
exact clause (file, page, heading) or web page it came from, and every answer
opens with a **not legal advice** notice.

Built from scratch in plain Python: no agent framework, no vector database, and
no search API key. The only key it needs is `ANTHROPIC_API_KEY`.

> ⚠️ This is a learning project, not a legal tool. It can misread documents,
> and the law differs by jurisdiction. Don't rely on it for real decisions.

## What it does

| Agent | Job | Tools |
|---|---|---|
| **Clause Finder** | Acts like a paralegal with a highlighter. It searches the documents, picks out the relevant clauses, rates how well they cover the question, and lists any questions about the *law* the answer depends on. | `search_documents` (BM25 over clauses), `file_report` (strict, structured hand-off) |
| **Legal Researcher** | Runs only when the documents are not enough. It looks up legal context on the web, such as "is a 10% late fee enforceable in California?". | Claude's server-side `web_search` (no extra key) |
| **Drafter** | Writes the answer from the evidence alone. It keeps "what your document says" apart from "what the law says", flags ambiguities, and ends with questions to ask a lawyer. | none: it reads and writes |

## Architecture

```
 PDF / TXT / MD
      │  pypdf, page by page
      ▼
 ingest.py ── split at clause headings ("4.2 Late Payment", "ARTICLE IX") ──▶ Chunk(id, page, heading, text)
      │
      ▼
 search.py ── BM25 index over clauses
      │
      ▼
 ┌──────────────────────── crew.py ────────────────────────────────────────┐
 │                                                                         │
 │  question ─▶ Clause Finder ──ClauseReport──┐                             │
 │              (tool loop)    verified ids   │ coverage != "full"          │
 │                                            │ or legal questions?         │
 │                                  no ◀──────┴──────▶ yes                 │
 │                                  │                   ▼                  │
 │                                  │        Legal Researcher (web_search) │
 │                                  │                   │ quoted passages  │
 │                                  ▼                   ▼                  │
 │                    Drafter ◀── clauses + web passages, both as          │
 │                                `search_result` blocks (citations on)    │
 │                                  │                                      │
 └──────────────────────────────────┼──────────────────────────────────────┘
                                    ▼
          "Not legal advice" notice (added by code)  +  answer with [n]  +  sources
```

Three design decisions carry most of the weight:

1. **The clause is the unit.** Legal documents arrive already split into
   numbered clauses, so we chunk at those headings, never across a page and
   never across two clauses. A citation such as "lease.pdf, p.2, 5. Early
   Termination" is then always literally true.
2. **Trust, then verify.** The Clause Finder reports clause ids. Our code keeps
   only the ids it was actually shown by a search and drops any it made up. If
   nothing survives, its "full coverage" claim is downgraded to "none".
3. **One citation mechanism for everything.** The verified clauses and the web
   passages both reach the Drafter as `search_result` content blocks. Claude
   returns native citations (source, title, quoted text) for both, so we never
   ask the model to write "[1]" markers and never parse the answer with a regex.

Routing is plain Python (`needs_research`), not another LLM call. Web research
costs money and time, so it runs only when the report says it's needed. The
disclaimer is a constant that our code adds. A disclaimer the model includes
only 99% of the time is not a disclaimer.

## Code map (reading order)

| # | File | What to look for |
|---|---|---|
| 1 | `counsel/config.py` | Every knob in one place, plus the `NOT_LEGAL_ADVICE` text |
| 2 | `counsel/ingest.py` | The `HEADING` regex, page-safe clause chunks, the "(cont.)" carry-over |
| 3 | `counsel/search.py` | BM25 by hand, and why "shall", "not" and "may" are *not* stop-words |
| 4 | `counsel/llm.py` | The one door to Claude: model, effort, refusal fallback |
| 5 | `counsel/clause_finder.py` | The tool loop, the strict `file_report` hand-off, `verify()` |
| 6 | `counsel/researcher.py` | The server-side web search, `pause_turn`, turning web quotes into `search_result` blocks |
| 7 | `counsel/drafter.py` | Evidence in as `search_result` blocks, citations out, the notice on top |
| 8 | `counsel/crew.py` | The pipeline and the routing rule |
| 9 | `app.py` | The Streamlit UI: upload, chat, sources, crew trace |
| 10 | `tests/` | A fake Claude, a real-SDK wire test and a headless UI test |

## How to run

```bash
cd counsel-copilot
uv sync
cp .env.example .env        # then put your ANTHROPIC_API_KEY in .env
uv run streamlit run app.py
```

Click **Load the sample lease** in the sidebar, which loads an invented
residential lease in `sample_docs/`. Then try:

- *What happens if I pay rent late?* (the documents alone answer it)
- *Is the 10% late fee legal in California?* (needs the Legal Researcher)
- *Can I end the lease early, and what will it cost?*

Set a jurisdiction in the sidebar to focus the research. Switch off **Allow web
research** to keep everything inside your documents.

To regenerate the sample PDF: `uv run python sample_docs/make_sample.py`.

### Tests (offline, no key needed)

```bash
uv run pytest -q
```

- `test_ingest_search.py`: clause splitting on the generated PDF, page numbers,
  cross-page continuation, windows and BM25 ranking.
- `test_crew.py`: all three agents run against a scripted fake Claude. Covers
  routing, verification (dropped ids, downgraded coverage), the fallback when
  the finder never files a report, refusals, and that every request carries
  the model, effort and fallback settings.
- `test_sdk_wire.py`: the whole crew runs through the **real** `anthropic` SDK,
  with HTTP served by an in-process mock. This proves the SDK accepts our
  request bodies and parses citations into the attributes we read.
- `test_app.py`: a Streamlit `AppTest` loads the sample, asks a question, and
  checks the notice, the answer and the sources.

## Cost and limits

- A question costs 2–3 Claude Opus 5.5 calls, plus up to 4 web searches when
  research runs. Lower `effort` in `config.py` to make it cheaper.
- Scanned (image-only) PDFs yield no text. Add OCR before indexing them.
- The heading regex fits common contract styles. A document with unusual
  numbering falls back to one chunk per page section, which still works but
  gives coarser citations.
- BM25 matches words, not meanings. The Clause Finder compensates by
  re-phrasing its searches, but a clause that uses none of the question's
  words can still be missed.

## What I learned

- **Domain structure beats generic chunking.** A clause-aware splitter took
  about 30 lines of regex. It makes retrieval better and citations exact,
  which a fixed 500-token window can't do.
- **Stop-words depend on the domain.** Most English stop-word lists drop
  "not", "no", "shall" and "may". In a contract those words carry the
  meaning ("Tenant shall not…").
- **Hand-offs between agents should be data, not prose.** Ending the finder
  with a `strict` tool call (`file_report`) gives a schema-guaranteed object.
  Forced `tool_choice` is rejected on Opus 5.5, so the prompt says "call
  exactly once, last", and the code has a fallback for when the finder doesn't.
- **An LLM's claim is not evidence.** Checking the cited ids against what was
  actually retrieved costs one dictionary lookup and catches invented
  clause numbers.
- **`search_result` blocks unify citations.** Web quotes repackaged as
  `search_result` blocks get cited exactly like contract clauses, so the UI
  has one citation type to render.
- **Guarantees belong in code.** The disclaimer, the routing rule and the
  id verification are all deterministic. The model does the reading and
  writing, and the code enforces the rules.
