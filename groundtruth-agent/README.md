# GroundTruth Agent: Cited Answers from Messy PDFs

An agent that answers questions about your PDFs, including the parts most tools drop: **tables** and **figures**. Every claim in the answer links to the exact passage it came from, with file, page and the quoted text.

Everything is built **from scratch** and kept small enough to read in one sitting. The PDF parser, the chunker, the search engine (BM25 in about 60 lines) and the agent loop are all plain Python with no RAG framework. Every file is heavily commented so it can be walked through line by line.

## The idea in one paragraph

A PDF is not a string. Its answers often sit in a table or a chart that a plain text dump turns into noise. So we **parse** each page into typed blocks: prose, tables rebuilt as Markdown, and figures turned into words by Claude's vision. We **cut** them into passages that never split a table or cross a page, then **index** them with a keyword search engine. Claude gets that search engine as a **tool**. It decides what to search for, reads the results, searches again if they miss, and only then answers. Search results go back to Claude as citable `search_result` blocks, so the API itself attaches each sentence of the answer to the passage it used. We don't ask the model to write "[1]" and hope.

## Architecture

```
                 ┌──────────────────── ingest (once per PDF) ─────────────────────┐
  PDF bytes ──►  │ parse.py            chunk.py                index.py           │
                 │ text  ─┐                                                       │
                 │ tables ├─► Blocks ─► Chunks (page-bound, ─► BM25 index         │
                 │ figures┘  (page,kind)  tables kept whole)                      │
                 │   └─ Claude vision describes each figure (optional)            │
                 └────────────────────────────────────────────────────────────────┘
                                                                  ▲
  question ──► agent.py ──► Claude (Opus 5.5) ──tool_use──► tool.py: search_documents
                  ▲              │        ▲                       │
                  │              │        └── search_result blocks ┘  (citations on)
                  │              ▼
                  └──── answer text + search_result_location citations
                                 │
                        app.py (Streamlit): answer with [n], quoted sources, search trail
```

## Code map: read in this order

| File | Role | Brick |
|------|------|-------|
| [`groundtruth/config.py`](groundtruth/config.py) | Every knob in one place (model, effort, chunk size, top-k) | – |
| [`groundtruth/parse.py`](groundtruth/parse.py) | PDF to typed blocks: prose, tables as Markdown, figures via caption + vision | 1 |
| [`groundtruth/chunk.py`](groundtruth/chunk.py) | Blocks to passages: prose windows with overlap, tables and figures whole | 2 |
| [`groundtruth/index.py`](groundtruth/index.py) | BM25 keyword search, from scratch | 3 |
| [`groundtruth/tool.py`](groundtruth/tool.py) | The tool schema Claude sees, plus `search_result` blocks for native citations | 4 |
| [`groundtruth/agent.py`](groundtruth/agent.py) | **The core:** the hand-written agent loop, plus collecting citations | 5 |
| [`groundtruth/ingest.py`](groundtruth/ingest.py) | Glue: parse, chunk, index | – |
| [`app.py`](app.py) | Streamlit UI: upload, index, chat, sources, agent trace | 6 |
| [`tests/test_pipeline.py`](tests/test_pipeline.py) | Worked examples for every brick, with a scripted fake Claude | – |
| [`tests/test_sdk_wire.py`](tests/test_sdk_wire.py) | The agent through the real Anthropic SDK, with the network mocked | – |
| [`sample_docs/make_sample.py`](sample_docs/make_sample.py) | Builds a test PDF with prose, a ruled table, a vector chart and an image | – |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Then put your `ANTHROPIC_API_KEY` in `.env`.

Run the tests. They take seconds and need no key:

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Upload a PDF (or try `sample_docs/northwind_report.pdf`), click **Parse and index**, then ask, for example: *"How did gross margin change and why?"* or *"Which quarter had the highest revenue?"*

Use it from Python without the UI:

```python
from groundtruth import BM25Index, ask, ingest_pdf, make_claude_describer

index = BM25Index()
ingest_pdf(index, open("report.pdf", "rb").read(), "report.pdf", make_claude_describer())
answer = ask("What was revenue in 2024?", index)
print(answer.markdown())
for c in answer.citations:
    print(c.number, c.title, c.quote)
```

### What it costs

- **Indexing:** free without figure descriptions. With them, it's one small vision call per figure, capped at `max_figures_per_doc` (25).
- **Each question:** usually 2 to 4 Claude calls (search, maybe search again, answer). Effort is `medium`. Turn it down with `GROUNDTRUTH_EFFORT=low`, or point `GROUNDTRUTH_MODEL` at a cheaper model.

## Where this differs from the reference project, and why

| Change | Why |
|--------|-----|
| Our own parser (PyMuPDF + table detection + Claude vision) instead of a hosted parsing service | No extra account or key, and the parsing is visible and hackable instead of a black box. |
| A hand-written agent loop instead of a multi-agent framework | One agent with one tool is enough here, and the loop is about 40 readable lines. |
| BM25 instead of a vector database | Exact on numbers, codes and names, which is where report questions live. No embedding model needed. The agent rephrases to cover synonyms. |
| Native `search_result` citations | Grounding is enforced by the API's citation mechanism rather than by prompt instructions alone. |
| Claude Opus 5.5 with server-side refusal fallbacks | Current default model. If a safety classifier declines, the API retries on a fallback model within the same call. |

## Limitations (honest list)

- **Scanned PDFs** (images of text) have no text layer. They would need OCR first; PyMuPDF can do this with Tesseract installed, but it's not wired in.
- **Table detection** works best on ruled tables. Tables without borders, built only from whitespace, may come out as prose. Their numbers are still searchable, but the rows aren't aligned.
- **Figure detection** finds raster images and clusters of vector drawings. A chart without a "Figure N" caption, with vision turned off, is skipped because there is nothing to say about it in words.
- **Keyword search** misses pure synonyms ("turnover" vs "revenue") unless the agent re-searches. It usually does, but not always.
- The index lives in memory for the Streamlit session; nothing is saved to disk.

## What I learned

- **Parsing is the product.** Retrieval quality is capped by what the parser keeps. One fix moved answers more than any prompt change: rebuilding tables as Markdown with the header repeated on every cited row.
- **Tables need their own extraction path.** PyMuPDF's `to_markdown()` borrowed the heading *above* a table as its header row. `extract()` dropped spaces and moved decimal points (`41.2` became `412 .`). Reading each cell by clipping the page to the cell's rectangle was the reliable fix. Check what the parser really produces, never what the docs promise.
- **Charts are often not images.** Most report charts are vector drawings, made of hundreds of lines and rectangles. `page.cluster_drawings()` groups them into one box; table borders are drawings too, so they have to be filtered out.
- **Let the API do the citing.** Returning `search_result` blocks means citations come back as structured data (`source`, `title`, `cited_text`, block range), not as text to regex. Splitting each passage into small text blocks (paragraphs, table rows) makes the quoted evidence tight.
- **The "agentic" part is the retry.** A single retrieve-then-answer pass fails quietly when the first query misses. Giving the model a search tool and telling it to search again with different words fixes most misses, and a round cap with `tool_choice: none` keeps that from running forever.
- **Send the whole assistant turn back.** Thinking blocks must go back to the API unchanged, so the loop appends `response.content` as is, never just the text.
