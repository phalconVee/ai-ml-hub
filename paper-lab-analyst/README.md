# Paper Lab Analyst

Upload a scientific paper (PDF) and get back a **structured intelligence report**: research question, methods, key findings with confidence ratings, an insight for each figure, limitations (marked as *stated by the authors* or *inferred by the reviewer*), open questions and practical implications.

A small team of Claude agents does the work. One reads each figure **as an image**, one transcribes scanned pages (OCR), two analysts draft methods and findings, and a sceptical **reviewer agent** checks the drafts against the paper with its own tools before writing the report. A final pass in plain Python looks up every quote in the report in the PDF and labels it ✅ verified, 📄 on another page, or ⚠️ not found.

One key: `ANTHROPIC_API_KEY`. OCR and figure reading use Claude's own vision, so there's no Tesseract install and no second API.

## Architecture

```
 PDF ──► pdf_extract.py (PyMuPDF, no LLM)
           ├─ text per page (reading order)
           ├─ scanned pages  → rendered PNG
           └─ figures: raster images + vector drawings
                       → group panels → pair with "Figure N" caption → crop PNG
                                    │
        ┌───────────────────────────┴──────────────────────────┐
        ▼  parallel                                            ▼  parallel
  vision.ocr_page  (scanned pages)                 vision.read_figure (each figure)
  Claude transcribes the image                     image + caption + citing sentences
        │                                                      │  → FIGURE_NOTE (JSON)
        └───────────────────────────┬──────────────────────────┘
                                    ▼  paper text + figure notes
               ┌──────────── analysts.py (parallel) ────────────┐
               │ methods analyst → METHODS_NOTE                 │
               │ findings analyst → FINDINGS_NOTE               │
               └───────────────────────┬────────────────────────┘
                                       ▼  drafts
               ┌──────────── reviewer.py (agent loop) ──────────┐
               │ Claude ⇄ search_paper / read_page / view_figure│
               │ ≤6 tool rounds, then must write the REPORT     │
               └───────────────────────┬────────────────────────┘
                                       ▼
                     grounding.py (no LLM): every quote looked up
                     in the PDF text; bad figure ids dropped
                                       ▼
                     app.py (Streamlit): Report · Figures · Drafts · Trace
                     report.py: Markdown download
```

The order of steps is fixed in code (`pipeline.py`). Only the reviewer gets to decide what to do next, and its freedom is bounded: three read-only tools and a turn budget.

## Code map: read in this order

| File | Role |
|------|------|
| [`paperlab/config.py`](paperlab/config.py) | Every knob: model, effort, page/figure/OCR caps, reviewer turn budget |
| [`paperlab/pipeline.py`](paperlab/pipeline.py) | **Start here for the big picture.** The six steps in order, with parallelism |
| [`paperlab/pdf_extract.py`](paperlab/pdf_extract.py) | PDF → pages, scanned-page detection, **figure finding** (pieces → panels → captions) |
| [`paperlab/llm.py`](paperlab/llm.py) | The single call to Claude: model, effort, refusal fallback, image blocks, JSON parsing |
| [`paperlab/schemas.py`](paperlab/schemas.py) | JSON Schemas for figure notes, drafts and the final report (structured outputs) |
| [`paperlab/vision.py`](paperlab/vision.py) | OCR of scanned pages, figure reader, finding the sentences that cite a figure |
| [`paperlab/analysts.py`](paperlab/analysts.py) | Methods and findings analysts, and the shared paper context |
| [`paperlab/paper_tools.py`](paperlab/paper_tools.py) | The reviewer's tools, including a small TF-IDF search over the paper |
| [`paperlab/reviewer.py`](paperlab/reviewer.py) | **The agent loop:** tool calls, tool results, turn budget, final JSON |
| [`paperlab/grounding.py`](paperlab/grounding.py) | Quote verification with a sliding-window word match |
| [`paperlab/report.py`](paperlab/report.py) | Report → Markdown |
| [`app.py`](app.py) | Streamlit UI |
| [`tests/conftest.py`](tests/conftest.py) | Synthetic 3-page paper built in code, plus a scripted fake Claude |
| `tests/test_*.py` | Extraction, grounding, tools, the full pipeline, and the UI render |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your Anthropic key in `.env`. (If you've run `ant auth login`, the SDK finds that profile and `.env` is optional.)

Run the tests (no key needed, about 2 seconds):

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Upload a paper and press **Analyse paper**. You can watch each step (and each reviewer tool call) appear as it runs.

**Cost control.** Each figure and each scanned page is one vision call, so `max_figures` (8) and `max_ocr_pages` (10) in `config.py` are the main levers. A typical 10-page paper with 8 figures makes about 12–18 requests.

## Design choices, and why

| Choice | Why |
|--------|-----|
| Claude vision for OCR and figures | No Tesseract binary, no OCR API key. It also reads tables, equations and plot axes, which classic OCR doesn't understand. |
| Figures found from **both** raster images and vector drawings | Most charts in papers are vector paths, not images. Looking only for embedded images (the common shortcut) misses them. |
| Figure reader gets the sentences that cite the figure | Reading "As Fig. 3 shows, X rose by 40%" alongside the image stops the model from guessing what the figure is meant to prove. |
| Two narrow analysts instead of one big prompt | One "analyse everything" prompt tends to skim the methods. Narrow jobs give deeper notes, and running them in parallel costs no extra time. |
| A reviewer agent with tools | The drafts are claims. The reviewer re-checks numbers, looks at figures itself and hunts for missed limitations, then records what it changed in `reviewer_notes`. |
| Turn budget, then `tool_choice: none` | An agent loop needs an exit. After 6 rounds the reviewer is told the budget is spent and can no longer call tools. (Opus 5.5 can't be *forced* to call a tool, but `none` is allowed.) |
| Structured outputs everywhere except OCR | Every agent's answer is JSON that matches a schema, so parsing never breaks on extra prose. |
| Grounding check in plain Python | An LLM can paraphrase or misplace a quote while sounding exact. Looking each quote up in the PDF is cheap and turns "trust me" into a visible ✅/⚠️. |
| `fallbacks="default"` on every call | If a safety classifier declines a request (biology papers can trip the `bio` classifier), the API retries it on a fallback model within the same call. Remove it in `llm.py` if you don't want that. |
| `claude-opus-5-5`, effort `medium` | Current default Opus. Raise effort to `high` in `config.py` for dense papers. |

## What was verified, and what wasn't

- ✅ **26 offline tests pass** (`uv run pytest`, ~2 s, no key). They cover extraction on a synthetic paper (raster + vector figures, captions, scanned-page detection, page limit), caption-regex cases, stacked panels, a figure between two captions, quote grounding (verified / wrong page / not found, tolerance to PDF noise), the reviewer's tools, the full pipeline with a scripted fake Claude (OCR text flows into the analysts, `tool_result`s matched by `tool_use_id`, an image returned inside a tool result, the turn budget and `tool_choice: none`, refusal → readable error), model/effort/fallback on every request, and a headless Streamlit render (`AppTest`).
- ✅ **The real `anthropic` SDK (2.8.0) accepts every request.** The whole pipeline was run through the actual SDK with only the HTTP layer mocked. That covers `fallbacks` + beta header, `output_config` with effort + JSON schema alongside tools, `strict` tools, an image inside a `tool_result`, and replaying thinking + `tool_use` blocks.
- ✅ **Extraction tested on 10 real papers** (journal and conference papers and a textbook chapter, 1–45 pages). Doing this found and fixed four bugs:
  - figures built from several small images were dropped;
  - top panels far above the caption were lost;
  - a figure between two captions took the wrong one;
  - body text like "Figure 2 shows…" was read as a caption.

  Book-style "Figure 9.1" numbering also works now.
- ✅ The Streamlit server boots (`/_stcore/health` → ok).
- ⬜ **Not run against the live API** (needs a paid key). The request shapes are checked against the SDK, but the quality of the figure readings and the report isn't tested yet.
- ⬜ **OCR on a genuinely scanned PDF** is only tested with the fake client (the scanned-page *detection* is tested for real).
- ⬜ **Known extraction limits:**
  - A "figure" drawn mostly as text (boxes made of words, as in one of the 10 papers) has no image or drawing to crop, so it's skipped. Its caption text still reaches the analysts.
  - Free-floating text labels at a figure's edge can be cropped off.
  - Tables are not extracted as figures. They reach the agents as text.

## What I learned

- **Figures in PDFs are mostly not images.** Charts from matplotlib/R are vector paths. You find them by clustering drawings (`cluster_drawings`), not by listing embedded images, and a figure is often many pieces that only a caption ties together.
- **Real documents beat synthetic tests.** The synthetic tests passed from the start. Running 10 real papers found four grouping/captioning bugs in a few minutes. Each bug became a regression test.
- **Vision is a tool, not just an input.** Returning an image inside a `tool_result` lets the reviewer decide *for itself* to look at a figure, rather than having every image put into one huge prompt.
- **Agentic where judgement helps, fixed code where order matters.** OCR must happen before analysis, and analysis before review. That's a pipeline. *Which* claims deserve a second look is a judgement call, so that part is the agent.
- **Treat model output as claims, then check them cheaply.** Quotes are easy to check (string search), so check them all. A ⚠️ next to a finding is more useful than a confident paragraph.
- **Make the LLM swappable.** Passing `client` in let one fake, which routes by system prompt so it's safe across threads, test the whole multi-agent flow for free.

## Ideas for next steps

- Extract tables as structured data (PyMuPDF `find_tables`) and give the reviewer a `read_table` tool.
- For captions with nothing to crop, render the block of text above the caption as an image instead.
- Compare two papers side by side (same schema, a "contrast" agent).
- Stream the reviewer's progress notes to the UI instead of waiting for the full report.
