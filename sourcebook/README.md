# Sourcebook: a Research Notebook That Cites Its Sources

Add PDFs, notes and web pages to a notebook, then ask questions about them. Every answer is grounded **only** in your sources, and each claim carries a clickable `[n]` that jumps to the exact sentences it came from, highlighted inside their passage. Web citations also link to the live page, scrolled to that passage. The chat remembers the conversation, so follow-ups like *"what evidence supports that?"* work. One button turns the notebook into a short **audio overview**: two hosts talking through your sources, voiced by free local text-to-speech.

Built **from scratch** and kept small enough to read in one sitting. The loaders, chunker, BM25 search, rank fusion, memory, citation mapping and WAV stitching are all plain Python with no RAG framework. Every file is heavily commented so it can be walked through line by line.

**Keys:** `ANTHROPIC_API_KEY` is the only one you need. Everything else is free and local: PDF parsing (PyMuPDF), web pages (standard-library fetch and HTML cleaning), search (BM25, plus optional local embeddings), voices (macOS `say` or `espeak-ng`). ElevenLabs voices are optional and only used if you set `ELEVENLABS_API_KEY`.

## The idea in one paragraph

Each source is split into **passages** (about 120 words, never crossing a page), and each passage into **sentences**. A question first goes through **memory**: if it's a follow-up, Claude rewrites it into a standalone search query. **Hybrid search** finds the best passages: BM25 keywords, plus local embeddings if installed, fused by rank. The passages go to Claude as `search_result` blocks with citations enabled, **one text block per sentence**. Claude's answer comes back with citation objects that name the passage and the sentence range it used, so we can highlight those exact sentences. We never ask the model to type "[1]" and hope. For the audio overview, Claude writes a two-host script as **structured output** (guaranteed JSON), and a local voice reads it turn by turn into one WAV file.

## Architecture

```
 add source ─► sources.py ──────► chunk.py ───────────► retrieve.py
 (PDF/URL/     text per page      passages (~120 words)  BM25 ─┐
  text)        PyMuPDF / stdlib   └─ sentences            dense ┴─► RRF fusion
                                                          (optional, local)
                                       notebook.py: sources + index + memory ⇄ .sourcebook/notebooks/<name>/notebook.json

 question ─► memory.py ──────────► retrieve.py ──► answer.py ──────────────────────► app.py
             follow-up → standalone  top-8 passages   Claude (Opus 5.5)                answer with [n] links
             query (structured out)  (ticked sources)  user turn = search_result       → highlighted sentence cards
             + last 6 turns verbatim                    blocks (1 block per sentence)   → web links with #:~:text=
             + running summary of older turns           ◄─ text + search_result_location citations

 🎧 button ─► podcast.py ─────────────────────────► tts.py ──────────────────────► overview.wav
              digest of sources (word budget each)   say / espeak-ng / ElevenLabs     stdlib `wave` stitching,
              Claude → {title, turns[A|B]} JSON       one voice per host               pauses between turns
```

## Code map: read in this order

| File | Role | Step |
|------|------|------|
| [`sourcebook/config.py`](sourcebook/config.py) | Every knob in one place (model, effort, passage size, top-k, memory window, TTS) | – |
| [`sourcebook/sources.py`](sourcebook/sources.py) | PDF, text and URL loaders, including a 60-line HTML-to-text cleaner | 1 |
| [`sourcebook/chunk.py`](sourcebook/chunk.py) | Pages to passages to sentences (the citation unit) | 2 |
| [`sourcebook/retrieve.py`](sourcebook/retrieve.py) | BM25 from scratch, optional local embeddings, Reciprocal Rank Fusion | 3 |
| [`sourcebook/llm.py`](sourcebook/llm.py) | The one place that calls Claude: effort, refusal fallbacks, structured outputs | – |
| [`sourcebook/memory.py`](sourcebook/memory.py) | Conversation memory: recent turns, running summary, follow-up rewriting | 4 |
| [`sourcebook/answer.py`](sourcebook/answer.py) | **The core:** grounded answer with `search_result` citations mapped to sentences | 5 |
| [`sourcebook/notebook.py`](sourcebook/notebook.py) | Sources + index + memory, saved as readable JSON | 6 |
| [`sourcebook/podcast.py`](sourcebook/podcast.py) | Two-host script via structured outputs, from a budgeted digest of the sources | 7a |
| [`sourcebook/tts.py`](sourcebook/tts.py) | Voice backends and WAV stitching with the standard library | 7b |
| [`app.py`](app.py) | Streamlit UI: sources panel, chat with clickable citations, audio overview | 8 |
| [`tests/`](tests/) | Offline tests with a scripted fake Claude, a real-SDK wire test, and an AppTest click-through | – |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Then put your `ANTHROPIC_API_KEY` in `.env`. Optionally add hybrid search with local embeddings (about 70 MB, downloaded once, no key):

```bash
uv sync --extra dense
```

Run the tests. They take seconds and need no key:

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Add the two files in `sample_sources/` (or any PDF or article URL), then ask, for example, *"How do bees measure distance?"*, then *"And how sure are researchers about that?"*. Open **🎧 Audio overview** and click **Generate**.

Use it from Python without the UI:

```python
from sourcebook import Notebook, ask, load_url

nb = Notebook("Bees")
nb.add_source(load_url("https://en.wikipedia.org/wiki/Waggle_dance"))
answer = ask(nb, "Who first decoded the waggle dance?")
print(answer.markdown())
for c in answer.citations:
    print(c.number, c.location, "→", " ".join(c.sentences[c.start:c.end]))
nb.save()
```

### Voices

`SOURCEBOOK_TTS=auto` (the default) uses ElevenLabs if `ELEVENLABS_API_KEY` is set, else macOS `say` (Samantha and Daniel), else `espeak-ng`/`espeak` on Linux (`sudo apt install espeak-ng`). With none of them, you still get the script as a transcript. Force one with `say`, `espeak`, `elevenlabs` or `none`.

### What it costs

- **Adding sources and searching:** free; everything runs locally.
- **Each question:** one Claude call (Opus 5.5, effort `medium`), plus one small low-effort call to rewrite follow-ups (skipped on the first question), plus an occasional low-effort summary call when the memory window overflows.
- **Audio overview:** one Claude call for the script (input capped at about 12,000 words of sources). Local voices are free.

## Where this differs from the reference project, and why

| Change | Why |
|--------|-----|
| BM25 written from scratch, optional local embeddings, rank fusion | No vector database or embedding API key. BM25 is exact on names and numbers; embeddings catch paraphrases; RRF combines them without tuning. |
| Native `search_result` citations, one block per sentence | The API returns citation objects, not "[1]" strings to parse, and they point at sentence ranges, so the UI highlights exactly what was used. |
| Memory = recent turns + running summary + follow-up rewriting, saved as JSON | No memory service or extra key. Long chats stay bounded, and memory survives a restart. |
| Web pages via the standard library | No paid scraping API. The trade-off: pages that need JavaScript to show their text are rejected with a clear message. |
| Free local TTS by default, ElevenLabs optional | The audio overview works with no extra account; better voices are one env var away. |
| Script as structured output | The JSON shape is guaranteed, so there's no parsing of `HOST:` lines or stray stage directions. |

## Limitations (honest list)

- **Scanned PDFs** have no text layer and are rejected; they would need OCR first.
- **JavaScript-rendered pages** come back nearly empty and are rejected. Save them as PDF or paste the text instead.
- **The HTML cleaner is heuristic.** It prefers `<article>`/`<main>` and drops nav/footer, which suits articles and docs; busy pages can still carry some boilerplate.
- **Sentence splitting is regex-based.** Abbreviations like "Dr." can split a sentence early; the only effect is a slightly shorter or longer highlighted span.
- **Top-8 retrieval** means a question like "summarise everything" sees only 8 passages. The audio overview uses a broader digest instead.
- **Local voices sound synthetic.** macOS `say` is decent, espeak is robotic. ElevenLabs is the optional upgrade.
- **Text-fragment links** (`#:~:text=`) work in Chromium browsers and Safari; others open the page without scrolling.

## What I learned

- **Pick the citation unit on purpose.** Sending each passage as one text block lets Claude cite only "somewhere in this passage". Splitting it into one block per sentence makes the returned `start_block_index`/`end_block_index` point at exact sentences, which is what makes highlighting possible. The granularity of the content blocks *is* the granularity of the citations.
- **You don't need a tool call to get search-result citations.** `search_result` blocks can sit directly in the user message. Our app already knows what to retrieve, so one call replaces a whole tool loop.
- **Memory is three different problems.** Replaying context (recent turns), bounding cost (a running summary), and searching with the right words (rewriting follow-ups) each need their own fix. Searching for *"what evidence supports that?"* verbatim retrieves nothing useful.
- **Answer the original question, search with the rewrite.** The rewrite is for the retriever. Claude still sees the user's own words, so a bad rewrite can't silently change what gets answered.
- **Fuse ranks, not scores.** BM25 scores and cosine similarities live on different scales; adding them needs tuning. Reciprocal Rank Fusion just uses positions and works out of the box.
- **Audio needs fewer dependencies than you'd think.** Asking every backend for the same format (24 kHz, 16-bit, mono) lets the standard library's `wave` module join the clips, with silence as zero bytes. No ffmpeg, no pydub.

## What was verified

- **Offline (27 pytest tests, no key):** loaders (PDF pages, HTML cleaning, URL errors), chunking, BM25, RRF and hybrid fusion, source filtering, persistence, citation-to-sentence mapping, follow-up rewriting, memory summarisation, text-fragment links, podcast script and digest, WAV stitching. A wire test runs the pipeline through the real Anthropic SDK (2.8) with mocked HTTP; a Streamlit AppTest drives the chat end to end with a fake Claude.
- **Live, keyless:** a real Wikipedia page fetched and indexed; macOS `say` rendering a two-voice overview; local embeddings (fastembed, bge-small) downloaded and used for hybrid search; the app in a browser, with `[n]` links jumping to their highlighted citation cards and the audio player loading the stitched WAV.
- **Not verified:** live Claude calls (no key was used), real ElevenLabs output, `espeak-ng` on Linux.
