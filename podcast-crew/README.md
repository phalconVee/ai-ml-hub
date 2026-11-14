# Podcast Crew

Turn **a web page or some pasted text into a two-host podcast episode**. Three Claude agents do the writing, and a free local voice records it:

1. a **Researcher** reads the source and outlines the episode: the hook, 3 to 6 key points with the detail that makes each land, the questions a listener would ask, and the takeaway;
2. a **Scriptwriter** turns that outline into a natural conversation between two hosts, a guide (*Maya*) and a curious co-host (*Theo*);
3. an **Editor** tightens the draft for accuracy, flow and speakability, working from a to-do list that plain code produced;
4. the **Studio** voices every line with that host's voice and stitches one WAV episode.

The Streamlit UI takes a link or text, shows live progress, then gives you an audio player, the script, the outline, the editor's notes, the first draft, and downloads (`.wav` and `.md`).

**Keys:** only `ANTHROPIC_API_KEY`. Pages are fetched with the standard library, and the default voices are free and local: **Kokoro** (an open neural TTS model, runs offline on CPU) or, with nothing to install, **macOS `say`**. **ElevenLabs** is an optional paid upgrade. It shows up only if you set `ELEVENLABS_API_KEY`, and is never picked automatically.

## The idea in one paragraph

A good podcast script is a chain of different skills: understanding a source, writing for the ear, and cutting. So each one is its own agent with its own brief, and **code decides the order** (there's no manager agent: the order never changes, so asking a model would only add cost). The writer never sees the raw page, only the outline, so the outline is a checkpoint you can read when something sounds off. Anything code can check perfectly (stage directions a TTS would read aloud, 80-word monologues, one host hogging the mic, length) is checked **by code** and handed to the editor as a to-do list. Then code gets the last word twice: an edit that breaks the script is thrown away, and `[laughs]`, `*sighs*` and markdown are stripped before any text reaches a voice. Every engine returns the same raw audio format (24 kHz mono 16-bit PCM), so "mixing" is joining bytes with short silences, with no audio library and no ffmpeg.

## Architecture

```
               ┌──────────────────────────── Streamlit (app.py) ─────────────────────────────┐
 link or text  │ hosts · length · style · engine + voices │ live log │ player · script · ⬇ │
 ─────────────►└───────────────┬─────────────────────────────────────────────────▲───────────┘
                               │ produce()                                         │ record()
                               ▼                                                   │
 ┌ source.py ─────── code ┐   crew.py runs these in a fixed order                  │
 │ URL? urllib + HTML     │                                                         │
 │ parser → article text  │──── too short / unreachable ──► friendly error, 0 tokens
 │ else: pasted text      │                                                         │
 └──────────┬─────────────┘                                                         │
            ▼                                                                       │
 ┌ researcher.py ─ Agent 1 ┐   JSON outline: title, hook, summary,                  │
 │ effort: medium          │   key_points[{point, detail}], questions, takeaway     │
 └──────────┬──────────────┘                                                        │
            ▼ outline only (never the raw page)                                     │
 ┌ scriptwriter.py Agent 2 ┐   JSON: {title, lines: [{speaker: "A"|"B", text}]}     │
 │ effort: high            │   the enum means no third, unvoiced speaker            │
 └──────────┬──────────────┘                                                        │
            ▼                                                                       │
 ┌ editor.py ────────────────────────────────────┐                                  │
 │ lint()  (code): stage directions, long lines, │                                  │
 │   monologues, host balance, URLs, length      │                                  │
 │ Editor (Agent 3, effort medium): fix + tighten│                                  │
 │ usable()? else keep the draft                 │                                  │
 └──────────┬────────────────────────────────────┘                                  │
            ▼ tidy(): strip [laughs] *sighs* markdown                               │
         final Script ─────────────────────────────────────────────────────────────┘
                                                         │
            ┌ studio.py ─────────────────────────────────▼───────────────┐
            │ per line: engine.synthesize(text, voice[A|B]) → PCM bytes  │
            │ lead-in · line · gap (250/450 ms) · line · … · lead-out    │
            │ wave module → episode.wav                                  │
            └──────┬───────────────────────┬────────────────────┬────────┘
                   ▼                       ▼                    ▼
            Kokoro (free, local)    macOS say (free)    ElevenLabs (paid, opt-in)
```

## Code map: read in this order

| File | Role |
|------|------|
| [`podcast/config.py`](podcast/config.py) | Every knob: model, per-agent effort, word rate, line limits, audio format, gaps, model URLs, default voices |
| [`podcast/llm.py`](podcast/llm.py) | The one door to Claude: structured JSON output, effort, refusal fallback, `stop_reason` checks |
| [`podcast/source.py`](podcast/source.py) | URL or text → clean article text, with a small stdlib HTML parser |
| [`podcast/researcher.py`](podcast/researcher.py) | **Agent 1:** source → episode outline |
| [`podcast/script.py`](podcast/script.py) | The script as data, `lint()` (the mechanical editor) and `clean_text()` (what a voice must never read) |
| [`podcast/scriptwriter.py`](podcast/scriptwriter.py) | **Agent 2:** outline → two-host dialogue, written for the ear |
| [`podcast/editor.py`](podcast/editor.py) | **Agent 3:** lint to-do list + LLM edit + the "never worse than the draft" rule |
| [`podcast/crew.py`](podcast/crew.py) | The producer: runs the steps in order and returns a `Production` |
| [`podcast/voices.py`](podcast/voices.py) | Three TTS engines behind one `synthesize()` method, plus engine choice and the Kokoro download |
| [`podcast/studio.py`](podcast/studio.py) | Voice each line, add pauses, write the WAV |
| [`app.py`](app.py) | Streamlit: settings, input, live progress, player, script, downloads, re-record |
| [`tests/fakes.py`](tests/fakes.py) | A scripted fake Claude, a sample HTML page, and a "tone" TTS that makes real PCM |
| [`tests/test_source.py`](tests/test_source.py) | Extraction: chrome dropped, entities decoded, duplicates removed, friendly errors |
| [`tests/test_crew.py`](tests/test_crew.py) | The crew offline: request settings, what each agent sees, broken edit rejected, cleaning, refusal, lint |
| [`tests/test_studio.py`](tests/test_studio.py) | Stitching: WAV format, exact length including gaps, right voice per host |
| [`tests/test_voices.py`](tests/test_voices.py) | Engine choice, the ElevenLabs request (mocked), and **live** local TTS checks |
| [`tests/test_sdk_wire.py`](tests/test_sdk_wire.py) | The real `anthropic` SDK over a mock HTTP transport: the wire JSON |
| [`tests/test_app.py`](tests/test_app.py) | The Streamlit script driven headless, end to end |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your Anthropic key in `.env`. (If you've run `ant auth login`, the SDK finds that profile and `.env` is optional.)

**Voices.** On a Mac you can skip this: macOS `say` works out of the box. For natural-sounding voices, install Kokoro and download its open model once (about 120 MB, from the [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) GitHub release, saved to `models/`):

```bash
uv sync --extra kokoro
```

```bash
uv run python -m podcast.voices --download-kokoro
```

When Kokoro is ready, "auto" picks it first. For ElevenLabs, add `ELEVENLABS_API_KEY` to `.env` and pick it in the sidebar.

Run the tests (no key, no network; the live `say` test runs on a Mac and the Kokoro test runs once the model is downloaded):

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Paste a link (an article or blog post works best) or some text, set the length and the hosts in the sidebar, and click **Produce the episode**. One episode is exactly **3 Claude requests**, then one TTS call per line. **Re-record** re-voices the same script with other voices or another engine, with no new Claude calls.

## Design choices, and why

| Choice | Why |
|--------|-----|
| Fetch pages in code, not with Claude's `web_fetch` | Free, keyless and testable. The researcher sees exactly the text the UI shows. A dead link fails before any tokens are spent. |
| Researcher → outline → writer | The writer works from a short digested brief, not 15k tokens of page. The outline is a readable checkpoint for "did it misunderstand, or mis-write?" |
| Fixed speaker keys `A`/`B` with a schema `enum` | Display names are only for prompts and the UI. The model can't invent a third speaker who has no voice. |
| Mechanical lint before the LLM editor | Stage directions, line length, monologues, balance and length are checked perfectly and for free. The editor spends its effort on what needs judgment. |
| `usable()` safety net + `tidy()` | The editor can improve a script but never destroy it. Nothing a voice would read aloud by mistake survives. |
| Structured outputs for all three agents | Outline and script arrive as validated JSON, so there's no prose parsing to break. |
| Effort `medium` / `high` / `medium` | Writing natural dialogue is the hard creative step. Reading and trimming need less thinking. |
| One raw audio format for every engine | 24 kHz mono 16-bit PCM in, so stitching is `b"".join(...)` plus silence. No pydub, no ffmpeg. |
| Longer pause on a speaker change (450 ms vs 250 ms) | A small detail that makes two synthetic voices sound like a conversation, not a list. |
| Free local TTS by default, paid only by name | `auto` never picks ElevenLabs, even when the key is set. Spending money is always an explicit choice. |
| Audio separate from writing | Re-voicing is free. The script is visible before the audio is done. |
| `fallbacks="default"` on every call | If a safety classifier declines, the API retries on a fallback model inside the same call. Remove it in `llm.py` if you don't want that. |

## What was verified, and what wasn't

- ✅ **23 offline tests pass, plus 1 live test** (`uv run pytest`, no key):
  - HTML extraction: script/style/nav/header/footer dropped, entities decoded, duplicate paragraphs removed, friendly errors for empty, short and unreachable input, truncation flag;
  - the crew with a scripted Claude: exactly 3 calls, model, effort per agent, JSON schema and fallbacks on each, the researcher sees the page, the writer sees only the outline, the editor sees `Name: text` lines plus the lint list, a broken edit keeps the draft, stage directions stripped, refusal and `max_tokens` handled;
  - the studio: valid WAV, exact frame count including every gap, the right voice per host, empty lines skipped;
  - the ElevenLabs request shape (URL, `output_format=pcm_24000`, header, body) with a mocked opener; `auto` never picks a paid engine;
  - the **real `anthropic` 2.8.0 SDK** over a mock HTTP transport: it accepts every parameter, and the `A`/`B` enum reaches the wire;
  - the Streamlit app headless (`AppTest`): paste a link, click, and get a title, script lines with host names, and a recorded episode.
- ✅ **Live, no keys:** macOS `say` recorded a 3-line, two-voice episode (8.5 s WAV, Samantha + Daniel, both found among 43 English voices); real pages extracted cleanly (a Wikipedia article: 26.7k characters; a blog post: 6.2k); `streamlit run app.py` starts and its health endpoint answers.
- ⬜ **Kokoro not run here.** Its model download (~120 MB) wasn't done in this session, so `test_live_kokoro_line` was skipped. The code follows the `kokoro-onnx` 0.6.1 API (`Kokoro(model, voices).create(text, voice=, speed=, lang=)` returns float samples at 24 kHz), and the two release URLs answer with 92 MB and 28 MB files. To run it: the two commands under "Voices", then `uv run pytest -m live`.
- ⬜ **Not run against the live Claude API** (needs a paid key). The requests pass the SDK's own validation and match the current docs, but the first real run is the true test of outline quality and how natural the dialogue sounds.
- ⬜ **ElevenLabs not called for real** (paid key). Only the request shape is tested.
- ⬜ JavaScript-only pages return little text. The app says so and suggests pasting the article instead.

## What I learned

- **Write for the ear, and enforce it in code.** "Don't write stage directions" in a prompt usually works. A regex that strips `[laughs]` before the voice sees it always works. Prompts set the intent, and code sets the floor.
- **A schema is a contract with the rest of the pipeline.** `enum: ["A", "B"]` costs one line and removes a whole class of bugs (a third speaker, a misspelt host, a line with no voice).
- **Editors need a "do no harm" rule.** An LLM edit is usually better and occasionally much worse. Keeping the draft when the edit is unusable makes the third agent safe to add.
- **Pick one audio format and make everyone speak it.** Once every engine returns 24 kHz mono 16-bit PCM, the "audio engineering" is a byte join. The `wave` module was all the mixing library this needed.
- **Fixed pipelines don't need a manager agent.** When the order of work never changes, a for-loop is the cheapest, most testable orchestrator.
- **Fakes beat keys for tests.** A fake Claude, a fake page and a "tone" TTS that makes real PCM ran the whole product (UI included) with no network and no keys, and the live tests cover what fakes can't.
- **Default arguments are frozen at import.** `def f(fetch=fetch_html)` captures the function once, so patching it later does nothing. Resolving it inside the function (`fetch = fetch or fetch_html`) fixed a test that passed alone and failed in the full suite.

## Ideas for next steps

- Stream the script into the page as it's written, and start voicing the first lines while the rest is still being edited.
- Voice lines in parallel (Kokoro and ElevenLabs are both happy with a few at once), and cache audio per line so re-recording after a small edit is instant.
- An MP3 export when `ffmpeg` is present, and a little intro/outro music bed.
- Multiple sources per episode (two articles that disagree make a great conversation), with the researcher noting who says what.
- An optional "fact-check" pass that compares each line against the source text, not just the outline.
