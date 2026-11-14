# Memoir Avatar

A **talking companion that remembers you**. You chat with *Mira*, an animated avatar; every exchange is read by Claude and folded into a **local knowledge graph** of the people, places, projects and interests in your life. The next time you talk, Mira's answers draw on that graph, so "what should I get my sister for her birthday?" can be answered with what you said about Ada's pottery class three weeks ago.

- **Avatar:** a hand-drawn SVG face with lip-sync, blinking and breathing. The mouth follows the real loudness of the reply audio, frame by frame.
- **Voice:** free and local by default. **Kokoro** (open neural TTS, offline), **macOS `say`**, or the **browser's own speech**. **ElevenLabs** is an optional paid upgrade.
- **Voice input:** a 🎙️ button that uses the browser's speech recognition (Chrome, Edge, Safari).
- **Memory you can see:** an interactive graph view (pan, zoom, hover, click a node for its facts), a fact table, and per-message "🧠 used N memories" and "📝 memory: +2 new" notes. You can **forget** any entity or wipe everything.

**Keys:** only `ANTHROPIC_API_KEY`. The graph is a SQLite file on your disk. ElevenLabs shows up only if you set `ELEVENLABS_API_KEY`, and is never picked automatically.

## The idea in one paragraph

Chat history makes a poor memory: it grows without limit, and the answer you need often shares no words with the question. A **knowledge graph** stores what was *learned*, not what was *said*: `User -sister-> Ada -likes-> Pottery`. So each turn has two Claude calls with different jobs. The **reply** call gets the relevant slice of the graph; the **extraction** call reads the exchange and returns entities and relations as guaranteed-valid JSON. **Code owns the bookkeeping**: merging duplicates, resolving aliases ("Henry" is the User), counting how often a fact was said, and retiring facts that stop being true ("I moved to Berlin" retires *lives in Lagos* instead of contradicting it). **Recall needs no LLM and no embeddings**: find the entities the message mentions, then spread outward two hops through the graph. On the output side, every server voice returns the same raw audio format, so one 30-line function turns any reply into a lip-sync track.

## Architecture

```
                     ┌─────────────────────────── Streamlit (app.py) ───────────────────────────┐
  typed message ────►│  sidebar: voice · memory stats · forget / wipe                           │
  🎙️ mic (browser) ──►│  tab 1: avatar (SVG, lip-sync) │ chat (+ memories used, memory changes)  │
   SpeechRecognition │  tab 2: graph view (SVG: pan/zoom/click) + fact table                    │
                     └───────┬───────────────────────────────────────────────▲──────────────────┘
                             │ run 1                                          │ run 2 (avatar already talking)
                             ▼                                                │
   ┌ companion.py ───────────────────────────────────────────┐                │
   │ recall()  retrieve.py, plain Python, no LLM             │                │
   │   seed: entities named in the message (+ the User)      │                │
   │   spread: 2 hops, relevance ×0.5 per hop                │                │
   │   rank: relevance + recency + repetition → top 25 facts │                │
   │ reply()   Claude (effort medium): <memory> + chat → text│                │
   └───────┬──────────────────────────────────────────────────┘                │
           ▼                                                                   │
   ┌ voice.py ─────────────────────────────────────┐   ┌ avatar.py ──────────┐ │
   │ Kokoro │ say │ ElevenLabs → 24 kHz PCM        │──►│ WAV + envelope →    │ │
   │ envelope(): RMS per 1/30 s → mouth 0..1       │   │ <audio> drives mouth│ │
   │ browser: nothing here, the page speaks        │   │ (or word boundaries)│ │
   └───────────────────────────────────────────────┘   └─────────────────────┘ │
                                                                               │
   ┌ remember() ──────────────────────────────────────────────────────────────┴┐
   │ extract.py   Claude (effort low, structured JSON): entities + relations   │
   │              {exclusive: lives_in-style, negated: "I left Acme"}          │
   │ graph.py     merge() in one SQLite transaction: upsert, aliases, weights, │
   │              retire old exclusive facts, revive re-stated ones            │
   └──────────────────────────────┬────────────────────────────────────────────┘
                                  ▼
                       .memoir/memory.db (SQLite) ──► networkx view for recall + layout
```

## Code map: read in this order

| File | Role |
|------|------|
| [`memoir/config.py`](memoir/config.py) | Every knob: model, effort per job, recall budget (hops, max facts), audio format, envelope fps, default voices |
| [`memoir/llm.py`](memoir/llm.py) | The one door to Claude: `ask_text` for replies, `ask_json` (structured outputs) for extraction, refusal and `max_tokens` checks |
| [`memoir/graph.py`](memoir/graph.py) | **The memory.** SQLite schema, name normalisation, `merge()` (upsert, aliases, weights, exclusive and negated facts), networkx view, forget/reset |
| [`memoir/extract.py`](memoir/extract.py) | The JSON schema and the prompt that turn one exchange into entities and relations |
| [`memoir/retrieve.py`](memoir/retrieve.py) | **Recall.** Seed → spread → rank, with no LLM; returns the facts *and* its working, for the UI |
| [`memoir/companion.py`](memoir/companion.py) | The pipeline: `reply()` (recall + Claude) then `remember()` (extract + merge); persona prompt; `speakable()` |
| [`memoir/voice.py`](memoir/voice.py) | Four TTS engines and `envelope()`, the lip-sync track computed from PCM |
| [`memoir/avatar.py`](memoir/avatar.py) | Two `st.components.v2` components: the SVG avatar (lip-sync, blink, breathe) and the mic button |
| [`memoir/graph_view.py`](memoir/graph_view.py) | `spring_layout` positions in Python, an SVG graph with pan/zoom/hover/click in JS |
| [`app.py`](app.py) | Streamlit: the two-run flow (speak first, remember second), tabs, sidebar controls |
| [`tests/fakes.py`](tests/fakes.py) | A scripted fake Claude, a three-turn life story, and a "tone" TTS that makes real PCM |
| [`tests/test_graph.py`](tests/test_graph.py) | Identity and bookkeeping: aliases, reinforcement, exclusive/negated facts, forget, persistence |
| [`tests/test_retrieve.py`](tests/test_retrieve.py) | Two-hop recall, islands left out unless named, retired facts ranked lower, the budget |
| [`tests/test_companion.py`](tests/test_companion.py) | What Claude is sent each turn, grounding, memory failure keeps the answer, markdown stripping |
| [`tests/test_voice.py`](tests/test_voice.py) | Envelope shape, WAV round trip, ElevenLabs request (mocked), **live** macOS `say` lip-sync |
| [`tests/test_sdk_wire.py`](tests/test_sdk_wire.py) | The real `anthropic` SDK over a mock HTTP transport: the JSON on the wire |
| [`tests/test_app.py`](tests/test_app.py) | The Streamlit script headless: one chat turn end to end |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your Anthropic key in `.env`. (If you've run `ant auth login`, the SDK finds that profile and `.env` is optional.)

**Voices.** On a Mac, macOS `say` works out of the box; on any OS, "Browser speech" does. For a natural voice, install Kokoro and download its open model once (about 120 MB, from the kokoro-onnx GitHub release, saved to `models/`):

```bash
uv sync --extra kokoro
```

```bash
uv run python -m memoir.voice --download-kokoro
```

Run the tests (no key, no network; the live `say` test runs on a Mac):

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Introduce yourself ("I'm Henry, I live in Lagos, my sister Ada is into pottery"), then open the **Memory graph** tab. Come back later and ask something that needs it ("any gift ideas for my sister?"). Each message is **2 Claude requests** (reply + extraction). Recall itself is free.

Notes:
- Memory lives in `.memoir/memory.db`. Set `MEMOIR_DB=/path/other.db` to keep separate memories.
- If the browser blocks autoplay, the avatar shows **▶ Tap to hear Mira**.
- Voice input uses the browser's `SpeechRecognition`. In Chrome the audio is transcribed by Google's servers (no key, but not offline). Firefox and embedded webviews don't support it. Typing always works.

## What I learned

- **Split understanding from bookkeeping.** Claude is excellent at reading one paragraph ("I just moved to Berlin" means *lives_in Berlin*, and it replaces the old city) and unreliable at keeping a growing database consistent. So the LLM only emits a small JSON diff, and plain code merges it: identity by normalised key and aliases, weights for repetition, retirement for change. The graph stays correct even when the model rephrases.
- **Give the extractor the names you already know.** Listing existing entity names in the prompt ("use them exactly") does most of the coreference work: "my sister" becomes the existing `Ada` instead of a new node. Code-side normalisation (`"the Louvre" == "Louvre"`, `me`/`I` → User) catches the rest.
- **Facts have a lifetime.** Deleting a contradicted fact loses history, and keeping both makes the assistant contradict itself. A third state, *retired* (with the turn it ended), lets Mira say "you used to live in Lagos". Two flags carry it: `exclusive` (one value at a time) and `negated` ("I left Acme").
- **Graph recall beats keyword or vector search for personal context.** The question and the answer are often two hops apart and share no words. Spreading activation from the mentioned entities (plus the User, always) finds them in microseconds, with no embedding model, and the "why" is explainable: these seeds, these hops.
- **Speak first, remember second.** Extraction is a second LLM call. Splitting the Streamlit run in two (reply → `st.rerun()` → avatar starts talking → extract and merge) hides that latency behind the speech. A memory failure then costs only that turn's memory, never the answer.
- **Lip-sync from loudness is cheap and convincing.** RMS per 1/30 s window, normalised by the 95th percentile, gated and smoothed, read at `audio.currentTime` each animation frame. It isn't phonemes, but it's genuinely synced to the sound, and it works for any engine because every engine returns the same 24 kHz PCM.
- **`st.components.v2` changes what a Streamlit app can do.** The JS runs in the page (no iframe), so autoplay inherits the user's click, and `setTriggerValue` sends data back to Python with no build step. That one API gives the avatar, the mic and the clickable graph. Two gotchas: a component `key` must not clash with a `session_state` key, and look up browser APIs at use time rather than mount time.
- **The user owns the memory.** It's a local file, visible as a graph, with "forget this" and "wipe all" buttons. For a companion that collects personal facts, that's a feature, not an afterthought.
