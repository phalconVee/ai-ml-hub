# soundcheck-mcp: an audio-analysis toolkit you can plug into any MCP client

Drop in a recording (a meeting, an interview, a podcast, a voice memo) and ask questions about it. An **MCP server** does the audio work: local Whisper transcription with timestamps, speaker labels, summaries, topics and entities, a sentiment curve over time, and search. A **Streamlit app** puts a Claude agent on top: Claude chooses which tools to call, reads the results and answers with timestamps you can jump to.

The server is the product; the app is one client of it. The same server plugs into Claude Desktop or Claude Code unchanged.

**Keys:** none for the audio side. Transcription, speakers, sentiment, search and the fallback summary all run on your machine. `ANTHROPIC_API_KEY` is needed for the chat agent (and makes summaries/topics better). AssemblyAI is an optional extra, used only if you ask for it.

## Architecture

```
                         ┌───────────────────────── Streamlit app (app.py) ─────────────────────────┐
  upload audio ────────▶ │  Upload: calls transcribe_audio directly          (no LLM)               │
                         │  Insights tab: sentiment_timeline / extract_topics (no LLM)               │
                         │  Chat tab: Claude agent loop (soundcheck/agent.py)                        │
                         │      user question → Claude → tool_use → MCP call_tool → tool_result → …  │
                         └──────────────────────────────────┬───────────────────────────────────────┘
                                                            │ MCP over stdio (toolbox.py)
   Claude Desktop / Claude Code ──── same protocol ────────▶│
                                                            ▼
          ┌──────────────────────── MCP server: soundcheck/server.py (own process) ──────────────────────┐
          │ transcribe_audio ──▶ transcribers.py ──▶ faster-whisper (local)  |  AssemblyAI (optional, paid) │
          │        │ saves                                                                                  │
          │        ▼                                                                                        │
          │  data/transcripts/<audio_id>.json  ◀── every other tool loads this, keyed by audio_id           │
          │                                                                                                 │
          │ get_transcript · search_transcript · sentiment_timeline  ──▶ analysis.py (local, deterministic) │
          │ identify_speakers  ──▶ speakers.py (MFCC voiceprints + k-means, numpy only)                     │
          │ summarize_recording · extract_topics ──▶ Claude (llm.py) if a key is set, else analysis.py      │
          │ resource transcript://{audio_id} · prompt analyze_recording                                      │
          └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

The key move is **transcribe once, analyse many times**. Transcription is the only slow step, so it happens once per file (cached by a hash of the audio bytes) and returns an `audio_id`. Every other tool takes that id and works on the saved transcript in milliseconds. Ten chat questions about a recording cost one transcription.

## Code map: read in this order

| File | Role |
|------|------|
| [`soundcheck/config.py`](soundcheck/config.py) | Every knob: engine, Whisper size, Claude models and effort, data folder |
| [`soundcheck/transcript.py`](soundcheck/transcript.py) | `Segment` / `Transcript`, the shared data structure, and the JSON store |
| [`soundcheck/transcribers.py`](soundcheck/transcribers.py) | Audio → segments. Local faster-whisper (default) or AssemblyAI (opt-in) |
| [`soundcheck/analysis.py`](soundcheck/analysis.py) | Local text analysis: sentiment curve, keywords, names, extractive summary, search |
| [`soundcheck/speakers.py`](soundcheck/speakers.py) | Speaker labels from scratch: MFCCs, k-means, silhouette, in numpy |
| [`soundcheck/llm.py`](soundcheck/llm.py) | The only file that calls Claude: structured outputs + one agent step |
| [`soundcheck/server.py`](soundcheck/server.py) | **The MCP server.** Eight tools, one resource, one prompt |
| [`soundcheck/toolbox.py`](soundcheck/toolbox.py) | MCP client: starts the server, converts its tools to Claude's format |
| [`soundcheck/agent.py`](soundcheck/agent.py) | The chat agent: a hand-written tool-use loop, ~50 lines |
| [`app.py`](app.py) | Streamlit UI: upload, transcript + search, insights, chat |
| [`tests/`](tests/) | Offline tests (fake Whisper, fake Claude, real MCP server) + one live test |

## The tools

| Tool | What it does | Needs |
|------|--------------|-------|
| `transcribe_audio(file_path, language?, engine?)` | Timestamped transcript → `audio_id` + preview. Cached | nothing (Whisper downloads once) |
| `list_recordings()` | What has been transcribed | – |
| `get_transcript(audio_id, start_seconds?, end_seconds?)` | `[mm:ss] Speaker: text` lines, pageable by time | – |
| `identify_speakers(audio_id, num_speakers?)` | Labels who speaks when, saves labels into the transcript, reports talk time | – |
| `summarize_recording(audio_id)` | Headline, summary, timestamped key points, action items | Claude, else local extractive |
| `extract_topics(audio_id)` | Topics with time ranges, named entities, keywords | Claude, else local |
| `sentiment_timeline(audio_id, window_seconds?)` | Mood per window (−1…+1), extremes, per-speaker averages | – |
| `search_transcript(audio_id, query)` | Exact, all-words and fuzzy (typo-tolerant) matches with context | – |

Plus a resource, `transcript://{audio_id}` (attach a transcript in Claude Desktop's "+" menu), and a prompt, `analyze_recording(file_path)`, which starts a full analysis.

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
cd soundcheck-mcp
```

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your `ANTHROPIC_API_KEY` in `.env` for the chat tab. Everything else works without it.

Start the app:

```bash
uv run streamlit run app.py
```

Upload a file. The first transcription downloads the Whisper `base` model (~145 MB) into `~/.cache/huggingface`. After that, transcription is local. To also stop the Hugging Face update check on each load, set `HF_HUB_OFFLINE=1`.

Run the tests (offline, no key, ~3 seconds):

```bash
uv run pytest
```

Run the live test too (real Whisper, real audio decoding, real stdio server; macOS only because it synthesises speech with `say`):

```bash
SOUNDCHECK_LIVE=1 uv run pytest tests/test_live.py -s
```

### Speed and accuracy knobs

- `SOUNDCHECK_WHISPER_MODEL=small` is noticeably better on accents and jargon, at roughly 3x the time of `base`. `tiny` is for quick checks.
- On Apple Silicon, CTranslate2 4.9 suggests its experimental MPS backend: `SOUNDCHECK_WHISPER_DEVICE=mps`. I haven't benchmarked it, so it isn't the default.
- Speaker labels are much more reliable if you say how many people there are ("it's a two-person interview").

## Use it from other MCP clients

The server speaks MCP over stdio, so any client that can launch a command can use it. Replace `/ABSOLUTE/PATH/TO` with where this repo lives.

**Claude Code:**

```bash
claude mcp add soundcheck -- uv --directory /ABSOLUTE/PATH/TO/ai-ml-hub/soundcheck-mcp run soundcheck-mcp
```

**Claude Desktop:** add this to `claude_desktop_config.json` (Settings → Developer → Edit Config), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "soundcheck": {
      "command": "uv",
      "args": ["--directory", "/ABSOLUTE/PATH/TO/ai-ml-hub/soundcheck-mcp", "run", "soundcheck-mcp"]
    }
  }
}
```

Then ask: *"Transcribe ~/Downloads/standup.m4a and tell me what each person committed to."* Paths are on the machine running the server, so use absolute paths. Inside Claude Desktop, Claude itself is the LLM, so the server doesn't need an Anthropic key. Without one, `summarize_recording` returns verbatim key sentences and Claude writes the summary from those.

To inspect the server by hand, use the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector uv --directory /ABSOLUTE/PATH/TO/ai-ml-hub/soundcheck-mcp run soundcheck-mcp
```

## Design choices

- **Local first, paid by name only.** `engine="auto"` always means local Whisper. AssemblyAI runs only when a call or `SOUNDCHECK_ENGINE` names it, so a key sitting in `.env` never starts costing money on its own.
- **Deterministic where it should be.** Sentiment and search use VADER and string matching, not an LLM. A mood curve should give the same numbers every time, and a search hit should be a real substring with a real timestamp.
- **Claude where it's clearly better, with a fallback.** Summaries and topic maps are much better from Claude, so the server uses it when a credential is present (structured outputs, low effort). If the call fails (no key, no credit, offline), the tool falls back to the local version and says so in a `note`. It doesn't fail.
- **A small diarizer instead of a big one.** Neural diarization (pyannote) means PyTorch, a gated Hugging Face model and gigabytes of downloads. `speakers.py` is ~200 lines of numpy (half of it comments) that you can read in one sitting. It's honest about being approximate: the tool output says "heuristic", and it falls back to "one speaker" when no split is clearly better.
- **Timestamps everywhere.** Every rendered line starts with `[mm:ss]`, so when a model quotes the transcript, it quotes where too. The agent's system prompt asks it to cite timestamps.
- **Paths, not bytes, over MCP.** Clients pass a file path. That keeps audio out of JSON-RPC and works the same for Claude Desktop (your files) and the app (it saves uploads to `data/uploads/` first).
- **Append-only chat history.** The agent appends whole response blocks (including thinking blocks) and never rewrites old turns. If a turn fails, the app rolls back to before the question, so history stays valid.

## What was verified, and what wasn't

Verified on 2026-11-13 (macOS, Apple Silicon, Python 3.12, `mcp` 2.3.0, `anthropic` 2.8.0, `faster-whisper` 1.2.1, `av` 18.1.0):

- **21 offline tests pass.** These cover the real MCP server in-process (tool schemas, caching, error messages, resource, prompt), the analysis functions, the diarizer on synthetic voices with known answers, the Claude-summary path and its fallback (fake Claude), and the agent loop (fake Claude, real server, parallel tool calls, errors fed back, turn budget).
- **Live test passes.** Two macOS voices (Samantha, Daniel) take turns. Local Whisper `base` transcribes them correctly over a real stdio subprocess server, and the heuristic diarizer finds 2 speakers with the right turn assignment. Search and sentiment then run on the result.
- **Streamlit app** renders with no key, checked with Streamlit's `AppTest` on a real transcribed clip: empty state, all three tabs, sentiment metric, search hits and the sidebar history work, with no exceptions.
- **The Claude Code / Claude Desktop launch command** (`uv --directory … run soundcheck-mcp`) completes an MCP handshake and serves tool calls from an unrelated working directory.

Not verified:

- **Live Claude calls.** I didn't spend API credit. Both the chat agent and Claude summaries ran against fakes. The request shapes match the pattern verified on the wire in `omni-scout/`.
- **AssemblyAI.** The REST flow (upload → job with `speaker_labels` → poll) is written from the public API docs, but only its "refuses without a key" path is tested.
- **Diarization on real human recordings.** It's tested only on synthetic and TTS voices so far. Expect it to struggle with crosstalk, phone audio and similar-sounding voices.
- **Claude Desktop itself.** I tested the config's command, not the app.

## What I learned

- **Transcribe once, then every question is cheap.** Making `transcribe_audio` return an id and caching by content hash turns a slow, expensive step into a one-off. All the "intelligence" tools then run in milliseconds on text.
- **Diarization is clustering, and clustering's hard part is choosing k.** MFCC means and standard deviations separate distinct voices easily. Deciding *how many* voices there are is the hard part. The live test caught a near-tie (silhouette 0.278 for 2 speakers vs 0.287 for 3) that produced a spurious one-line "Speaker 3". Two standard guards fixed it: at most one cluster per two segments, and a larger k must win by a clear margin.
- **Read the dependency chain, not just your dependencies.** faster-whisper 1.2.1 calls `av.open(metadata_errors=...)`, which PyAV 19 removed, so every real transcription crashed. Only the live test (real decoding, not the fake engine) found it. It's pinned to `av<19`, with a comment saying why.
- **The MCP stdio client doesn't pass your environment through.** A child server gets only a minimal environment (PATH, HOME…). Settings exported in your shell silently don't reach it. `toolbox.py` forwards just the `SOUNDCHECK_*`, `ANTHROPIC_*`, `ASSEMBLYAI_*` and `HF_*` variables.
- **MCP logging is deprecated as of the 2026-07-28 spec.** `ctx.info()` now warns. Progress notifications (`ctx.report_progress`) still carry a message, and blocking work runs in a thread that hands progress back to the event loop with `run_coroutine_threadsafe`.
- **Some tools should stay non-LLM even when an LLM is on hand.** Sentiment from a lexicon is cruder than Claude's judgement, but it's reproducible, explainable and free. That's what you want from a chart. Claude then *interprets* the curve in chat, which is where judgement adds value.
- **Sentence-start capitals aren't names.** The local entity extractor's best trick: a capitalised word that also appears lowercase, or only ever starts sentences, is a sentence start. "Then Maria Lopez" becomes "Maria Lopez".
