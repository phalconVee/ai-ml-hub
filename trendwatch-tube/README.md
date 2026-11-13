# TrendWatch Tube

You pick some YouTube channels. A small crew of Claude agents watches their recent uploads (it reads the transcripts) and tells you **what's trending, why now, and which videos show it**. It runs as a Streamlit app.

Every file is heavily commented so it can be walked through line by line.

## The idea in one paragraph

**Code** collects the facts: recent videos, view counts and transcripts. An **Analyst** agent reads each video *on its own* and files a short digest with reusable topic tags. **Code** then counts those tags across videos and channels, weighting by *views per day*, so the evidence is real numbers rather than an LLM's guess. A **Trend Spotter** agent reads only the digests and the numbers, never the raw transcripts, and names the trends. An **Editor** agent turns that verdict into a short brief, streamed to the screen. Each agent sees only what its job needs.

## Architecture

```
 channels ──►  COLLECT            ANALYSE (parallel)          COUNT              SPOT TRENDS          WRITE
 (@handle,     youtube.py         agents.run_analyst ×N       signals.py         run_trend_spotter    run_editor
  URL, UC…)    ───────────►       ─────────────────►         ──────────►        ────────────►        ──────────►  brief.md
               RSS or Data API    worker model, effort=low   tags × channels    lead model,          lead model,
               + transcripts      → VideoDigest (schema)     × views/day        effort=high          streamed
               (no LLM)                                       (no LLM)           → TrendReport        (Markdown)
```

| Agent | Sees | Produces | Model / effort |
|-------|------|----------|----------------|
| Analyst | one video's metadata + transcript | `VideoDigest` (summary, topic tags, claims, tone, novelty) | worker (`claude-opus-5-5` by default) · low |
| Trend Spotter | all digests + topic signal table | `TrendReport` (3–7 trends, momentum, evidence ids, confidence) | lead (`claude-opus-5-5`) · high |
| Editor | the report + real video links | a Markdown brief under 600 words | lead · medium, streamed |

The orchestration is a fixed **workflow** in plain Python ([`pipeline.py`](trendwatch/pipeline.py)), not agents negotiating with each other. The steps are known in advance, so code sets the order and Claude does the judgement inside each step.

## Code map: read in this order

| File | Role |
|------|------|
| [`trendwatch/config.py`](trendwatch/config.py) | Every knob in one place: models, keys, time window, limits |
| [`trendwatch/models.py`](trendwatch/models.py) | The data shapes; `VideoDigest` and `TrendReport` are the contracts Claude must fill |
| [`trendwatch/youtube.py`](trendwatch/youtube.py) | The collector: channel refs → videos → transcripts (RSS or Data API) |
| [`trendwatch/llm.py`](trendwatch/llm.py) | The only file that calls Claude: structured outputs, effort, refusal fallbacks |
| [`trendwatch/agents.py`](trendwatch/agents.py) | **The core:** the three roles (system prompts plus input builders) |
| [`trendwatch/signals.py`](trendwatch/signals.py) | Views per day and the topic "buzz" table, computed in code |
| [`trendwatch/pipeline.py`](trendwatch/pipeline.py) | The orchestrator: fan-out, error tolerance, evidence checks; also a CLI |
| [`app.py`](app.py) | Streamlit UI: inputs, live progress, Brief / Trends / Signals / Videos tabs |
| [`tests/`](tests/) | Offline tests: fake YouTube (`httpx.MockTransport`), fake Claude, signal maths |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your `ANTHROPIC_API_KEY` in `.env` (or run `ant auth login`). `YOUTUBE_API_KEY` is optional.

Run the offline tests (no keys, no network, about 2 seconds):

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Or run it headless from the terminal:

```bash
uv run python -m trendwatch.pipeline @fireship @t3dotgg --days 7
```

### Keys and cost

| Key | Needed? | Without it |
|-----|---------|------------|
| `ANTHROPIC_API_KEY` | yes | nothing runs past collection |
| `YOUTUBE_API_KEY` | no | public RSS feeds: last 15 uploads per channel, view counts but no likes or comments; @handles resolved by reading the channel page (best-effort, may fail behind EU cookie walls) |

A run reads every transcript in full (up to 60k characters each; longer ones are clipped, and the UI says so). With Opus 5.5 as the worker, 15 videos is roughly a dollar or two. Set `TRENDWATCH_WORKER_MODEL=claude-haiku-5-5` to make the per-video step about 40× cheaper. These are estimates; no paid run has been measured yet.

## Design choices, and why

| Choice | Why |
|--------|-----|
| No LLM in collection | Fetching is deterministic. Code is cheaper, faster and can't "decide" to skip a channel. |
| One analyst per video, in parallel | Small, focused contexts. An analyst that can't see other videos can't invent a pattern. |
| Signals counted in code | LLMs are poor at counting and averaging over 30 items. A trend claim backed by a checkable number is more honest. |
| Views **per day**, `log` damped | 50k views in 2 days beats 200k in a month; `log` stops one viral video from drowning everything. Breadth (number of channels) is multiplied in, so one channel repeating itself doesn't look like a trend. |
| Structured outputs (`messages.parse` + Pydantic) | Replies are guaranteed to fit the schema. No JSON-in-prose parsing, no retries for bad format. |
| Effort per role (`low` / `high` / `medium`) | Opus 5.5 always thinks; effort sets how hard. Spend thinking where comparison happens, not on reading one video. |
| Server-side refusal fallbacks | Transcripts are arbitrary internet text; if a safety classifier declines, the API retries on a fallback model in the same call. Only sent for models that support it. |
| Transcripts in XML tags + "data, not instructions" | A video can say anything, including "ignore your instructions". The prompts mark it as material to analyse. |
| Evidence ids checked after the fact | The pipeline drops any `evidence_video_id` the trend spotter didn't receive, so every cited video is real. |
| Failures become warnings | One video without captions, or one refused analysis, shouldn't sink the whole run. |
| Data API uses `playlistItems`, not `search` | 1 quota unit instead of 100 per channel. |

## What was verified

- ✅ `uv run pytest`: 19 offline tests pass (channel-ref parsing, RSS parsing, date filter, transcript clipping, Data API path, signal maths, the full pipeline with a fake Claude, refusal handling, fallback parameters, SDK schema transform).
- ✅ Streamlit app renders and completes a full run with Claude and YouTube faked (`streamlit.testing.AppTest`): all four tabs, no exceptions, a failed analyst shown as a warning.
- ✅ Live keyless YouTube path: `@fireship` resolved to its channel id, RSS listed recent videos with view counts, and both transcripts downloaded.
- ❌ **Not yet run against the real Claude API** (no key in this environment). The first paid run is the real test of prompts, cost and output quality.
- ❌ Data API path tested only against mocked responses, not a real `YOUTUBE_API_KEY`.

## What I learned

- **Multi-agent is mostly about context, not conversation.** The value comes from giving each role a narrow view: one video, then only digests, then only the verdict. No agent ever "talks" to another; code passes the baton.
- **Put code between the LLM steps.** The analyst → *count* → trend-spotter sandwich turns vague "this seems popular" into numbers the model must reason over and you can check.
- **Make the schema do the prompting.** `Field(description=...)` text reaches the model through the JSON schema, so the contract and the instructions live in one place.
- **Velocity beats volume** for spotting what's *emerging*; breadth across channels beats repetition within one.
- **Design for partial failure.** Missing captions, a 404 channel, a refused request: each becomes a visible warning, and the run still finishes.
- **Inject dependencies and the whole system tests offline.** The collector takes an HTTP client and a transcript fetcher, the pipeline takes an LLM; swap in fakes and nothing needs a key.
