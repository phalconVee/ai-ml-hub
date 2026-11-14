# Sightline

A live webcam vision app built on **Jev**, TypeSafe's System One decision model. Point your laptop camera at your desk and Sightline asks Jev several **typed questions about the same frame in one request**:

- **Classify** (Choice): which of *your* labels fits ("mug, phone, laptop, book, nothing"), with a probability for every label.
- **Locate** (Noul + Box): for each object you name, *is it there?* and *where?*, drawn as a box on the frame.
- **Check** (Noul): free-form yes/no questions such as "Is a person present?".
- **Quality gate** (Score): how clearly the frame shows anything at all.

Then a **decision layer** in plain Python takes over: thresholds you set in the sidebar decide which answers are accepted automatically and which go to a **human review queue**, where you confirm or correct them. Every decision and every human verdict is appended to a local log, shown as a history table.

It runs with **no key** in demo mode (canned answers on a drawn sample image), so you can click through everything before getting a Jev key.

## Why a decision model, not an LLM

An LLM can look at a photo and *write* "I think that's a mug". For software that has to act on the answer, three things matter more than prose:

| | Vision LLM | Jev (decision model) |
|---|---|---|
| Output | Text you parse (and hope it's in your label set) | Typed: `choice` is always one of your labels; `box` is always four numbers in 0..1 |
| Uncertainty | "I'm fairly sure", or a confidence it made up | A **calibrated** probability for every label, plus `confidence` derived from that distribution |
| Several questions | One prompt per question, or one long prompt and fragile parsing | One request, many typed questions, each answered against the same state |
| Who decides? | The model's wording | **Your code**: the threshold is a line you can read ("act when the chance of a wrong label is under 20%") |

The point of this app is the last row. Jev gives numbers you can trust as probabilities; the app shows what you can build on them: auto-accept, send to a person, or ignore.

## Architecture

```
  ┌─────────────── Streamlit (app.py) ────────────────┐
  │ Snapshot: st.camera_input (browser camera)        │
  │ Live:     camera.py, OpenCV, 1 frame / N seconds  │──┐
  │ Sample:   frames.make_sample_image()              │  │ raw frame
  └───────────────────────────────────────────────────┘  ▼
                                                 pipeline.analyze()
   frames.py      upright RGB, long side ≤ 1024 px, sides ≥ 280 px, JPEG
   questions.py   Plan (sidebar) → typed questions → ONE request body
                    state  {"frame": <uuid>}   assets {<uuid>: part "frame"}
                    classify (choice) · present_x (noul) + box_x (box) · check_i (noul) · quality (score)
   jev.py         POST /v1/systemone, multipart [request JSON, frame.jpg]
                    header TypeSafe-Danger-Zone-Images: true · retry 429/529
                    → typed answers: ChoiceAnswer, NoulAnswer, BoxAnswer, ScoreAnswer
                  (demo.py answers the same request offline, with canned values)
   decisions.py   answers + Thresholds → auto / review / skipped, with a reason
   log.py         append-only JSONL: one row per decision, one per human verdict
                                                          │
  ┌─────────────── app.py tabs ──────────────────────────▼────────────────────┐
  │ Camera: frame + boxes (green auto, amber review) · probability bars · table│
  │ Review queue: confirm or correct each "not sure" answer                    │
  │ History: the decision log, with counts by status                           │
  └────────────────────────────────────────────────────────────────────────────┘
```

### The decision rules (decisions.py)

| Answer | Number compared | Rule (defaults, all in the sidebar) |
|---|---|---|
| Choice | P(winning label) | ≥ 0.80 → auto, else review |
| Noul | P(yes) | ≥ 0.70 → "yes", ≤ 0.30 → "no", between → review |
| Box | its presence Noul, then box confidence | absent → **skipped** (Box always returns a box, even for nothing); unsure → review; confidence ≥ 0.60 → auto; `null` → review |
| Score | expected level (0..2) | < 1.0 → the frame is too poor: **every** auto answer is demoted to review |

These follow Jev's own guidance: threshold the chosen label's probability for a Choice (it reads as a risk statement), threshold the value itself for a Noul and give the middle band its own outcome, and pair every Box with a Noul because a Box has no "not found" answer.

## Code map: read in this order

| File | Role |
|------|------|
| [`sight/config.py`](sight/config.py) | Every knob: endpoint, model, images header, size limits, default thresholds |
| [`sight/frames.py`](sight/frames.py) | Frame prep to Jev's image limits; the drawn sample desk |
| [`sight/questions.py`](sight/questions.py) | **Typed questions:** Plan → Choice / Noul+Box / Noul / Score → one request body |
| [`sight/jev.py`](sight/jev.py) | **The thin client:** answer dataclasses, `parse_response()`, multipart POST with retries |
| [`sight/decisions.py`](sight/decisions.py) | **The point:** thresholds → auto / review / skipped, each with a readable reason |
| [`sight/draw.py`](sight/draw.py) | Box fractions → pixels; rectangles coloured by decision |
| [`sight/log.py`](sight/log.py) | Append-only JSONL decision log, human verdicts as new rows |
| [`sight/pipeline.py`](sight/pipeline.py) | One frame end to end: prepare → ask → route → log |
| [`sight/demo.py`](sight/demo.py) | Offline stand-in for Jev with canned answers that hit every branch |
| [`sight/camera.py`](sight/camera.py) | Live mode: OpenCV frame grab from the built-in camera |
| [`app.py`](app.py) | Streamlit: sidebar, camera sources, results, review queue, history |
| [`tests/`](tests/) | Wire format, parsing, routing, drawing, logging, and the app headless |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

```bash
cp .env.example .env
```

Put your TypeSafe key in `.env` (get one at [console.typesafe.ai/keys](https://console.typesafe.ai/keys); Jev is early access). Without a key the app opens in demo mode.

Run the tests (no key, no camera, a few seconds):

```bash
uv run pytest
```

Start the app:

```bash
uv run streamlit run app.py
```

Try it: pick **Sample image** and click *Analyse the sample desk* (demo mode). You'll see the laptop classified, three boxes, the book's box in amber because its confidence (0.48) is under 0.60, and one item in the review queue. Move the *Box* slider to 0.45 and analyse again: the book is now accepted automatically. That's the decision layer at work. Then switch to **Snapshot** or **Live** for your own camera.

### Camera permissions on macOS

- **Snapshot** uses the browser's camera (`st.camera_input`). The browser asks once; if you denied it, allow it in the address-bar camera icon. macOS also has to allow the browser itself: *System Settings → Privacy & Security → Camera → your browser*. `localhost` counts as a secure origin, so no HTTPS is needed.
- **Live** reads the camera from Python (OpenCV), because the browser widget only captures on click. macOS asks for permission for the app that **launched** Python: Terminal, iTerm, or VS Code. If the first try fails, allow it under *Privacy & Security → Camera* and restart that app.
- Live mode sends one frame every N seconds (default 5). **Each frame is one paid request**, so raise the interval or stop live mode when you're not watching.

## Design choices, and why

| Choice | Why |
|--------|-----|
| One request per frame, image in `state` once | Jev reads the state once and answers every question against it. Putting the image in each question's instructions would add a copy of it per question (more tokens, closer to the size limit). |
| A Noul next to every Box | A Box always returns four edges, even for an object that isn't there. The Noul is what lets code ignore phantom boxes. |
| Thresholds in code, set in the UI | The model reports how sure it is; the app decides what that's enough for. Moving a slider changes behaviour without touching the model or the prompt. |
| Quality Score demotes everything | A dark or blurred frame makes every other answer shakier, however confident each looks on its own. |
| Raw `httpx` multipart, not `typesafe-sdk` | To show the wire format (multipart part order, images header, raw answers). The SDK does the same with `Image(...)` objects and built-in retries; switching is a change inside `HttpJevClient.ask()` only. Multipart instead of base64 because base64 makes each image ~33% bigger. |
| Frames resized to ≤ 1024 px | Jev rejects (never resizes) images outside 280–1568 px a side. A laptop webcam has no more detail than ~1024 px to give, and smaller images cost fewer input tokens. |
| Append-only log | A correction never overwrites what Jev said, so the log holds both: exactly what you'd use to tune thresholds later. |
| Demo mode answers the real request | `DemoJevClient` reads the questions the app actually built and replies through the same parser, so demo mode exercises everything except the network call. |
| Frames not saved by default | It's your camera. Turn on *Save analysed frames* if you want them in `data/frames/` (git-ignored). |

## What was verified, and what wasn't

- ✅ 22 offline tests pass. They cover: the request body (one image in `state`, Box/Noul pairing, fresh UUID references, criteria shapes); the multipart upload read back byte by byte (JSON part first, image part name matching `assets[ref].part`, `Authorization` and `TypeSafe-Danger-Zone-Images` headers) through an `httpx.MockTransport`; retry on 429 honouring `retry-after`; a clear error on 400 and on a missing key; parsing every answer type, including a Box with `confidence: null`; every routing branch; box fraction → pixel conversion and drawing; the append-only log; the whole pipeline in demo mode; and the Streamlit app headless (`AppTest`): analyse the sample, see one queued item, save a verdict, queue empties, verdict logged.
- ✅ Ran the app in a browser in demo mode: sample analysis, boxes, probability chart, decision table and review queue render correctly.
- ⬜ **No live Jev call has been made.** There is no key in this environment (Jev is early access/paid). The request follows the current docs (Images, Box, API reference, checked November 2026), but the first real run is the true test. Images and Box are "Dreamer-ready" (pre-production) at TypeSafe: their format and accuracy can still change.
- ⬜ Camera capture (browser `st.camera_input` and OpenCV live mode) hasn't been exercised with a real camera here; the browser used for checking blocks camera access.
- ⬜ Default thresholds are starting points from Jev's docs, not tuned on real webcam frames. Use the history log to tune them.

## What I learned

- **A decision model changes where the "intelligence" lives.** Jev answers narrow typed questions with calibrated probabilities; everything that looks like *judgement about what to do* sits in ~130 lines of readable Python in `decisions.py`.
- **Confidence isn't one number for every type.** For a Choice, threshold the chosen label's probability; for a Noul, the value itself, with a middle band that means "ask a person"; a Box's confidence is only meaningful once a Noul says the object is there at all.
- **Typed outputs remove a whole class of bugs.** There is no parsing of free text: `choice` is always one of my labels and a box is always four fractions from the top-left corner, so drawing is one multiplication per edge.
- **One request, many questions.** Batching classify + locate + checks + quality into one call over one image is cheaper and faster than one call per question, and the answers are all about exactly the same frame.
- **Cost control is a UI decision.** Live video at 30 fps would be 30 paid requests a second. Sampling every N seconds, and saying so next to the slider, keeps a demo affordable.
- **Make the client swappable.** A one-method `JevClient` protocol let the demo client, the HTTP client and the test fakes all plug into the same pipeline.

## Ideas for next steps

- **Noul screen first** (a Jev pattern): ask the cheap "anything changed / anyone there?" Noul on every frame, and run the full question set only when it says yes.
- **What changed**: send the previous frame with the current one in `state` and ask about motion or arrivals.
- **Look closer**: when a Box is in review, crop to it at higher resolution and ask again before bothering a person.
- **Tune from the log**: compute, per threshold, how many reviews it caused and how many of those the human confirmed unchanged.
- Pin `jev-1.14.0` once thresholds are tuned, so an alias update can't shift them silently.
