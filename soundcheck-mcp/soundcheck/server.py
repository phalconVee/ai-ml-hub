"""The MCP server: the audio toolkit, exposed as tools any MCP client can call.

Why MCP rather than importing these functions into the app directly?
  * The same server plugs into Claude Desktop, Claude Code, Cursor, or our own
    Streamlit app unchanged (see README: "Use it from other MCP clients").
  * It runs in its own process: a 10-minute Whisper job can't freeze the UI.
  * The client needs no knowledge of Whisper, VADER or numpy. It sees a list
    of tool names, JSON schemas and descriptions, and that's all.

Flow every client follows:
    transcribe_audio(file_path)  ->  audio_id   (slow, once per file, cached)
    any other tool(audio_id, ...)               (fast, reads the cached transcript)

Run it standalone (speaks MCP over stdin/stdout):

    uv run soundcheck-mcp          # or: uv run python -m soundcheck.server
"""

import asyncio
import json
from pathlib import Path

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from soundcheck import analysis, speakers, transcribers
from soundcheck import transcript as store
from soundcheck.config import AUDIO_EXTENSIONS, settings
from soundcheck.llm import LLM, Summary, TopicMap
from soundcheck.transcript import Transcript, fmt_ts

# `instructions` is sent to the client when it connects. Good clients put it in
# the model's context, so this is where we teach any agent the workflow.
server = MCPServer(
    "soundcheck",
    instructions=(
        "Audio analysis tools. Always start with transcribe_audio(file_path) to get an audio_id "
        "(results are cached, so calling it again on the same file is instant). Then use the other "
        "tools with that audio_id. Quote timestamps (mm:ss) when you cite what was said."
    ),
)


def _json(obj) -> str:
    """Tool results are text. Indented JSON keeps them readable for models and humans."""
    return json.dumps(obj, ensure_ascii=False, indent=1)


def _load(audio_id: str) -> Transcript:
    """Load a transcript or fail with a message the model can act on."""
    try:
        return store.load(audio_id)
    except KeyError as exc:
        # ToolError = an *expected* failure: MCP passes our text through to the
        # model. Any other exception would reach it as a bare "Error executing tool".
        raise ToolError(str(exc).strip('"')) from exc


def _check_audio_path(file_path: str) -> Path:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise ToolError(f"File not found: {path}. Pass an absolute path to an audio file on this machine.")
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise ToolError(f"{path.suffix or 'No extension'} doesn't look like audio. Supported: {sorted(AUDIO_EXTENSIONS)}")
    size_mb = path.stat().st_size / 1e6
    if size_mb > settings.max_upload_mb:
        raise ToolError(f"File is {size_mb:.0f} MB; the limit is {settings.max_upload_mb} MB (SOUNDCHECK_MAX_UPLOAD_MB).")
    return path


def _decode(path: str):
    """Audio file -> 16 kHz mono float32 numpy array (via PyAV's bundled FFmpeg).

    A separate function so tests can swap in synthetic audio.
    """
    from faster_whisper import decode_audio

    return decode_audio(path, sampling_rate=speakers.SAMPLE_RATE)


def make_llm() -> LLM:
    """Factory, so tests can substitute a fake Claude."""
    return LLM()


# =============================================================================
# Tools. Each one is: a typed signature (-> the JSON schema clients see), a
# docstring (-> the description the model reads when choosing a tool), and a
# few lines of work. The docstrings are prompts; write them for the model.
# =============================================================================


@server.tool()
async def transcribe_audio(file_path: str, ctx: Context, language: str | None = None, engine: str = "auto") -> str:
    """Transcribe an audio file (wav, mp3, m4a, flac, ogg, webm, mp4...) into timestamped text.
    Returns an audio_id that every other tool takes, plus a preview of the transcript.
    file_path: absolute path on the machine running this server.
    language: ISO code like 'en' or 'fr' if known; omit to auto-detect.
    engine: 'auto' or 'local' (free, on-device Whisper) or 'assemblyai' (paid; needs a key; real speaker labels).
    Slow the first time (roughly 1/5 to 1/10 of the audio's length on a laptop CPU), instant when cached."""
    path = _check_audio_path(file_path)
    audio_id = store.audio_id_for(path)
    try:
        engine_name = transcribers.resolve_engine(engine)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc

    if store.exists(audio_id):
        t = store.load(audio_id)
        cached = True
    else:
        # The transcriber is blocking CPU work. Run it in a worker thread, and
        # forward its progress callbacks back onto our event loop as MCP
        # progress notifications (clients that show progress bars use these).
        loop = asyncio.get_running_loop()
        last = [0.0]

        def on_progress(fraction: float) -> None:
            if fraction - last[0] >= 0.05 or fraction >= 1.0:  # throttle: ~20 updates per file
                last[0] = fraction
                asyncio.run_coroutine_threadsafe(
                    ctx.report_progress(fraction, 1.0, f"Transcribing {path.name}"), loop)

        # (No ctx.info() log line here: MCP's logging capability is deprecated
        # as of the 2026-07-28 spec. Progress notifications carry the message.)
        await ctx.report_progress(0.0, 1.0, f"Transcribing {path.name} with {engine_name}")
        try:
            result = await asyncio.to_thread(transcribers.transcribe, path, engine_name, language, on_progress)
        except Exception as exc:
            raise ToolError(f"Transcription failed: {type(exc).__name__}: {exc}") from exc
        t = Transcript(
            audio_id=audio_id, source_name=path.name, source_path=str(path),
            duration=result.duration, language=result.language, engine=result.engine,
            segments=result.segments, speaker_method=result.speaker_method,
        )
        store.save(t)
        cached = False

    preview = t.render()
    return _json({
        "audio_id": t.audio_id,
        "file": t.source_name,
        "path": t.source_path,
        "duration": fmt_ts(t.duration),
        "duration_seconds": t.duration,
        "language": t.language,
        "engine": t.engine,
        "segments": len(t.segments),
        "words": len(t.text.split()),
        "speakers_labelled": t.speaker_method,
        "cached": cached,
        "preview": preview[:3000] + ("\n...[use get_transcript for the rest]" if len(preview) > 3000 else ""),
    })


@server.tool()
async def list_recordings() -> str:
    """List every recording already transcribed (audio_id, file name, duration, language).
    Use this when you don't know the audio_id."""
    return _json([
        {"audio_id": t.audio_id, "file": t.source_name, "path": t.source_path, "duration": fmt_ts(t.duration),
         "language": t.language, "engine": t.engine, "speakers_labelled": t.speaker_method}
        for t in store.list_all()
    ] or "No recordings yet. Call transcribe_audio first.")


@server.tool()
async def get_transcript(audio_id: str, start_seconds: float = 0, end_seconds: float | None = None,
                         max_chars: int = 15000) -> str:
    """Return the transcript as '[mm:ss] Speaker: text' lines, optionally only a time range.
    For long recordings, page through with start_seconds/end_seconds instead of asking for everything."""
    t = _load(audio_id)
    segs = t.window(start_seconds, end_seconds)
    text = t.render(segs)
    if len(text) > max_chars:
        cut = text[:max_chars].rsplit("\n", 1)[0]
        last_line_ts = cut.rsplit("\n", 1)[-1][1:].split("]")[0]
        text = cut + f"\n...[truncated at {last_line_ts}; call again with a later start_seconds]"
    return text or "No speech in that range."


@server.tool()
async def identify_speakers(audio_id: str, num_speakers: int | None = None) -> str:
    """Label who speaks when ('Speaker 1', 'Speaker 2', ...) and save the labels into the transcript,
    so later get_transcript/search/sentiment calls show them. Pass num_speakers if you know it
    (e.g. 2 for an interview): it makes the result much more reliable.
    Uses a local voice-similarity heuristic unless the file was transcribed with AssemblyAI."""
    t = _load(audio_id)
    if t.speaker_method == "assemblyai" and not num_speakers:
        diag = {"method": "assemblyai (neural diarization at transcription time)"}
    else:
        if not Path(t.source_path).is_file():
            raise ToolError(f"The original audio ({t.source_path}) is gone; speaker labelling needs it.")
        samples = await asyncio.to_thread(_decode, t.source_path)
        spans = [(s.start, s.end) for s in t.segments]
        labels, diag = await asyncio.to_thread(speakers.assign_speakers, samples, spans, num_speakers)
        for seg, lab in zip(t.segments, labels):
            seg.speaker = lab
        t.speaker_method = "heuristic"
        store.save(t)
        diag["method"] = "heuristic (MFCC voiceprints + k-means); approximate, best on clean audio with distinct voices"

    # Talk-time per speaker, plus the turn sequence (consecutive segments merged).
    talk: dict[str, float] = {}
    turns = []
    for s in t.segments:
        who = s.speaker or "unknown"
        talk[who] = talk.get(who, 0) + (s.end - s.start)
        if turns and turns[-1]["speaker"] == who:
            turns[-1]["end"] = fmt_ts(s.end)
            turns[-1]["text"] += " " + s.text
        else:
            turns.append({"speaker": who, "start": fmt_ts(s.start), "end": fmt_ts(s.end), "text": s.text})
    total = sum(talk.values()) or 1
    return _json({
        "audio_id": audio_id,
        "speakers": len(talk),
        "talk_time": {k: {"seconds": round(v, 1), "share": f"{100 * v / total:.0f}%"} for k, v in talk.items()},
        "turns": len(turns),
        "first_turns": [{**tr, "text": tr["text"][:200]} for tr in turns[:12]],
        "diagnostics": diag,
    })


@server.tool()
async def summarize_recording(audio_id: str) -> str:
    """Summarise the recording: headline, short summary, timestamped key points and action items.
    Uses Claude when the server has an Anthropic credential, otherwise a local extractive summary
    (the most representative sentences, verbatim, with timestamps)."""
    t = _load(audio_id)
    if settings.claude_available():
        try:
            result = await make_llm().structured(
                system="You summarise transcripts of audio recordings faithfully. Use only what is in the "
                       "transcript; never invent names, numbers or commitments.",
                prompt=f"Recording: {t.source_name} ({fmt_ts(t.duration)})\n\n<transcript>\n{t.render()}\n</transcript>",
                schema=Summary,
            )
            return _json({"method": f"claude ({settings.analysis_model})", **result.model_dump()})
        except Exception as exc:  # no credit, network down, refusal...: degrade, don't fail
            note = f"Claude unavailable ({type(exc).__name__}); used local fallback."
    else:
        note = "No Anthropic credential on the server; used local extractive summary."
    return _json({
        "method": "local extractive (verbatim sentences)",
        "note": note,
        "key_sentences": analysis.extractive_summary(t),
        "keywords": [k["term"] for k in analysis.keywords(t, top=8)],
    })


@server.tool()
async def extract_topics(audio_id: str) -> str:
    """Map the recording's topics (with time ranges), named entities (people, organisations, places,
    products) and keywords. Uses Claude when available, otherwise local keyword/name extraction."""
    t = _load(audio_id)
    if settings.claude_available():
        try:
            result = await make_llm().structured(
                system="You index transcripts of audio recordings. Only list entities that are actually named "
                       "in the transcript. Topic time ranges must come from the transcript's timestamps.",
                prompt=f"<transcript>\n{t.render()}\n</transcript>",
                schema=TopicMap,
            )
            return _json({"method": f"claude ({settings.analysis_model})", **result.model_dump()})
        except Exception as exc:
            note = f"Claude unavailable ({type(exc).__name__}); used local fallback."
    else:
        note = "No Anthropic credential on the server; used local extraction."
    return _json({
        "method": "local (frequency keywords + capitalised-name heuristic)",
        "note": note,
        "keywords": analysis.keywords(t),
        "entities": analysis.entities(t),
    })


@server.tool()
async def sentiment_timeline(audio_id: str, window_seconds: float = 30) -> str:
    """Sentiment over time: the recording cut into windows (default 30 s), each scored from -1
    (negative) to +1 (positive), plus the overall mood, the most negative and most positive moments,
    and per-speaker averages if speakers are labelled. Deterministic (VADER lexicon), no LLM."""
    return _json(analysis.sentiment_timeline(_load(audio_id), window_seconds))


@server.tool()
async def search_transcript(audio_id: str, query: str, max_results: int = 10) -> str:
    """Find where something was said. Matches the exact phrase, then all words in any order, then
    near-spellings (tolerates transcription typos). Each hit has its timestamp, speaker and the
    surrounding lines. Use short queries: a name, a term, or a 2-4 word phrase."""
    return _json(analysis.search(_load(audio_id), query, max_results))


# =============================================================================
# Resources and prompts: the other two MCP primitives.
#   * A resource is data a client can *attach* (Claude Desktop shows them in
#     the "+" menu) rather than a function the model calls.
#   * A prompt is a reusable, parameterised starting message.
# =============================================================================


@server.resource("transcript://{audio_id}", mime_type="text/plain",
                 description="Full timestamped transcript of a recording.")
async def transcript_resource(audio_id: str) -> str:
    t = _load(audio_id)
    return f"# {t.source_name} ({fmt_ts(t.duration)}, {t.language or '?'})\n\n{t.render()}"


@server.prompt(description="Full analysis of an audio file: transcript, speakers, summary, topics, mood.")
def analyze_recording(file_path: str) -> str:
    return (
        f"Analyse the audio file at {file_path}. Transcribe it, label the speakers, then give me: "
        "a summary with timestamped key points, the main topics, how the sentiment shifts over time, "
        "and anything surprising. Cite timestamps."
    )


def main() -> None:
    """Entry point for `uv run soundcheck-mcp` (see [project.scripts])."""
    # stdio transport: the client launches us and talks JSON-RPC over our
    # stdin/stdout. Never print() to stdout in a stdio server; it corrupts the stream.
    server.run()


if __name__ == "__main__":
    main()
