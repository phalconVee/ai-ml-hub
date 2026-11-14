"""Speech to text: audio file in, list of timestamped Segments out.

Two engines behind one function signature:

    LocalWhisper   faster-whisper (Whisper re-implemented on CTranslate2).
                   Free, offline after a one-time model download, the default.
    AssemblyAI     paid cloud API, only used when you ask for it AND have
                   ASSEMBLYAI_API_KEY. Its edge: real neural speaker labels.

Both are plain *blocking* functions. The MCP server runs them in a worker
thread (`asyncio.to_thread`) so a 20-minute transcription doesn't freeze the
server's event loop (it can still answer pings and report progress).
"""

import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

from soundcheck.config import settings
from soundcheck.transcript import Segment

# Called with a number in [0, 1] as transcription advances.
ProgressFn = Callable[[float], None]


class TranscriptionResult:
    """What every engine hands back. Plain class: nothing to validate here."""

    def __init__(self, segments: list[Segment], duration: float, language: str | None, engine: str,
                 speaker_method: str | None = None):
        self.segments = segments
        self.duration = duration
        self.language = language
        self.engine = engine
        self.speaker_method = speaker_method


# --- Engine 1: local Whisper -------------------------------------------------


@lru_cache(maxsize=2)
def _whisper(model_size: str, device: str, compute_type: str):
    """Load a Whisper model once per process and keep it.

    Loading takes a second or two (and the very first time, downloads the
    weights from Hugging Face into ~/.cache/huggingface). lru_cache turns
    "load on every call" into "load on first call".
    Imported lazily so the server starts instantly and tests that never
    transcribe never pay for importing CTranslate2.
    """
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_local(path: Path, language: str | None = None, on_progress: ProgressFn | None = None) -> TranscriptionResult:
    model = _whisper(settings.whisper_model, settings.whisper_device, settings.whisper_compute_type)
    # `segments` is a *generator*: decoding actually happens as we iterate.
    # That is what lets us report progress segment by segment.
    segments_iter, info = model.transcribe(
        str(path),
        language=language,      # None = let Whisper detect it from the first 30 s
        vad_filter=True,        # Silero VAD skips silence: faster, and fewer "hallucinated" lines in quiet parts
        beam_size=5,
        condition_on_previous_text=False,  # stops one misheard phrase from snowballing through the file
    )
    out: list[Segment] = []
    for s in segments_iter:
        text = s.text.strip()
        if text:
            out.append(Segment(start=round(s.start, 2), end=round(s.end, 2), text=text))
        if on_progress and info.duration:
            on_progress(min(1.0, s.end / info.duration))
    return TranscriptionResult(
        segments=out,
        duration=round(info.duration, 2),
        language=info.language,
        engine=f"whisper:{settings.whisper_model}",
    )


# --- Engine 2: AssemblyAI (optional, paid) -----------------------------------
# Plain REST over httpx, no SDK: three calls (upload, create job, poll).
# Not exercised by the offline tests beyond "refuses without a key"; see README.

ASSEMBLYAI_BASE = "https://api.assemblyai.com/v2"


def transcribe_assemblyai(path: Path, language: str | None = None, on_progress: ProgressFn | None = None) -> TranscriptionResult:
    import httpx

    if not settings.assemblyai_key:
        raise RuntimeError("engine='assemblyai' needs ASSEMBLYAI_API_KEY. Use engine='local' for the free path.")
    headers = {"authorization": settings.assemblyai_key}
    with httpx.Client(base_url=ASSEMBLYAI_BASE, headers=headers, timeout=120) as http:
        # 1. Upload the raw bytes; we get back a private URL only AssemblyAI can read.
        with open(path, "rb") as f:
            upload_url = http.post("/upload", content=f).raise_for_status().json()["upload_url"]
        # 2. Start a job. speaker_labels=True is the whole reason to pay for this engine.
        job = {"audio_url": upload_url, "speaker_labels": True}
        if language:
            job["language_code"] = language
        else:
            job["language_detection"] = True
        job_id = http.post("/transcript", json=job).raise_for_status().json()["id"]
        # 3. Poll until done. Jobs usually finish in ~15-30% of the audio length.
        while True:
            data = http.get(f"/transcript/{job_id}").raise_for_status().json()
            if data["status"] == "completed":
                break
            if data["status"] == "error":
                raise RuntimeError(f"AssemblyAI error: {data.get('error')}")
            if on_progress:
                on_progress(0.5)  # the API gives no percentage; "working" is all we know
            time.sleep(3)
    # AssemblyAI reports milliseconds and letters ("A", "B"); normalise both.
    segments = [
        Segment(start=u["start"] / 1000, end=u["end"] / 1000, text=u["text"].strip(),
                speaker=f"Speaker {ord(u['speaker'][0]) - ord('A') + 1}")
        for u in data.get("utterances") or []
    ]
    return TranscriptionResult(
        segments=segments,
        duration=float(data.get("audio_duration") or (segments[-1].end if segments else 0.0)),
        language=data.get("language_code"),
        engine="assemblyai",
        speaker_method="assemblyai",
    )


# --- Dispatcher ----------------------------------------------------------------

ENGINES: dict[str, Callable[..., TranscriptionResult]] = {
    "local": transcribe_local,
    "assemblyai": transcribe_assemblyai,
}


def resolve_engine(requested: str | None) -> str:
    """Map "auto"/None to a concrete engine name.

    "auto" always means local. A paid API is used only when someone names it,
    either per call (engine="assemblyai") or via SOUNDCHECK_ENGINE, so a key
    sitting in .env never starts costing money by surprise.
    """
    name = (requested or settings.engine or "auto").lower()
    if name == "auto":
        name = settings.engine.lower() if settings.engine.lower() != "auto" else "local"
    if name not in ENGINES:
        raise ValueError(f"Unknown engine {name!r}. Choose one of: auto, {', '.join(ENGINES)}")
    return name


def transcribe(path: Path, engine: str | None = None, language: str | None = None,
               on_progress: ProgressFn | None = None) -> TranscriptionResult:
    return ENGINES[resolve_engine(engine)](path, language=language, on_progress=on_progress)
