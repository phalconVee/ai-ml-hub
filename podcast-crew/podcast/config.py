"""Every knob in one place.

Change a number here and the whole crew follows. No other file hard-codes a
model name, a word budget, a voice or a pause length.
"""

from dataclasses import dataclass


@dataclass(frozen=True)  # frozen: settings are read, never changed at runtime
class Settings:
    # ── The model ────────────────────────────────────────────────────────────
    # Claude Opus 5.5 always thinks before it answers; `effort` is the dial for
    # how hard (low → max). Its default is "medium", so each agent sets its own.
    model: str = "claude-opus-5-5"
    researcher_effort: str = "medium"    # reading + outlining: comprehension, not invention
    writer_effort: str = "high"          # natural dialogue is the hard creative step
    editor_effort: str = "medium"        # tightening an existing script
    max_tokens: int = 16000              # room for thinking + a long JSON script
    # On a safety refusal, the API re-runs the request on a fallback model
    # inside the same call. "default" lets Anthropic choose that model.
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # ── Source (source.py) ───────────────────────────────────────────────────
    fetch_timeout_s: float = 20.0        # seconds to wait for a web page
    max_source_chars: int = 60_000       # ~15k tokens of article text; the rest is cut
    min_source_chars: int = 200          # less than this is not worth an episode

    # ── Script (scriptwriter.py / editor.py) ─────────────────────────────────
    words_per_minute: int = 150          # typical conversational speaking rate
    max_line_words: int = 70             # a longer turn sounds like a lecture → editor splits it
    min_lines: int = 6                   # anything shorter isn't a conversation

    # ── Audio (voices.py / studio.py) ────────────────────────────────────────
    # Every engine is asked for (or converted to) the same format, so stitching
    # is just concatenating bytes: 24 kHz, mono, 16-bit little-endian PCM.
    sample_rate: int = 24_000
    gap_same_speaker_ms: int = 250       # breath between two lines of one host
    gap_turn_ms: int = 450               # slightly longer pause when the speaker changes
    lead_in_ms: int = 300                # silence at the very start and end

    # Kokoro (local, free). The int8 model is ~92 MB, voices ~28 MB.
    kokoro_dir: str = "models"
    kokoro_model_url: str = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                             "model-files-v1.0/kokoro-v1.0.int8.onnx")
    kokoro_voices_url: str = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                              "model-files-v1.0/voices-v1.0.bin")

    # ElevenLabs (optional, paid, needs ELEVENLABS_API_KEY).
    elevenlabs_model: str = "eleven_multilingual_v2"


SETTINGS = Settings()

# Default voice per engine for the two hosts. The first host is "A", the
# second "B". Any of these can be changed in the Streamlit sidebar.
DEFAULT_VOICES = {
    "kokoro": {"A": "af_heart", "B": "am_michael"},       # Kokoro voice ids
    "say": {"A": "Samantha", "B": "Daniel"},               # macOS voices (`say -v '?'`)
    "elevenlabs": {"A": "21m00Tcm4TlvDq8ikWAM",            # "Rachel"
                   "B": "pNInz6obpgDQGcFmaJgB"},           # "Adam"
}
