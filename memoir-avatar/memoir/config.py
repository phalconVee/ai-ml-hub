"""Every knob in one place.

Change a number here and the whole app follows: no other file hard-codes a
model name, a retrieval budget, a voice or an audio rate.
"""

from dataclasses import dataclass


@dataclass(frozen=True)  # frozen: settings are read, never changed at runtime
class Settings:
    # ── The model ────────────────────────────────────────────────────────────
    # Claude Opus 5.5 thinks before it answers; `effort` is how hard (low → max).
    # Two different jobs, two different dials:
    model: str = "claude-opus-5-5"
    reply_effort: str = "medium"       # a warm, accurate conversational answer
    extract_effort: str = "low"        # reading facts out of one exchange: comprehension, not invention
    max_tokens: int = 8000             # room for thinking + a short reply / a JSON fact list
    # On a safety refusal the API can re-run the request on a fallback model
    # inside the same call. "default" lets Anthropic choose that model.
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # ── Memory (graph.py / retrieve.py) ──────────────────────────────────────
    db_path: str = ".memoir/memory.db"  # one SQLite file = the whole long-term memory
    user_node: str = "User"            # the person talking; every personal fact hangs off it
    hops: int = 2                      # how far out from a matched entity we walk
    max_facts: int = 25                # facts handed to Claude per turn (a context budget)
    known_entities_hint: int = 60      # names shown to the extractor so it reuses them
    history_turns: int = 6             # recent chat turns sent along with the facts

    # ── Voice (voice.py) ─────────────────────────────────────────────────────
    # Every server-side engine returns the same raw format: 24 kHz, mono,
    # 16-bit little-endian PCM. The avatar's lip-sync envelope is computed
    # from those samples, so one format means one envelope function.
    sample_rate: int = 24_000
    envelope_fps: int = 30             # mouth positions per second (one per animation frame at 30 fps)
    audio_dir: str = ".memoir/audio"   # generated replies (git-ignored)

    # Kokoro (local, free). The int8 model is ~92 MB, voices ~28 MB.
    kokoro_dir: str = "models"
    kokoro_model_url: str = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                             "model-files-v1.0/kokoro-v1.0.int8.onnx")
    kokoro_voices_url: str = ("https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
                              "model-files-v1.0/voices-v1.0.bin")

    # ElevenLabs (optional, paid, needs ELEVENLABS_API_KEY; never auto-picked).
    elevenlabs_model: str = "eleven_multilingual_v2"


SETTINGS = Settings()

# Default voice per engine. "browser" means the viewer's own browser speaks
# (Web Speech API), so its voice is chosen in JavaScript, not here.
DEFAULT_VOICES = {
    "kokoro": "af_heart",
    "say": "Samantha",
    "elevenlabs": "21m00Tcm4TlvDq8ikWAM",   # "Rachel"
    "browser": "",
}
