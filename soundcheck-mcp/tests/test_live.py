"""Live checks: real Whisper, real audio decoding, real stdio subprocess server.

Skipped by default. Needs macOS `say` (to synthesise speech) and downloads the
Whisper model once (~145 MB for "base"). No API keys.

    SOUNDCHECK_LIVE=1 uv run pytest tests/test_live.py -s
"""

import json
import os
import shutil
import subprocess
import wave

import numpy as np
import pytest

from soundcheck.toolbox import open_toolbox

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("SOUNDCHECK_LIVE") != "1", reason="set SOUNDCHECK_LIVE=1"),
    pytest.mark.skipif(shutil.which("say") is None, reason="needs macOS `say`"),
]

TURNS = [
    ("Samantha", "Welcome back to the show. Today we are talking about solar power in Nigeria."),
    ("Daniel", "Thanks for having me. Solar panels have become much cheaper over the last five years."),
    ("Samantha", "That is wonderful news. But what about the batteries? Are they still expensive?"),
    ("Daniel", "Sadly, yes. Batteries are the biggest problem, and many families cannot afford them."),
]


def make_conversation(tmp_path) -> str:
    """Two macOS voices taking turns, joined into one 16 kHz WAV."""
    from faster_whisper import decode_audio

    pieces = []
    for i, (voice, line) in enumerate(TURNS):
        aiff = tmp_path / f"turn{i}.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), line], check=True)
        pieces += [decode_audio(str(aiff), 16000), np.zeros(8000, dtype=np.float32)]  # 0.5 s gap
    audio = np.concatenate(pieces)
    out = tmp_path / "conversation.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((audio * 32767).astype("<i2").tobytes())
    return str(out)


async def test_live_pipeline_over_stdio(tmp_path, monkeypatch):
    monkeypatch.setenv("SOUNDCHECK_DATA_DIR", str(tmp_path / "data"))  # forwarded to the subprocess
    monkeypatch.setenv("SOUNDCHECK_ANALYSIS", "local")
    path = make_conversation(tmp_path)

    async with open_toolbox() as tb:  # spawns `python -m soundcheck.server`, like Claude Desktop would
        text, err = await tb.call("transcribe_audio", {"file_path": path})
        assert not err, text
        info = json.loads(text)
        print("\n", info["preview"])
        aid = info["audio_id"]
        speakers = json.loads((await tb.call("identify_speakers", {"audio_id": aid}))[0])
        print(json.dumps(speakers["talk_time"]), speakers["diagnostics"])
        hits = json.loads((await tb.call("search_transcript", {"audio_id": aid, "query": "batteries"}))[0])
        senti = json.loads((await tb.call("sentiment_timeline", {"audio_id": aid, "window_seconds": 10}))[0])
        transcript, _ = await tb.call("get_transcript", {"audio_id": aid})
        print(transcript)

    assert info["language"] == "en" and info["engine"].startswith("whisper:")
    words = transcript.lower()
    for w in ("solar", "nigeria", "batteries", "cheaper"):
        assert w in words, f"{w!r} not transcribed"
    assert hits["total_hits"] >= 1
    assert senti["points"]
    # Two clearly different synthetic voices: the heuristic should find 2.
    assert speakers["speakers"] == 2, speakers
