"""Speech out, plus the numbers that make the avatar's mouth move.

Four engines. Three run on the server and return raw audio; one runs in the
viewer's browser:

  engine      where    cost   needs                                    quality
  ─────────   ──────   ────   ──────────────────────────────────────   ──────────
  kokoro      server   free   `uv sync --extra kokoro` + ~120 MB model  natural, neural
  say         server   free   macOS (built in)                         robotic, instant
  browser     browser  free   nothing (Web Speech API)                 depends on the OS
  elevenlabs  server   paid   ELEVENLABS_API_KEY, opt-in only           studio-grade

Every server engine returns the same format: 24 kHz mono 16-bit PCM. From
those samples `envelope()` computes the lip-sync track: one loudness value
per animation frame. The avatar plays the audio and, each frame, opens its
mouth by the envelope value at audio.currentTime. Loud = open, silent =
closed. It isn't phoneme-accurate, but it is truly synced to the sound.

The browser engine has no samples (the OS speaks), so the avatar fakes it
from the speech engine's word-boundary events instead (see avatar.py).
"""

from __future__ import annotations

import array
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

from .config import SETTINGS


class VoiceError(RuntimeError):
    """An engine is unavailable or failed."""


@dataclass
class Speech:
    """One spoken reply, ready for the avatar."""
    engine: str
    wav: bytes = b""                                   # empty for the browser engine
    envelope: list[float] = field(default_factory=list)
    duration_s: float = 0.0


# ── The lip-sync envelope ───────────────────────────────────────────────────

def envelope(pcm: bytes, *, rate: int = SETTINGS.sample_rate,
             fps: int = SETTINGS.envelope_fps) -> list[float]:
    """PCM bytes → one mouth-openness value (0..1) per animation frame.

    For each 1/fps-second window: RMS loudness (root of the mean squared
    sample), then normalise by a loud-but-not-max reference so normal speech
    opens the mouth well, and smooth a little so the mouth doesn't flicker.
    Pure stdlib (`array`), so it works with no numpy.
    """
    samples = array.array("h")                         # signed 16-bit
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if sys.byteorder == "big":
        samples.byteswap()                              # our PCM is little-endian
    if not samples:
        return []
    step = max(1, rate // fps)
    rms = []
    for i in range(0, len(samples), step):
        win = samples[i:i + step]
        rms.append(math.sqrt(sum(s * s for s in win) / len(win)))
    # Reference = the 95th-percentile frame: one shout won't make everything else look closed.
    ref = sorted(rms)[int(0.95 * (len(rms) - 1))] or 1.0
    raw = [min(1.0, r / ref) for r in rms]
    # Gate (quiet hiss = closed), then a 3-frame moving average.
    gated = [0.0 if v < 0.08 else v for v in raw]
    smooth = [sum(gated[max(0, i - 1):i + 2]) / len(gated[max(0, i - 1):i + 2])
              for i in range(len(gated))]
    return [round(v, 3) for v in smooth]


def pcm_to_wav(pcm: bytes, rate: int = SETTINGS.sample_rate) -> bytes:
    """Wrap raw PCM in a WAV header (stdlib `wave`), playable by any browser."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def read_wav_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if fmt != (1, 2, SETTINGS.sample_rate):
            raise VoiceError(f"Unexpected WAV format {fmt}")
        return w.readframes(w.getnframes())


# ── Engine: macOS `say` (free, built in) ────────────────────────────────────

class SayEngine:
    name = "say"

    def __init__(self):
        if platform.system() != "Darwin" or not shutil.which("say"):
            raise VoiceError("macOS `say` is only available on a Mac.")

    def synthesize(self, text: str, voice: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reply.wav"
            # LEI16@24000 = little-endian 16-bit integers at 24 kHz: our exact format.
            # "--" ends the options, so text starting with "-" is still spoken.
            cmd = ["say", "-v", voice, "-o", str(out), "--file-format=WAVE",
                   f"--data-format=LEI16@{SETTINGS.sample_rate}", "--", text]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0 or not out.exists():
                raise VoiceError(f"say failed: {done.stderr.strip() or 'no output'}")
            return read_wav_pcm(out)


# ── Engine: Kokoro (free, local neural TTS, optional install) ───────────────

def kokoro_paths(model_dir: str | Path = SETTINGS.kokoro_dir) -> tuple[Path, Path]:
    d = Path(model_dir)
    return d / Path(SETTINGS.kokoro_model_url).name, d / Path(SETTINGS.kokoro_voices_url).name


def kokoro_ready() -> bool:
    try:
        import kokoro_onnx  # noqa: F401
    except ImportError:
        return False
    return all(p.exists() for p in kokoro_paths())


class KokoroEngine:
    name = "kokoro"

    def __init__(self):
        from kokoro_onnx import Kokoro  # optional dependency
        model, voices = kokoro_paths()
        if not (model.exists() and voices.exists()):
            raise VoiceError("Kokoro model missing. Run: uv run python -m memoir.voice --download-kokoro")
        self._tts = Kokoro(str(model), str(voices))

    def synthesize(self, text: str, voice: str) -> bytes:
        import numpy as np
        samples, rate = self._tts.create(text, voice=voice, speed=1.0, lang="en-us")
        if rate != SETTINGS.sample_rate:
            raise VoiceError(f"Kokoro returned {rate} Hz; expected {SETTINGS.sample_rate}.")
        return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def download_kokoro() -> None:
    """One-time download of the open Kokoro model files from their GitHub release."""
    for url, path in zip((SETTINGS.kokoro_model_url, SETTINGS.kokoro_voices_url), kokoro_paths()):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"downloading {url} → {path}")
            urllib.request.urlretrieve(url, path.with_suffix(".part"))
            path.with_suffix(".part").rename(path)


# ── Engine: ElevenLabs (paid, optional, never auto-picked) ──────────────────

class ElevenLabsEngine:
    name = "elevenlabs"
    URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_{rate}"

    def __init__(self, api_key: str | None = None, *, opener=urllib.request.urlopen):
        self.key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        if not self.key:
            raise VoiceError("ElevenLabs needs ELEVENLABS_API_KEY in .env.")
        self._open = opener                            # injectable: tests never hit the network

    def synthesize(self, text: str, voice: str) -> bytes:
        req = urllib.request.Request(
            self.URL.format(voice=voice, rate=SETTINGS.sample_rate),
            data=json.dumps({"text": text, "model_id": SETTINGS.elevenlabs_model}).encode(),
            headers={"xi-api-key": self.key, "Content-Type": "application/json"},
            method="POST")
        try:
            with self._open(req, timeout=60) as resp:
                return resp.read()                     # pcm_24000 = our raw format already
        except Exception as e:
            raise VoiceError(f"ElevenLabs failed: {e}") from e


# ── Choosing and running an engine ──────────────────────────────────────────

SERVER_ENGINES = {"kokoro": KokoroEngine, "say": SayEngine, "elevenlabs": ElevenLabsEngine}


def available_engines() -> list[str]:
    """What works right now, best free option first. 'browser' always works."""
    found = []
    if kokoro_ready():
        found.append("kokoro")
    if platform.system() == "Darwin" and shutil.which("say"):
        found.append("say")
    found.append("browser")
    if os.getenv("ELEVENLABS_API_KEY"):
        found.append("elevenlabs")
    return found


def speak(text: str, engine: str, voice: str = "", *, engine_obj=None) -> Speech:
    """Text → Speech. For 'browser' nothing is synthesised here: the avatar speaks."""
    if engine == "browser":
        return Speech(engine="browser")
    tts = engine_obj or SERVER_ENGINES[engine]()
    pcm = tts.synthesize(text, voice)
    return Speech(engine=engine, wav=pcm_to_wav(pcm), envelope=envelope(pcm),
                  duration_s=len(pcm) / 2 / SETTINGS.sample_rate)


if __name__ == "__main__":
    # uv run python -m memoir.voice --download-kokoro
    if "--download-kokoro" in sys.argv:
        download_kokoro()
    print("available engines:", available_engines())
