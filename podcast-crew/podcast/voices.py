"""The voice step: text in, raw audio out. Three interchangeable engines.

Every engine implements one method,

    synthesize(text, voice) -> bytes     # 24 kHz, mono, 16-bit little-endian PCM

so studio.py can stitch lines from any engine by simply joining bytes. Each
engine is asked for exactly that format (or converts to it), which is why
there's no audio library (pydub, ffmpeg) in this project.

  engine       cost   needs                                   quality
  ─────────    ────   ─────────────────────────────────────   ──────────
  kokoro       free   `uv sync --extra kokoro` + ~120 MB model  natural, neural
  say          free   macOS (built in)                        robotic but instant
  elevenlabs   paid   ELEVENLABS_API_KEY                      studio-grade

`pick_engine("auto")` prefers the free local ones: Kokoro if it's installed
and its model files are on disk, otherwise macOS `say`. ElevenLabs is only
ever used when you ask for it by name *and* the key is set.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path

from .config import SETTINGS


class VoiceError(RuntimeError):
    """An engine is unavailable or failed on a line."""


# ── Engine 1: Kokoro (local neural TTS, free) ───────────────────────────────

class KokoroEngine:
    """Kokoro-82M via ONNX Runtime: a small open model that sounds human.

    Runs fully offline on the CPU once the two model files are downloaded
    (see download_kokoro()). Returns float samples, which we convert to PCM.
    """

    name = "kokoro"

    def __init__(self, model_dir: str | Path = SETTINGS.kokoro_dir):
        from kokoro_onnx import Kokoro  # optional dependency, imported only here

        model, voices = kokoro_paths(model_dir)
        if not (model.exists() and voices.exists()):
            raise VoiceError("Kokoro model files are missing. Run: "
                             "uv run python -m podcast.voices --download-kokoro")
        self._tts = Kokoro(str(model), str(voices))

    def voices(self) -> list[str]:
        return sorted(self._tts.get_voices())

    def synthesize(self, text: str, voice: str) -> bytes:
        samples, rate = self._tts.create(text, voice=voice, speed=1.0, lang="en-us")
        if rate != SETTINGS.sample_rate:
            raise VoiceError(f"Kokoro returned {rate} Hz; expected {SETTINGS.sample_rate}.")
        return float_to_pcm16(samples)


def kokoro_paths(model_dir: str | Path) -> tuple[Path, Path]:
    d = Path(model_dir)
    return (d / Path(SETTINGS.kokoro_model_url).name, d / Path(SETTINGS.kokoro_voices_url).name)


def kokoro_ready(model_dir: str | Path = SETTINGS.kokoro_dir) -> bool:
    """Installed and downloaded? (Checked without loading the model.)"""
    try:
        import kokoro_onnx  # noqa: F401
    except ImportError:
        return False
    return all(p.exists() for p in kokoro_paths(model_dir))


def download_kokoro(model_dir: str | Path = SETTINGS.kokoro_dir) -> None:
    """One-time download of the open Kokoro model files from their GitHub release."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    for url, path in zip((SETTINGS.kokoro_model_url, SETTINGS.kokoro_voices_url),
                         kokoro_paths(model_dir)):
        if path.exists():
            print(f"already have {path}")
            continue
        print(f"downloading {url} → {path}")
        tmp = path.with_suffix(path.suffix + ".part")    # no half-files if interrupted
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)


def float_to_pcm16(samples) -> bytes:
    """[-1.0, 1.0] floats (a numpy array) → 16-bit signed little-endian bytes."""
    import numpy as np  # installed alongside kokoro-onnx

    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767).astype("<i2").tobytes()


# ── Engine 2: macOS `say` (built in, free) ──────────────────────────────────

class SayEngine:
    """The speech synthesiser every Mac already has. Zero install.

    `say` can write a WAV file directly in our exact format:
    --data-format=LEI16@24000 means Little-Endian Integer, 16-bit, 24 kHz.
    """

    name = "say"

    def __init__(self):
        if platform.system() != "Darwin" or not shutil.which("say"):
            raise VoiceError("macOS `say` is only available on a Mac.")

    def voices(self) -> list[str]:
        # `say -v '?'` prints "Name    lang_REGION  # sample sentence" per voice.
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        return [ln.split("  ")[0].strip() for ln in out.splitlines()
                if "en_" in ln or "en-" in ln]

    def synthesize(self, text: str, voice: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "line.wav"
            # "--" ends the options, so a line starting with "-" is still spoken text.
            cmd = ["say", "-v", voice, "-o", str(out), "--file-format=WAVE",
                   f"--data-format=LEI16@{SETTINGS.sample_rate}", "--", text]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0 or not out.exists():
                raise VoiceError(f"say failed: {done.stderr.strip() or 'no output'}")
            return read_wav_pcm(out)


def read_wav_pcm(path: Path) -> bytes:
    """Frames from a WAV file, after checking it's in the format we expect."""
    with wave.open(str(path), "rb") as w:
        fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if fmt != (1, 2, SETTINGS.sample_rate):
            raise VoiceError(f"Unexpected WAV format {fmt}")
        return w.readframes(w.getnframes())


# ── Engine 3: ElevenLabs (cloud, paid, optional) ────────────────────────────

class ElevenLabsEngine:
    """Optional paid upgrade. Plain HTTPS, no SDK: one POST per line.

    output_format=pcm_24000 asks for raw 16-bit PCM at 24 kHz, our exact
    format, so the response body can be used as-is.
    """

    name = "elevenlabs"
    URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_{rate}"

    def __init__(self, api_key: str | None = None, *, opener=urllib.request.urlopen):
        self.key = api_key or os.getenv("ELEVENLABS_API_KEY", "")
        if not self.key:
            raise VoiceError("ElevenLabs needs ELEVENLABS_API_KEY in .env.")
        self._open = opener  # injectable, so tests never hit the network

    def voices(self) -> list[str]:
        return []  # voice ids are typed in the sidebar; listing needs another API call

    def synthesize(self, text: str, voice: str) -> bytes:
        req = urllib.request.Request(
            self.URL.format(voice=voice, rate=SETTINGS.sample_rate),
            data=json.dumps({"text": text, "model_id": SETTINGS.elevenlabs_model}).encode(),
            headers={"xi-api-key": self.key, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._open(req, timeout=60) as resp:
                return resp.read()
        except Exception as e:
            raise VoiceError(f"ElevenLabs failed: {e}") from e


# ── Choosing an engine ──────────────────────────────────────────────────────

ENGINES = {"kokoro": KokoroEngine, "say": SayEngine, "elevenlabs": ElevenLabsEngine}


def available_engines() -> list[str]:
    """Engines that would work right now, best free one first."""
    found = []
    if kokoro_ready():
        found.append("kokoro")
    if platform.system() == "Darwin" and shutil.which("say"):
        found.append("say")
    if os.getenv("ELEVENLABS_API_KEY"):
        found.append("elevenlabs")
    return found


def pick_engine(choice: str = "auto"):
    """'auto' → the first free local engine available; otherwise the one named."""
    if choice == "auto":
        free = [e for e in available_engines() if e != "elevenlabs"]
        if not free:
            raise VoiceError("No free local voice found. Install Kokoro "
                             "(uv sync --extra kokoro, then download the model) "
                             "or pick ElevenLabs with a key.")
        choice = free[0]
    if choice not in ENGINES:
        raise VoiceError(f"Unknown voice engine: {choice}")
    return ENGINES[choice]()


if __name__ == "__main__":
    # uv run python -m podcast.voices --download-kokoro
    import sys

    if "--download-kokoro" in sys.argv:
        download_kokoro()
    print("available engines:", available_engines() or "none")
