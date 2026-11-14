"""Step 7b: voice the script and stitch it into one WAV file.

Free and local by default. Backends, tried in this order by "auto":

    elevenlabs  only if ELEVENLABS_API_KEY is set (paid, optional, nicer voices)
    say         macOS built-in speech synthesiser, no install, no key
    espeak      espeak-ng / espeak on Linux (apt install espeak-ng), no key

Each backend renders one turn to a WAV file in one host's voice. `render()`
then concatenates the turns with a short pause between speakers using only the
standard library's `wave` module. No ffmpeg, no pydub.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import wave
from pathlib import Path
from typing import Optional, Protocol

from .config import SETTINGS

SAMPLE_RATE = 24_000   # all backends are asked for 24 kHz, 16-bit, mono


class Backend(Protocol):
    name: str

    def synth(self, text: str, voice: int, out: Path) -> None:
        """Write `text` spoken by host `voice` (0 or 1) as a WAV file at `out`."""


class SayBackend:
    """macOS `say`. Two distinct built-in voices, one per host."""

    name = "macOS say"
    PREFERRED = (("Samantha", "Ava", "Allison"), ("Daniel", "Alex", "Fred"))

    def __init__(self) -> None:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
        installed = {line.split("  ")[0].strip() for line in listing.splitlines()}
        # Pick the first preferred voice that exists; None = system default.
        self.voices = [next((v for v in group if v in installed), None) for group in self.PREFERRED]

    def synth(self, text: str, voice: int, out: Path) -> None:
        cmd = ["say", "-o", str(out), f"--data-format=LEI16@{SAMPLE_RATE}", "-f", "-"]
        if self.voices[voice]:
            cmd[1:1] = ["-v", self.voices[voice]]
        # Text goes in on stdin, so a line starting with "-" can't be read as a flag.
        subprocess.run(cmd, input=text, text=True, check=True, capture_output=True)


class EspeakBackend:
    """espeak-ng (or classic espeak) on Linux. Robotic, but free and offline."""

    name = "espeak"
    VOICES = ("en-us+f3", "en-gb+m3")

    def __init__(self, exe: str) -> None:
        self.exe = exe

    def synth(self, text: str, voice: int, out: Path) -> None:
        subprocess.run([self.exe, "-v", self.VOICES[voice], "-s", "165", "-w", str(out), "--stdin"],
                       input=text, text=True, check=True, capture_output=True)


class ElevenLabsBackend:
    """ElevenLabs REST API (optional, paid). Asks for raw 24 kHz PCM and wraps it in WAV."""

    name = "ElevenLabs"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.voices = (os.getenv("SOURCEBOOK_ELEVEN_VOICE_A", "21m00Tcm4TlvDq8ikWAM"),
                       os.getenv("SOURCEBOOK_ELEVEN_VOICE_B", "pNInz6obpgDQGcFmaJgB"))

    def synth(self, text: str, voice: int, out: Path) -> None:
        url = (f"https://api.elevenlabs.io/v1/text-to-speech/{self.voices[voice]}"
               f"?output_format=pcm_{SAMPLE_RATE}")
        req = urllib.request.Request(
            url, method="POST",
            data=json.dumps({"text": text, "model_id": "eleven_multilingual_v2"}).encode(),
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            pcm = resp.read()
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)


def pick_backend(choice: str = SETTINGS.tts) -> Optional[Backend]:
    """Resolve the SOURCEBOOK_TTS setting to a backend, or None (script only)."""
    key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if choice in ("auto", "elevenlabs") and key:
        return ElevenLabsBackend(key)
    if choice in ("auto", "say") and shutil.which("say"):
        return SayBackend()
    if choice in ("auto", "espeak") and espeak:
        return EspeakBackend(espeak)
    return None


def render(turns: list[tuple[str, str]], backend: Backend, out: Path,
           pause_s: float = 0.35, on_progress=lambda done, total: None) -> float:
    """Voice every (speaker, text) turn and join them. Returns duration in seconds."""
    out.parent.mkdir(parents=True, exist_ok=True)
    params = None
    frames = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        for i, (speaker, text) in enumerate(turns):
            part = Path(tmp) / f"{i:03d}.wav"
            backend.synth(text, 0 if speaker == "A" else 1, part)
            with wave.open(str(part), "rb") as w:
                p = (w.getnchannels(), w.getsampwidth(), w.getframerate())
                if params is None:
                    params = p
                elif p != params:
                    raise RuntimeError(f"{backend.name} returned mixed audio formats: {p} vs {params}")
                if i:   # silence between turns: zero bytes = digital silence for PCM
                    frames.write(b"\x00" * int(params[2] * pause_s) * params[0] * params[1])
                frames.write(w.readframes(w.getnframes()))
            on_progress(i + 1, len(turns))
    if params is None:
        raise ValueError("The script has no turns to voice.")
    with wave.open(str(out), "wb") as w:
        w.setnchannels(params[0])
        w.setsampwidth(params[1])
        w.setframerate(params[2])
        w.writeframes(frames.getvalue())
    channels, width, rate = params
    return len(frames.getvalue()) / (channels * width * rate)
