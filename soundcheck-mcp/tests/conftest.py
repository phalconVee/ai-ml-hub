"""Shared fixtures: an isolated data folder, a sample transcript, synthetic voices."""

import numpy as np
import pytest

from soundcheck.config import settings
from soundcheck.transcript import Segment, Transcript


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets its own empty data dir and never calls Claude from the server."""
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "analysis_mode", "local")
    yield


LINES = [
    (0, 4, "Welcome to the show. Today we talk about Kubernetes at Acme Corp."),
    (4, 9, "Thanks for having me. I love working with Kubernetes, it is wonderful and fun."),
    (9, 15, "Last year the migration to Kubernetes was a disaster and the outage was terrible."),
    (15, 21, "We lost customers and everyone was angry and exhausted after the outage."),
    (21, 27, "Then Maria Lopez joined the platform team and rebuilt the deployment pipeline."),
    (27, 33, "The deployment pipeline is fantastic now, releases are fast and reliable."),
    (33, 39, "So what is next for Acme Corp and the platform team?"),
    (39, 45, "We will move the billing service to Kubernetes before March. That is the plan."),
]


@pytest.fixture
def sample_transcript() -> Transcript:
    return Transcript(
        audio_id="abc123def456", source_name="show.wav", source_path="/nowhere/show.wav",
        duration=45.0, language="en", engine="fake",
        segments=[Segment(start=a, end=b, text=t) for a, b, t in LINES],
    )


def synth_voice(seconds: float, f0: float, formant: float, seed: int, sr: int = 16000) -> np.ndarray:
    """A crude 'voice': a harmonic buzz at pitch f0, shaped by one resonance
    (formant) and given a little vibrato and noise. Two of these with different
    f0/formant sound (and look, spectrally) like two different people."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    pitch = f0 * (1 + 0.03 * np.sin(2 * np.pi * 5 * t))  # vibrato
    phase = 2 * np.pi * np.cumsum(pitch) / sr
    sig = np.zeros_like(t)
    for h in range(1, 30):
        if h * f0 > sr / 2:
            break
        gain = np.exp(-((h * f0 - formant) ** 2) / (2 * 400.0 ** 2)) + 0.05
        sig += gain * np.sin(h * phase)
    sig += 0.02 * rng.standard_normal(len(t))
    return (sig / np.abs(sig).max() * 0.5).astype(np.float32)
