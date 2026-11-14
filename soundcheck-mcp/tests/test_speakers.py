"""The numpy diarizer on synthetic voices with a known answer."""

import numpy as np

from soundcheck.speakers import assign_speakers, silhouette
from tests.conftest import synth_voice


def _conversation(pattern: str, seg_s: float = 2.0):
    """Build audio + spans from a pattern like 'ABAB': one 2 s turn per letter."""
    voices = {"A": (110, 700), "B": (230, 2200), "C": (160, 1300)}
    chunks, spans = [], []
    for i, who in enumerate(pattern):
        f0, formant = voices[who]
        chunks.append(synth_voice(seg_s, f0, formant, seed=i))
        spans.append((i * seg_s, (i + 1) * seg_s))
    return np.concatenate(chunks), spans


def test_two_voices_found_without_being_told():
    audio, spans = _conversation("ABABABAB")
    labels, diag = assign_speakers(audio, spans)
    assert diag["speakers"] == 2 and diag["chosen_by"] == "silhouette"
    assert labels == ["Speaker 1", "Speaker 2"] * 4  # first voice heard = Speaker 1


def test_three_voices_when_count_is_given():
    audio, spans = _conversation("ABCABCABC")
    labels, diag = assign_speakers(audio, spans, num_speakers=3)
    assert len(set(labels)) == 3
    assert labels[:3] == ["Speaker 1", "Speaker 2", "Speaker 3"] and labels[3:6] == labels[:3]


def test_one_voice_stays_one_speaker():
    audio, spans = _conversation("AAAAAAAA")
    labels, diag = assign_speakers(audio, spans)
    assert set(labels) == {"Speaker 1"}, diag


def test_silhouette_sanity():
    x = np.array([[0, 0], [0, 0.1], [5, 5], [5, 5.1]])
    assert silhouette(x, np.array([0, 0, 1, 1])) > 0.9
    assert silhouette(x, np.array([0, 1, 0, 1])) < 0
