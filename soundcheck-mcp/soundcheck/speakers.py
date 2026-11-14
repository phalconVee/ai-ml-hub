"""Who spoke when, without a neural diarization model.

Proper diarization (pyannote, NeMo, AssemblyAI) uses networks trained to embed
voices. Those mean a gated Hugging Face model + PyTorch (gigabytes) or a paid
API. This module is the keyless, dependency-free baseline, in numpy only:

    1. cut the audio at Whisper's segment boundaries
    2. describe each segment's *timbre* with MFCC statistics (a "voiceprint")
    3. cluster the voiceprints with k-means, trying k = 1..max
    4. pick the k with the best silhouette score; if no k separates the
       segments clearly, say "one speaker" rather than invent a second

It works well on clean recordings with distinct voices (an interview, a
two-host podcast) and poorly on crosstalk, phone audio, or two similar voices.
The server labels its output as heuristic so the agent can say so. For
production-grade labels, transcribe with engine="assemblyai".
"""

import numpy as np

SAMPLE_RATE = 16000


# --- 1+2. A voiceprint per segment -------------------------------------------


def _mel_filterbank(n_filters: int, n_fft: int, sr: int) -> np.ndarray:
    """Triangular filters spaced evenly on the mel scale (how pitch is *heard*).

    Result shape: (n_filters, n_fft//2 + 1). Multiplying a power spectrum by it
    gives "how much energy in each perceptual band".
    """
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    mels = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_filters + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mels) / sr).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left, centre, right = bins[i - 1], bins[i], bins[i + 1]
        for k in range(left, centre):
            fb[i - 1, k] = (k - left) / max(centre - left, 1)   # rising edge
        for k in range(centre, right):
            fb[i - 1, k] = (right - k) / max(right - centre, 1)  # falling edge
    return fb


_FB = _mel_filterbank(40, 512, SAMPLE_RATE)
# DCT-II matrix: turns 40 correlated log-band energies into decorrelated
# cepstral coefficients. Row 0 (overall loudness) is dropped below, so a
# speaker who leans into the mic doesn't become "a different speaker".
_DCT = np.cos(np.pi / 40 * (np.arange(40)[None, :] + 0.5) * np.arange(21)[:, None])


def mfcc(samples: np.ndarray) -> np.ndarray:
    """Classic MFCCs: 25 ms frames every 10 ms -> 20 coefficients per frame."""
    frame, hop = 400, 160  # 25 ms and 10 ms at 16 kHz
    if len(samples) < frame:
        samples = np.pad(samples, (0, frame - len(samples)))
    n = 1 + (len(samples) - frame) // hop
    # Build all frames at once with one fancy-indexing step: no Python loop over frames.
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = samples[idx] * np.hamming(frame)
    power = np.abs(np.fft.rfft(frames, n=512)) ** 2
    log_mel = np.log(power @ _FB.T + 1e-10)
    return (log_mel @ _DCT.T)[:, 1:]  # (n_frames, 20), c0 dropped


def voiceprint(samples: np.ndarray) -> np.ndarray:
    """Mean + std of each MFCC over the segment = a 40-number summary of timbre.

    Mean captures the voice's average spectral shape (vocal tract, mic);
    std captures how much it moves (pitch range, speaking style).
    """
    m = mfcc(samples)
    return np.concatenate([m.mean(axis=0), m.std(axis=0)])


# --- 3. k-means and silhouette, from scratch ----------------------------------


def kmeans(x: np.ndarray, k: int, seed: int = 0, restarts: int = 8, iters: int = 50) -> np.ndarray:
    """Standard k-means with k-means++ starts. Returns a label per row.

    Restarts with different seeds and keeps the tightest result, because a
    single k-means run can get stuck with a bad start.
    """
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    for _ in range(restarts):
        # k-means++: first centre at random, each next one far from existing ones.
        centres = [x[rng.integers(len(x))]]
        for _ in range(1, k):
            d2 = np.min([((x - c) ** 2).sum(1) for c in centres], axis=0)
            probs = d2 / d2.sum() if d2.sum() > 0 else None
            centres.append(x[rng.choice(len(x), p=probs)])
        centres = np.array(centres)
        for _ in range(iters):
            labels = np.argmin(((x[:, None, :] - centres[None]) ** 2).sum(-1), axis=1)
            new = np.array([x[labels == j].mean(0) if np.any(labels == j) else centres[j] for j in range(k)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = ((x - centres[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia
    return best_labels


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette: for each point, (b - a) / max(a, b) where
    a = mean distance to its own cluster, b = mean distance to the nearest other.
    +1 = crisp, well-separated clusters; 0 = overlapping; <0 = wrong.
    """
    ks = np.unique(labels)
    if len(ks) < 2:
        return 0.0
    # Pairwise distances via |a-b|^2 = |a|^2 + |b|^2 - 2ab: an (n, n) matrix
    # without materialising the (n, n, dims) difference cube (an hour of audio
    # is ~1000 segments; the cube would be hundreds of MB).
    sq = (x ** 2).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0))
    scores = []
    for i in range(len(x)):
        own = labels == labels[i]
        if own.sum() <= 1:
            scores.append(0.0)
            continue
        a = d[i, own].sum() / (own.sum() - 1)  # exclude the zero distance to itself
        b = min(d[i, labels == k].mean() for k in ks if k != labels[i])
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores))


# --- 4. Put it together ----------------------------------------------------------


def assign_speakers(
    samples: np.ndarray,
    spans: list[tuple[float, float]],
    num_speakers: int | None = None,
    max_speakers: int = 6,
    min_silhouette: float = 0.15,
    margin: float = 0.05,
) -> tuple[list[str], dict]:
    """Label each (start, end) span "Speaker N". Returns (labels, diagnostics).

    num_speakers: if the user knows it ("it's a two-person interview"), we skip
    model selection entirely, which is the single biggest accuracy win.
    """
    if not spans:
        return [], {"speakers": 0}
    feats = []
    for start, end in spans:
        a, b = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
        chunk = samples[a:b]
        feats.append(voiceprint(chunk if len(chunk) else np.zeros(400)))
    x = np.array(feats)
    # Standardise each dimension across the recording (z-score), so no single
    # coefficient dominates the distance just because its numbers are bigger.
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)

    n = len(spans)
    diag: dict = {"segments": n}
    if num_speakers:
        k = max(1, min(num_speakers, n))
        labels = kmeans(x, k) if k > 1 else np.zeros(n, dtype=int)
        diag["chosen_by"] = "caller"
    elif n < 4:
        labels, k = np.zeros(n, dtype=int), 1  # too few segments to judge anything
        diag["chosen_by"] = "too few segments"
    else:
        scores = {}
        candidates = {}
        # At most one speaker per two segments: with few segments, k-means can
        # always "explain" a stray short line as its own speaker, and the
        # silhouette score barely penalises it.
        for k_try in range(2, max(2, min(max_speakers, n // 2)) + 1):
            lab = kmeans(x, k_try)
            scores[k_try] = silhouette(x, lab)
            candidates[k_try] = lab
        # Parsimony: walk up from k=2 and only move to a larger k when it is
        # *clearly* better (by `margin`), not when it wins a near-tie.
        # (Seen live: 5 segments, two voices, silhouette 0.278 for k=2 vs
        # 0.287 for k=3; without this rule a one-line "Speaker 3" appeared.)
        best_k = 2
        for k_try in sorted(scores)[1:]:
            if scores[k_try] > scores[best_k] + margin:
                best_k = k_try
        diag["silhouette"] = {k_: round(v, 3) for k_, v in scores.items()}
        if scores[best_k] >= min_silhouette:
            labels, k = candidates[best_k], best_k
            diag["chosen_by"] = "silhouette"
        else:
            # Nothing separates cleanly: one voice is the honest answer.
            labels, k = np.zeros(n, dtype=int), 1
            diag["chosen_by"] = "no clear separation"

    # Rename clusters in order of first appearance: whoever speaks first is
    # "Speaker 1". Cluster ids from k-means are arbitrary otherwise.
    order: dict[int, int] = {}
    for lab in labels:
        order.setdefault(int(lab), len(order) + 1)
    diag["speakers"] = len(order)
    return [f"Speaker {order[int(lab)]}" for lab in labels], diag
