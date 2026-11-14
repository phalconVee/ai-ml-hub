"""Every knob in one place: the Jev endpoint, image limits, default thresholds.

Nothing here talks to the network. Reading this file first tells you the
whole "shape" of the app: what it sends, how big the pictures are, and how
cautious the decision layer is out of the box.
"""

from pathlib import Path

# ── The Jev API ─────────────────────────────────────────────────────────────
# One endpoint serves every System One question type (choice, score, noul, box).
JEV_URL = "https://api.typesafe.ai/v1/systemone"

# `jev-latest` is an alias that moves with each release. Box questions need
# Jev 1.14+, which `jev-latest` points to today. If you tune thresholds on
# your own images, pin a versioned id (e.g. "jev-1.14.0") so they don't
# silently drift when the alias moves.
MODEL = "jev-latest"

# Images and Box questions are "Dreamer-ready" (pre-production). The API
# refuses them (HTTP 400) unless every such request carries this header.
IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}

# The name of the environment variable holding your key (from .env).
API_KEY_ENV = "TYPESAFE_API_KEY"

# Network behaviour. 429 = rate limited, 529 = overloaded: both are worth
# retrying with a growing pause. Anything else is a real error.
TIMEOUT_SECONDS = 30.0
RETRY_STATUSES = {429, 529}
MAX_ATTEMPTS = 3

# ── Image limits (from the Jev Images page) ─────────────────────────────────
# Each side must be 280..1568 px. Images outside are REJECTED, never resized,
# so we resize ourselves before sending.
MIN_SIDE = 280
MAX_SIDE = 1568
# Webcam frames are 640x480 or 1280x720: we cap the long side lower than the
# API max because a smaller frame means fewer input tokens (= cheaper) and
# laptop webcams don't have more detail than this to give anyway.
SEND_LONG_SIDE = 1024
JPEG_QUALITY = 88

# ── Decision layer defaults (all adjustable in the sidebar) ─────────────────
# Jev's docs suggest 0.7 / 0.3 as starting points for a Noul (a probability
# of "yes"). Between the two, the honest answer is "not sure": a person decides.
NOUL_YES = 0.70
NOUL_NO = 0.30
# For a Choice we threshold the probability of the label we'd act on.
CHOICE_ACCEPT = 0.80
# Box confidence: there's no recommended value yet (docs: "tune it on your
# own images"), so we start cautious.
BOX_ACCEPT = 0.60
# Score: below this level index the frame is "too poor to trust", and every
# automatic decision from it is sent to review instead.
QUALITY_MIN_LEVEL = 1.0

# ── Live mode ───────────────────────────────────────────────────────────────
# Each Jev call costs money, so live mode samples one frame every N seconds
# instead of sending every video frame.
DEFAULT_INTERVAL_SECONDS = 5

# ── Local files (git-ignored) ───────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "decisions.jsonl"
FRAMES_DIR = DATA_DIR / "frames"
