"""Every knob in one place. Override any of them with an environment variable."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")  # ANTHROPIC_API_KEY and any keys your own servers need

# Models offered in the sidebar. The first one is the default.
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
DEFAULT_MODEL = os.environ.get("SWITCHBOARD_MODEL", MODELS[0])
# Effort = how hard Claude thinks before answering (and how many tokens that costs).
EFFORTS = ["low", "medium", "high"]
DEFAULT_EFFORT = os.environ.get("SWITCHBOARD_EFFORT", "medium")

CONFIG_PATH = Path(os.environ.get("SWITCHBOARD_CONFIG", PROJECT_DIR / "servers.json"))

MAX_AGENT_STEPS = 12          # model calls per user message before we stop the loop
MAX_TOKENS = 16000            # per model call
MAX_TOOL_RESULT_CHARS = 20000 # clip huge tool output before it reaches the model
CONNECT_TIMEOUT_S = 30.0      # starting a stdio server (e.g. `npx` downloading a package) can be slow
TOOL_TIMEOUT_S = 90.0         # a single tool call
