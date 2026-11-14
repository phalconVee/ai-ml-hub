"""Every knob in one place. Read this file first.

All of these can be set in `.env` (see `.env.example`) and most can also be
changed live in the Streamlit sidebar. The defaults are the *safe* ones:
read-only browsing, a modest step budget, a headless browser.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Pull variables from a local `.env` into os.environ (real env vars win).
load_dotenv()


def _env(name: str, default: str) -> str:
    """An env var, or `default` when it is unset or blank."""
    return os.environ.get(name) or default


def _flag(name: str, default: bool) -> bool:
    """Read a yes/no env var: 1/true/yes/on mean True."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_domains(raw: str) -> tuple[str, ...]:
    """'Example.com, news.ycombinator.com\\n*.wikipedia.org' -> ('example.com', ...).

    Accepts commas, spaces or newlines, so it works for env vars and for the
    sidebar's text box. A leading '*.' or '.' is dropped: 'wikipedia.org'
    already covers every subdomain (see guards.py).
    """
    parts = raw.replace(",", " ").split()
    return tuple(p.strip().lower().lstrip("*").lstrip(".") for p in parts if p.strip())


# Tools that change something in the outside world. The agent never runs these
# without a human clicking "Approve" in the UI (see agent.py). Today that is
# just form submission; add a name here to gate another tool.
CONFIRM_TOOLS: frozenset[str] = frozenset({"submit_form"})

# Tools that only exist when read-only mode is OFF. In read-only mode the MCP
# server doesn't even register them, so Claude never sees them (see mcp_server.py).
WRITE_TOOLS: frozenset[str] = frozenset({"type_text", "submit_form"})


@dataclass
class Settings:
    # --- Model ------------------------------------------------------------
    # Claude Opus 5.5 is the current default model. Browsing is many small,
    # mechanical decisions, so `low` effort is usually enough; raise it for
    # harder multi-page goals.
    model: str = field(default_factory=lambda: _env("WEBPILOT_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: _env("WEBPILOT_EFFORT", "low"))
    # Beta header for the server-side refusal fallback (see llm.py).
    fallback_beta: str = "server-side-fallback-2026-07-01"

    # --- Guard rails ------------------------------------------------------
    # Max model turns per goal. Each turn may run one or more tool calls.
    max_steps: int = field(default_factory=lambda: int(_env("WEBPILOT_MAX_STEPS", "15")))
    # Empty allow list = any domain not on the deny list.
    allow: tuple[str, ...] = field(default_factory=lambda: parse_domains(_env("WEBPILOT_ALLOW", "")))
    deny: tuple[str, ...] = field(default_factory=lambda: parse_domains(_env("WEBPILOT_DENY", "")))
    # Read-only: no typing, no form submission. On by default.
    read_only: bool = field(default_factory=lambda: _flag("WEBPILOT_READ_ONLY", True))

    # --- Browser ----------------------------------------------------------
    headless: bool = field(default_factory=lambda: _flag("WEBPILOT_HEADLESS", True))
    nav_timeout_ms: int = 20_000
    # Page text is clipped to this many characters before it reaches Claude.
    max_page_chars: int = field(default_factory=lambda: int(_env("WEBPILOT_MAX_PAGE_CHARS", "8000")))
    # At most this many clickable elements are listed per read_page.
    max_elements: int = 120

    def server_env(self) -> dict[str, str]:
        """The guard-rail settings, encoded as env vars for the MCP subprocess.

        The server process reads these back in `mcp_server.main()`. Passing
        them at launch (not per call) means the model has no way to change them.
        """
        return {
            "WEBPILOT_ALLOW": ",".join(self.allow),
            "WEBPILOT_DENY": ",".join(self.deny),
            "WEBPILOT_READ_ONLY": "1" if self.read_only else "0",
            "WEBPILOT_HEADLESS": "1" if self.headless else "0",
            "WEBPILOT_MAX_PAGE_CHARS": str(self.max_page_chars),
        }
