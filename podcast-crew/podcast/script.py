"""The script as data, plus the plain-code checks every script must pass.

A script is a list of (speaker, text) lines. The two hosts are always the
keys "A" and "B" internally; their display names ("Maya", "Theo") are only
attached for the prompt and the UI. Using fixed keys means a model that
misspells a name can't create a third, unvoiced speaker.

`lint()` is the mechanical half of editing. Things code can check perfectly
and for free (stage directions the TTS would read aloud, lines that are too
long, one host hogging the mic) are found here, then handed to the LLM
editor as a to-do list. `clean_text()` is the last line of defence: whatever
the editor returns, bracketed directions and markdown never reach the voice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import SETTINGS

SPEAKERS = ("A", "B")

# The JSON shape both the scriptwriter and the editor must return.
LINES_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "speaker": {"type": "string", "enum": list(SPEAKERS)},  # only A or B is possible
            "text": {"type": "string"},
        },
        "required": ["speaker", "text"],
        "additionalProperties": False,
    },
}


@dataclass
class Line:
    speaker: str   # "A" or "B"
    text: str


@dataclass
class Script:
    title: str
    lines: list[Line] = field(default_factory=list)

    @classmethod
    def from_json(cls, title: str, lines: list[dict]) -> "Script":
        return cls(title, [Line(l["speaker"], l["text"]) for l in lines])

    def words(self) -> int:
        return sum(len(l.text.split()) for l in self.lines)

    def minutes(self) -> float:
        """Estimated spoken length."""
        return self.words() / SETTINGS.words_per_minute

    def as_text(self, names: dict[str, str]) -> str:
        """`Maya: …` lines, the way the model reads (and you download) a script."""
        return "\n".join(f"{names[l.speaker]}: {l.text}" for l in self.lines)

    def markdown(self, names: dict[str, str]) -> str:
        body = "\n\n".join(f"**{names[l.speaker]}:** {l.text}" for l in self.lines)
        return f"# {self.title}\n\n{body}\n"


# ── Cleaning: what the voice must never see ─────────────────────────────────

# [anything in square brackets] is always a direction. In (round) brackets or
# *asterisks* only when it names an action, so "(about 40%)" and "*really*"
# keep their words.
ACTIONS = r"laugh|sigh|pause|chuckle|music|beat|giggle|clears throat|smil|grin|intro|outro"
STAGE = re.compile(rf"\[[^\]]*\]|\([^)]*({ACTIONS})[^)]*\)|\*[^*]*({ACTIONS})[^*]*\*",
                   re.IGNORECASE)


def clean_text(text: str) -> str:
    """Remove [laughs], (pause), *sighs*, markdown and odd spacing."""
    text = STAGE.sub("", text)
    text = re.sub(r"[*#_`>]", "", text)       # stray markdown symbols (emphasis keeps its word)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # "word ," → "word,"
    return text


def tidy(script: Script) -> Script:
    """Clean every line and drop lines that end up empty."""
    lines = [Line(l.speaker, clean_text(l.text)) for l in script.lines]
    return Script(script.title, [l for l in lines if l.text])


# ── Lint: mechanical problems, phrased as notes for the editor ──────────────

def lint(script: Script, target_words: int) -> list[str]:
    issues: list[str] = []
    if len(script.lines) < SETTINGS.min_lines:
        issues.append(f"Only {len(script.lines)} lines; a conversation needs at least {SETTINGS.min_lines}.")
    for i, line in enumerate(script.lines, 1):
        n = len(line.text.split())
        if n > SETTINGS.max_line_words:
            issues.append(f"Line {i} is {n} words; split it or trim to under {SETTINGS.max_line_words}.")
        if STAGE.search(line.text):
            issues.append(f"Line {i} has a stage direction or markdown; the voice would read it aloud.")
        if re.search(r"https?://", line.text):
            issues.append(f"Line {i} contains a URL; say the site's name instead.")
    # Monologue check: the same host speaking 3+ lines in a row.
    run = 1
    for prev, cur in zip(script.lines, script.lines[1:]):
        run = run + 1 if cur.speaker == prev.speaker else 1
        if run == 3:
            issues.append(f"Host {cur.speaker} speaks 3+ lines in a row; let the other host react.")
    # Balance: neither host should carry less than a quarter of the words.
    total = script.words() or 1
    for s in SPEAKERS:
        share = sum(len(l.text.split()) for l in script.lines if l.speaker == s) / total
        if share < 0.25:
            issues.append(f"Host {s} has only {share:.0%} of the words; balance the conversation.")
    # Length: more than 30% off the target is worth fixing.
    if total > 1.3 * target_words:
        issues.append(f"Script is {total} words; target is about {target_words}. Cut.")
    elif total < 0.7 * target_words:
        issues.append(f"Script is {total} words; target is about {target_words}. It may feel thin.")
    return issues
