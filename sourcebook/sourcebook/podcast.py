"""Step 7a: write the audio overview - a short two-host conversation script.

Two hosts make a summary easier to listen to than one voice reading notes:
one asks the questions a listener would, the other explains. Claude writes the
script with **structured outputs**, so the reply is guaranteed JSON of the
shape {title, turns: [{speaker, text}]} - no parsing of "MAYA:" lines, no
stray stage directions. tts.py then voices it.

What Claude sees is a *digest* of the notebook: each enabled source's passages
in reading order, up to a word budget per source and overall, so one huge PDF
can't crowd out the others.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import llm
from .config import SETTINGS

HOSTS = {"A": "Maya", "B": "Theo"}

SCRIPT_SYSTEM = f"""You write scripts for a short audio overview of a user's research notebook, spoken by two hosts:
- {HOSTS['A']} (speaker "A"): curious and warm; asks the questions a smart listener would and reacts naturally.
- {HOSTS['B']} (speaker "B"): has read every source closely; explains clearly with concrete details and examples from them.

Rules:
- Use only facts from the provided sources. Mention which source a point comes from when it helps ("the second article argues...").
- Open with a one-line hook that names the topic; end with the single most useful takeaway.
- 14 to 22 turns, alternating speakers, starting with A. Each turn 1-3 spoken sentences. About 450-650 words in total.
- Write for the ear: no markdown, bullet points, emojis, URLs or stage directions. Spell out symbols and abbreviations a voice would stumble on."""

SCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker": {"type": "string", "enum": ["A", "B"]},
                    "text": {"type": "string"},
                },
                "required": ["speaker", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "turns"],
    "additionalProperties": False,
}


@dataclass
class Script:
    title: str
    turns: list[tuple[str, str]] = field(default_factory=list)   # (speaker "A"/"B", text)

    def transcript(self) -> str:
        return "\n\n".join(f"**{HOSTS[s]}:** {t}" for s, t in self.turns)

    @property
    def word_count(self) -> int:
        return sum(len(t.split()) for _, t in self.turns)

    def save(self, path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"title": self.title, "turns": self.turns}, indent=1))

    @classmethod
    def load(cls, path) -> "Script | None":
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return cls(d["title"], [tuple(t) for t in d["turns"]])


def build_digest(notebook, per_source: int = SETTINGS.podcast_words_per_source,
                 total: int = SETTINGS.podcast_words_total) -> str:
    """Concatenate each enabled source's passages up to the word budgets."""
    enabled = [s for s in notebook.sources.values() if s.enabled]
    if not enabled:
        return ""
    per_source = min(per_source, total // len(enabled))   # share the total fairly
    by_source: dict[str, list[str]] = {s.id: [] for s in enabled}
    seen: set[str] = set()
    for p in notebook.passages:
        if p.source_id not in by_source:
            continue
        # Passages overlap by a sentence; drop sentences we've already included.
        fresh = [s for s in p.sentences if (p.source_id, s) not in seen]
        seen.update((p.source_id, s) for s in fresh)
        by_source[p.source_id].extend(fresh)
    parts = []
    for i, s in enumerate(enabled, start=1):
        words, kept = 0, []
        for sent in by_source[s.id]:
            words += len(sent.split())
            if words > per_source:
                break
            kept.append(sent)
        parts.append(f"### Source {i}: {s.title}\n" + " ".join(kept))
    return "\n\n".join(parts)


def generate_script(notebook, client=None, focus: str = "") -> Script:
    client = client or llm.make_client()
    digest = build_digest(notebook)
    if not digest:
        raise ValueError("The notebook has no enabled sources.")
    prompt = f"<sources>\n{digest}\n</sources>\n\n"
    prompt += f"Focus the episode on: {focus}" if focus.strip() else "Cover the most important ideas across all sources."
    # Script writing benefits from a bit more thought than the helper calls.
    data = llm.create_json(client, system=SCRIPT_SYSTEM, prompt=prompt,
                           schema=SCRIPT_SCHEMA, effort=SETTINGS.effort)
    turns = [(t["speaker"], t["text"].strip()) for t in data["turns"] if t["text"].strip()]
    return Script(title=data["title"].strip(), turns=turns)
