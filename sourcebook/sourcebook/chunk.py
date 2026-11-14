"""Step 2: cut each source into passages, and each passage into sentences.

Why two levels?

- A *passage* (~120 words) is the unit of retrieval: big enough to carry
  context, small enough that the top 8 fit comfortably in one prompt.
- A *sentence* is the unit of citation. Each passage goes to Claude as a
  `search_result` block whose content is a list of text blocks, one per
  sentence. Claude's citations then name the exact sentence range it used,
  so the UI can highlight just those sentences, not a whole page.

Passages never cross a page boundary, so every citation has one page number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import SETTINGS
from .sources import Source

# A sentence ends at . ! or ? followed by whitespace and something that looks
# like the start of a new sentence. Not perfect (e.g. "Dr. Smith"), but a
# missed split only makes one citation a little longer.
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass
class Passage:
    id: str               # "<source id>:<n>", unique within a notebook
    source_id: str
    page: int             # 1-based
    sentences: list[str]  # the citation units, in order

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


def split_sentences(paragraph: str) -> list[str]:
    paragraph = " ".join(paragraph.split())   # collapse internal newlines
    if not paragraph:
        return []
    # Headings and list items are short; keep them as one unit each.
    if paragraph.startswith("#") or len(paragraph.split()) < 8:
        return [paragraph]
    return [s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()]


def chunk_source(source: Source, words: int = SETTINGS.passage_words,
                 overlap: int = SETTINGS.overlap_sentences) -> list[Passage]:
    passages: list[Passage] = []
    for page_no, page_text in enumerate(source.pages, start=1):
        # Paragraphs first (blank lines), then sentences inside them.
        sentences: list[str] = []
        for para in re.split(r"\n\s*\n", page_text):
            sentences.extend(split_sentences(para))

        current: list[str] = []
        count = 0
        for sent in sentences:
            n = len(sent.split())
            # Close the passage when adding this sentence would overflow it.
            if current and count + n > words:
                passages.append(Passage(f"{source.id}:{len(passages)}", source.id, page_no, current))
                current = current[-overlap:] if overlap else []   # carry overlap forward
                count = sum(len(s.split()) for s in current)
            current.append(sent)
            count += n
        # Flush the tail. (It always holds at least one new sentence: the
        # overlap is only carried forward right before a sentence is added.)
        if current:
            passages.append(Passage(f"{source.id}:{len(passages)}", source.id, page_no, current))
    return passages
