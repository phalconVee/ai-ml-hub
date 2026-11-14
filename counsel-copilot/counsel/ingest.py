"""Step 1 of the pipeline: turn uploaded files into clause-sized chunks.

Legal documents have a gift for retrieval: they are already split into
numbered clauses ("4.2 Late Payment", "ARTICLE IX", "Section 12").
A clause is the unit a lawyer cites, so it is the unit we index and cite.

    PDF ─pypdf─▶ page text ─heading regex─▶ clauses ─too long?─▶ word windows

Two rules keep citations honest:
  • A chunk never crosses a page, so "page 3" in a citation is always true.
    A clause that runs onto the next page continues as a second chunk that
    keeps the same heading, marked "(cont.)".
  • A chunk never mixes two clauses, so a cited passage is one clause.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from .config import SETTINGS


@dataclass(frozen=True)
class Chunk:
    id: str        # stable handle the agents use: "lease.pdf#p3-c2"
    doc: str       # file name
    page: int      # 1-based page number
    heading: str   # the clause heading, e.g. "4.2 Late Payment" ("" if none)
    text: str      # the clause text (heading included)

    @property
    def title(self) -> str:
        """Human label shown in citations: 'lease.pdf, p.3, 4.2 Late Payment'."""
        label = f"{self.doc}, p.{self.page}"
        return f"{label}, {self.heading}" if self.heading else label


# A line that starts a new clause. Read it as "any of":
#   ARTICLE IV / Article 4 / SECTION 2 / Section 2.1 / § 7
#   4. Rent   /  4.2 Late Payment  /  4.2.1) Interest
# The numbered form needs a capital letter after the number, so a sentence
# like "3 days after notice" in the middle of a paragraph does not split.
HEADING = re.compile(
    r"""^\s*(
        (ARTICLE|Article|SECTION|Section|CLAUSE|Clause|§)\s*[0-9IVXLC]+(\.[0-9]+)*\b
      | [0-9]{1,2}(\.[0-9]{1,2}){0,3}[.)]?\s+[A-Z]
    )""",
    re.VERBOSE,
)


def read_pages(data: bytes, filename: str) -> list[str]:
    """Return the text of each page. PDFs via pypdf; .txt/.md are one page
    per form-feed (\\f) section, which is how plain-text exports mark pages."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader  # lazy: tests of other modules skip it

        reader = PdfReader(io.BytesIO(data))
        # extract_text() can return None on image-only pages (scans): treat as empty.
        return [(page.extract_text() or "") for page in reader.pages]
    return data.decode("utf-8", errors="replace").split("\f")


def split_clauses(page_text: str) -> list[tuple[str, str]]:
    """Split one page into (heading, text) pairs at heading lines.

    Text before the first heading (a title, the parties block, or the tail of
    a clause from the previous page) comes back with heading "".
    """
    clauses: list[tuple[str, list[str]]] = [("", [])]
    for line in page_text.splitlines():
        if not line.strip():
            continue
        if HEADING.match(line):
            # A heading line is short; keep at most the first 60 chars as its label.
            clauses.append((line.strip()[:60], [line.strip()]))
        else:
            clauses[-1][1].append(line.strip())
    return [(h, " ".join(lines)) for h, lines in clauses if lines]


def windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    """Cut a long word list into overlapping windows: [0:size], [size-overlap: ...]."""
    if len(words) <= size:
        return [words]
    step = size - overlap
    return [words[i:i + size] for i in range(0, len(words) - overlap, step)]


def chunk_document(data: bytes, filename: str) -> list[Chunk]:
    """The whole ingest step for one file."""
    chunks: list[Chunk] = []
    last_heading = ""  # carried across pages so continuations keep their label
    for page_no, page_text in enumerate(read_pages(data, filename), start=1):
        for heading, text in split_clauses(page_text):
            if not heading and last_heading and chunks:
                # Orphan text at the top of a page = the previous clause continuing.
                heading = f"{last_heading} (cont.)"
            elif heading:
                last_heading = heading
            for piece in windows(text.split(), SETTINGS.max_clause_words, SETTINGS.window_overlap):
                n = len(chunks) + 1
                chunks.append(Chunk(f"{filename}#p{page_no}-c{n}", filename, page_no,
                                    heading, " ".join(piece)))
    return chunks
