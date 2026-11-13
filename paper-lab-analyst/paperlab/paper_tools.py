"""The reviewer agent's tools: three ways to go back to the source.

These are *client tools*: Claude asks for one (a `tool_use` block), our Python
runs it, and we send the answer back as a `tool_result`. The model never
touches the PDF directly; it can only see what these functions return.

* `search_paper(query)`  → best-matching passages, with page numbers
* `read_page(page)`      → one page's full text
* `view_figure(figure_id)` → the figure image itself (vision inside a tool result)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .config import SETTINGS
from .llm import image_block
from .pdf_extract import Paper

# Tool definitions sent to the API. `strict: true` guarantees the arguments
# match the schema, so `run_tool` never sees a missing or mistyped field.
TOOLS = [
    {
        "name": "search_paper",
        "description": "Keyword search over the paper's full text (including OCR'd pages). "
                       "Returns the best-matching passages with their page numbers. Use it to "
                       "check a claim, find a number, or look for limitations the drafts missed.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Keywords, not a question."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_page",
        "description": "Return the full text of one page (1-based page number).",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"page": {"type": "integer"}},
            "required": ["page"],
            "additionalProperties": False,
        },
    },
    {
        "name": "view_figure",
        "description": "Look at a figure image yourself, by our figure id (e.g. \"F2\"). "
                       "Use it when a figure note seems doubtful or a finding depends on it.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"figure_id": {"type": "string"}},
            "required": ["figure_id"],
            "additionalProperties": False,
        },
    },
]

WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")  # keeps "p-value", "0.05", "il-6" whole


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


@dataclass
class Passage:
    page: int
    text: str
    tokens: Counter


class PaperIndex:
    """A tiny TF-IDF search engine over overlapping word windows.

    Each page is cut into ~120-word windows (stepping 80 words, so a sentence
    on a boundary appears whole in at least one window). A query's score for a
    window is the sum, over query words, of tf × idf: words that are rare in
    the paper (e.g. "phosphatase") count far more than common ones ("the").
    """

    def __init__(self, paper: Paper, window: int = 120, step: int = 80):
        self.paper = paper
        self.passages: list[Passage] = []
        for page in paper.pages:
            words = page.text.split()
            for start in range(0, max(len(words), 1), step):
                chunk = " ".join(words[start:start + window])
                if chunk:
                    self.passages.append(Passage(page.number, chunk, Counter(tokenize(chunk))))
                if start + window >= len(words):
                    break
        # Document frequency: in how many passages does each word appear?
        df = Counter(word for p in self.passages for word in p.tokens)
        n = len(self.passages) or 1
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}

    def search(self, query: str, k: int = SETTINGS.search_hits) -> list[Passage]:
        terms = set(tokenize(query))
        scored = []
        for p in self.passages:
            score = sum(p.tokens[t] * self.idf.get(t, 0.0) for t in terms)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda sp: -sp[0])
        return [p for _, p in scored[:k]]


def run_tool(name: str, args: dict, paper: Paper, index: PaperIndex) -> list[dict] | str:
    """Execute one tool call. Returns tool_result content (text, or image + text)."""
    if name == "search_paper":
        hits = index.search(args["query"])
        if not hits:
            return f"No passages match {args['query']!r}. Try other keywords."
        return "\n\n".join(f"[page {p.page}] {p.text}" for p in hits)

    if name == "read_page":
        page_no = args["page"]
        for page in paper.pages:
            if page.number == page_no:
                return f"=== Page {page_no} ===\n{page.text}"
        return f"No page {page_no}. Valid pages: 1-{len(paper.pages)}."

    if name == "view_figure":
        for fig in paper.figures:
            if fig.fig_id == args["figure_id"]:
                # A tool_result may hold images: this is how the reviewer
                # gets to *see* the figure, not just read about it.
                return [image_block(fig.image_png),
                        {"type": "text", "text": f"{fig.fig_id} (page {fig.page}) "
                                                 f"caption: {fig.caption or '(none)'}"}]
        valid = ", ".join(f.fig_id for f in paper.figures) or "none"
        return f"No figure {args['figure_id']!r}. Valid ids: {valid}."

    return f"Unknown tool {name!r}."
