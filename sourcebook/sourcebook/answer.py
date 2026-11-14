"""Step 5 (the core): answer a question from the sources, with exact citations.

One question = this pipeline:

    question --(memory: rewrite follow-up)--> standalone query
             --(retriever)-->                 top-k passages
             --(Claude, citations on)-->      answer text + citation objects
             --(map back)-->                  [n] -> source, page, exact sentences

The grounding trick: passages go to Claude as `search_result` content blocks
with `citations: {enabled: true}`, placed directly in the user message (no
tool call needed, we already did the search). Each passage's content is a list
of one-sentence text blocks. Claude's answer then comes back split into text
blocks, and every cited block carries `search_result_location` citations that
say *which* passage (search_result_index) and *which* sentences
(start_block_index .. end_block_index, end exclusive). We never ask the model
to type "[1]" and hope; the API returns the links as data.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import llm
from .chunk import Passage
from .config import SETTINGS

SYSTEM_PROMPT = """You are Sourcebook, a research assistant that answers only from the user's own sources.

For each question you receive the most relevant passages from those sources as search results.
- Ground every factual statement in the search results and cite them. Do not use outside knowledge, even if you are confident.
- If the results don't answer the question, say so plainly, then say briefly what the sources do cover on the topic.
- When sources disagree, say so and cite each side.
- Earlier turns of the conversation are context for resolving what the user means, not evidence. Re-ground any claim in the current search results.
- Be concise: lead with the direct answer, then supporting detail in short paragraphs or bullets."""


@dataclass
class Citation:
    number: int            # the [n] shown in the answer
    source_id: str
    source_title: str
    page: int
    passage_id: str
    sentences: list[str]   # the whole passage, sentence by sentence
    start: int             # first cited sentence (index into `sentences`)
    end: int               # one past the last cited sentence
    cited_text: str        # what the API says was cited
    url: str | None = None  # for web sources: a link that jumps to the passage
    kind: str = "text"     # "pdf" | "text" | "web"

    @property
    def location(self) -> str:
        """Human label: PDFs get a page number, other sources are one page."""
        return f"{self.source_title}, p. {self.page}" if self.kind == "pdf" else self.source_title


@dataclass
class Answer:
    question: str
    search_query: str = ""
    # The answer as (text, [citation numbers]) segments, in order, so a UI can
    # put each [n] exactly where Claude attached it.
    segments: list[tuple[str, list[int]]] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    passages_used: int = 0
    stop_reason: str = ""

    def plain_text(self) -> str:
        return "".join(t for t, _ in self.segments).strip()

    def markdown(self) -> str:
        """Answer with [n] markers inline, e.g. for printing in a terminal."""
        return "".join(t + "".join(f"[{n}]" for n in nums) for t, nums in self.segments).strip()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["segments"] = [list(s) for s in self.segments]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Answer":
        d = dict(d)
        d["segments"] = [(t, list(n)) for t, n in d.get("segments", [])]
        d["citations"] = [Citation(**c) for c in d.get("citations", [])]
        return cls(**d)


def text_fragment_link(url: str, quote: str) -> str:
    """A URL that scrolls to and highlights `quote` in most browsers.

    Uses the standard text-fragment syntax `#:~:text=start,end`. For long
    quotes we match the first and last few words, which survives small
    whitespace differences between our extracted text and the live page.
    """
    words = quote.split()
    enc = lambda s: urllib.parse.quote(s, safe="").replace("-", "%2D")
    frag = enc(" ".join(words)) if len(words) <= 10 else enc(" ".join(words[:5])) + "," + enc(" ".join(words[-5:]))
    return f"{url.split('#')[0]}#:~:text={frag}"


def passage_block(p: Passage, title: str, url: str | None) -> dict:
    """One retrieved passage as a citable `search_result` block."""
    return {
        "type": "search_result",
        "source": url or f"{title}#page={p.page}",
        "title": f"{title}" + ("" if url else f", page {p.page}"),
        # One text block per sentence: citations can point at single sentences.
        "content": [{"type": "text", "text": s} for s in p.sentences],
        "citations": {"enabled": True},
    }


def collect(response_content, sent: list[Passage], sources: dict, answer: Answer) -> None:
    """Walk Claude's text blocks; number each distinct cited sentence range."""
    seen: dict[tuple, int] = {}
    for block in response_content:
        if block.type != "text":
            continue   # thinking blocks, fallback markers...
        nums: list[int] = []
        for cite in block.citations or []:
            if cite.type != "search_result_location":
                continue
            idx = cite.search_result_index
            if not (0 <= idx < len(sent)):
                continue   # defensive: never trust an index blindly
            p = sent[idx]
            start = max(0, cite.start_block_index)
            end = min(len(p.sentences), max(start + 1, cite.end_block_index))
            key = (p.id, start, end)
            if key not in seen:
                seen[key] = len(seen) + 1
                src = sources[p.source_id]
                answer.citations.append(Citation(
                    number=seen[key], source_id=p.source_id, source_title=src.title,
                    page=p.page, passage_id=p.id, sentences=p.sentences, start=start, end=end,
                    cited_text=cite.cited_text,
                    url=text_fragment_link(src.url, cite.cited_text) if src.url else None,
                    kind=src.kind,
                ))
            if seen[key] not in nums:
                nums.append(seen[key])
        answer.segments.append((block.text, nums))


def ask(notebook, question: str, client=None,
        on_event: Callable[[str], None] = lambda _m: None) -> Answer:
    """Answer one question against `notebook`, remember it, and return it."""
    client = client or llm.make_client()
    answer = Answer(question=question)
    sources = {s.id: s for s in notebook.sources.values() if s.enabled}
    if not sources:
        answer.segments.append(("Add a source (or tick one in the sources panel) and ask again.", []))
        return answer

    # 1. Memory: turn a follow-up into a standalone search query.
    on_event("Understanding the question...")
    answer.search_query = notebook.memory.standalone_query(question, client)

    # 2. Retrieval, restricted to the ticked sources.
    on_event(f"Searching sources for: {answer.search_query!r}")
    sent = notebook.retriever.search(answer.search_query, k=SETTINGS.top_k,
                                     allowed_sources=set(sources))
    answer.passages_used = len(sent)

    # 3. One grounded call. The user turn = search results first, then the
    #    original question (not the rewrite: Claude should answer what was asked).
    if sent:
        content = [passage_block(p, sources[p.source_id].title, sources[p.source_id].url) for p in sent]
    else:
        content = [{"type": "text", "text": "(No passages matched this question.)"}]
    content.append({"type": "text", "text": question})

    system = SYSTEM_PROMPT + "\n\nSources in this notebook: " + "; ".join(s.title for s in sources.values())
    if notebook.memory.summary:
        system += "\n\nSummary of the earlier conversation: " + notebook.memory.summary

    on_event("Writing a cited answer...")
    response = llm.create(client, system=system, effort=SETTINGS.effort,
                          messages=notebook.memory.as_messages() + [{"role": "user", "content": content}])
    answer.stop_reason = response.stop_reason or ""

    # 4. Map citations back to passages and sentences.
    if response.stop_reason == "refusal":
        answer.segments.append(("The model declined to answer this request.", []))
    else:
        collect(response.content, sent, sources, answer)
        if response.stop_reason == "max_tokens":
            answer.segments.append(("\n\n(Answer cut off at the length limit.)", []))

    # 5. Remember the turn (may fold old turns into the running summary).
    notebook.memory.add(answer, client)
    return answer
