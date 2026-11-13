"""Check the report against the PDF, in plain Python. No LLM.

Every finding and limitation carries a "verbatim" quote + page. An LLM can
still paraphrase, misremember the page, or invent a quote outright. So after
the reviewer finishes, we look each quote up in the extracted text:

* ✅ `verified`   — the quote's words appear on the cited page
* 📄 `wrong_page` — found, but on a different page (we report which)
* ⚠️ `not_found`  — not in the paper: treat the claim with suspicion

Matching is forgiving about things PDFs mangle (line-break hyphens, curly
quotes, ligatures, whitespace) but strict about words: a quote is found when
at least `quote_match_threshold` of its words appear inside one window of the
page text that is as long as the quote.

We also drop figure ids the paper doesn't have (a typo like "F9" would
otherwise link to nothing).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .config import SETTINGS
from .paper_tools import tokenize
from .pdf_extract import Paper


@dataclass
class QuoteCheck:
    status: str          # "verified" | "wrong_page" | "not_found"
    cited_page: int
    found_page: int | None
    score: float         # best word-coverage found, 0..1


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)        # "ﬁ" ligature → "fi", full-width → ascii
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # "phos-\nphatase" → "phosphatase"
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split()).lower()


def coverage(quote_tokens: list[str], page_tokens: list[str]) -> float:
    """Best share of the quote's words found in any same-length window of the page.

    A sliding window keeps a running word count, so each step is O(1):
    add the word entering the window, remove the word leaving it.
    """
    n = len(quote_tokens)
    if n == 0 or not page_tokens:
        return 0.0
    need = Counter(quote_tokens)
    window = Counter(page_tokens[:n])
    best = sum(min(c, window[w]) for w, c in need.items())
    for i in range(n, len(page_tokens)):
        window[page_tokens[i]] += 1
        window[page_tokens[i - n]] -= 1
        # Only recount when the entering word matters (cheap shortcut).
        if page_tokens[i] in need:
            best = max(best, sum(min(c, window[w]) for w, c in need.items()))
    return best / n


def check_quote(quote: str, cited_page: int, paper: Paper,
                threshold: float = SETTINGS.quote_match_threshold) -> QuoteCheck:
    q_norm = normalise(quote)
    q_tokens = tokenize(q_norm)
    pages = {p.number: normalise(p.text) for p in paper.pages}

    def score_on(page_no: int) -> float:
        text = pages.get(page_no, "")
        if q_norm and q_norm in text:          # exact (normalised) match: done
            return 1.0
        return coverage(q_tokens, tokenize(text))

    # Cited page first: it's the claim we're testing.
    cited_score = score_on(cited_page)
    if cited_score >= threshold:
        return QuoteCheck("verified", cited_page, cited_page, cited_score)

    best_page, best_score = None, cited_score
    for page_no in pages:
        if page_no != cited_page:
            s = score_on(page_no)
            if s > best_score:
                best_page, best_score = page_no, s
    if best_page is not None and best_score >= threshold:
        return QuoteCheck("wrong_page", cited_page, best_page, best_score)
    return QuoteCheck("not_found", cited_page, None, best_score)


def ground_report(report: dict, paper: Paper) -> dict:
    """Annotate the report in place with `grounding` on every quoted item.

    Returns a summary: counts per status, plus any unknown figure ids removed.
    """
    valid_ids = {f.fig_id for f in paper.figures}
    summary = {"verified": 0, "wrong_page": 0, "not_found": 0, "dropped_figure_ids": []}

    def annotate(item: dict) -> None:
        ev = item.get("evidence") or {}
        check = check_quote(ev.get("quote", ""), int(ev.get("page", 0)), paper)
        item["grounding"] = {"status": check.status, "found_page": check.found_page,
                             "score": round(check.score, 2)}
        summary[check.status] += 1

    for finding in report.get("key_findings", []):
        annotate(finding)
        kept = [fid for fid in finding.get("figure_ids", []) if fid in valid_ids]
        summary["dropped_figure_ids"] += [f for f in finding.get("figure_ids", []) if f not in valid_ids]
        finding["figure_ids"] = kept
    for lim in report.get("limitations", []):
        annotate(lim)

    insights = report.get("figure_insights", [])
    summary["dropped_figure_ids"] += [i["figure_id"] for i in insights if i["figure_id"] not in valid_ids]
    report["figure_insights"] = [i for i in insights if i["figure_id"] in valid_ids]
    return summary
