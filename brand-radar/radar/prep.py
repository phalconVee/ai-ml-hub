"""Clean the raw haul before any AI sees it. Plain Python, no Claude.

Why bother? Every mention sent to an analyst costs tokens, and search engines
return plenty of junk: the same article syndicated to five sites, pages that
mention "notion" as in "the notion of", empty snippets. Throwing those away
here is free; throwing them away inside Claude is not.

Three steps, in order:
  1. keep only mentions that actually contain a brand term (and no excluded phrase)
  2. drop duplicates (same URL, or same opening text on a different URL)
  3. clip long text and give every survivor a short, citable id like "news-3"
"""

import re

from radar.schemas import BrandQuery, Mention


def mentions_brand(m: Mention, q: BrandQuery) -> bool:
    """Does the text name the brand as a word, and avoid every excluded phrase?

    Word boundaries matter: brand "Arc" must not match "search" or "archive".
    """
    hay = f"{m.title} {m.text}".lower()
    if any(x.lower() in hay for x in q.exclude if x.strip()):
        return False
    return any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", hay) for t in q.terms)


def _fingerprint(m: Mention) -> str:
    """First 8 words of the text, letters and digits only: catches syndicated
    copies, which often share an opening but append different endings."""
    words = re.findall(r"[a-z0-9]+", m.text.lower())
    return " ".join(words[:8])


def _canonical_url(url: str) -> str:
    """Same page, different spelling: drop scheme, www, #fragment, tracking
    parameters (utm_*, ref, fbclid...) and trailing slash.

    The rest of the query string MUST stay: on Hacker News (`item?id=123`) and
    YouTube (`watch?v=abc`) it is the only thing telling two pages apart.
    """
    url = re.sub(r"^https?://(www\.)?", "", url.strip().lower()).split("#")[0]
    path, _, query = url.partition("?")
    keep = [kv for kv in query.split("&") if kv and not re.match(r"(utm_[a-z]+|ref|fbclid|gclid)=", kv)]
    return path.rstrip("/") + ("?" + "&".join(keep) if keep else "")


def prepare(raw: dict[str, list[Mention]], q: BrandQuery, max_chars: int = 600) -> tuple[dict[str, list[Mention]], dict[str, int]]:
    """Filter, de-duplicate and label. Returns (clean mentions per source, dropped counts per source).

    De-duplication is GLOBAL across sources: if a news article also shows up as
    a web result, only the first copy (in source order) survives.
    """
    seen_urls: set[str] = set()
    seen_text: set[str] = set()
    out: dict[str, list[Mention]] = {}
    dropped: dict[str, int] = {}

    for source, mentions in raw.items():
        kept: list[Mention] = []
        for m in mentions:
            url, fp = _canonical_url(m.url), _fingerprint(m)
            is_dupe = (url and url in seen_urls) or (len(fp.split()) >= 6 and fp in seen_text)
            if not m.text or is_dupe or not mentions_brand(m, q):
                continue
            seen_urls.add(url)
            seen_text.add(fp)
            if len(m.text) > max_chars:
                m.text = m.text[:max_chars].rsplit(" ", 1)[0] + " ..."
            m.id = f"{source}-{len(kept) + 1}"
            kept.append(m)
        out[source] = kept
        dropped[source] = len(mentions) - len(kept)
    return out, dropped
