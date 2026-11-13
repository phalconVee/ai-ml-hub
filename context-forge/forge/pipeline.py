"""The whole flow, start to finish: plan → gather → rank → answer → remember.

    question
       │
   [1 Planner]  which sources, what query for each        (1 small LLM call)
       │
   [2 Gather]   docs · web · papers · memory, in parallel  (threads; each may fail alone)
       │
   [3 Ranker]   relevance · judge · trust · dedupe · caps · token budget
       │
   [4 Synth]    answer from the window only, citations checked
       │
   [5 Memory]   save Q+A so later questions can retrieve it

Everything is recorded on a `Run` object, which is what the UI renders: the
plan, every candidate with its kept/dropped reason, the exact context string,
timings, and any source that failed.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .items import ContextItem
from .planner import Plan, make_plan
from .ranker import rank
from .sources.documents import DocumentStore
from .sources.memory import MemoryStore
from .sources.papers import search_arxiv
from .sources.web import search_web
from .synthesizer import Answer, render_context, synthesize

ALL_SOURCES = ("doc", "web", "paper", "memory")


@dataclass
class Run:
    question: str
    plan: Plan | None = None
    candidates: list[ContextItem] = field(default_factory=list)  # everything gathered
    kept: list[ContextItem] = field(default_factory=list)        # what entered the window
    context: str = ""                                            # the exact window text
    answer: Answer | None = None
    web_queries: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)         # source → error message
    timings: dict[str, float] = field(default_factory=dict)      # stage → seconds


def run(question: str, *, client: Any, docs: DocumentStore, memory: MemoryStore,
        enabled: frozenset[str] | set[str] = frozenset(ALL_SOURCES), use_judge: bool = False,
        budget: int | None = None, arxiv: Callable = search_arxiv,
        web: Callable = search_web) -> Run:
    """`arxiv` and `web` are injectable so tests can swap in fakes."""
    r = Run(question)
    clock = time.perf_counter

    # 1 ── Plan ────────────────────────────────────────────────────────────────
    t = clock()
    r.plan = plan = make_plan(question, client)
    r.timings["plan"] = clock() - t

    # 2 ── Gather (parallel: arXiv and web are slow network calls) ────────────
    # The user's toggles win; the planner may only switch web/papers *off*.
    jobs: dict[str, Callable[[], list[ContextItem]]] = {}
    if "doc" in enabled:
        jobs["doc"] = lambda: docs.search(plan.doc_query or question)
    if "memory" in enabled:
        jobs["memory"] = lambda: memory.search(question)
    if "paper" in enabled and plan.use_papers:
        jobs["paper"] = lambda: arxiv(plan.paper_query or question)
    if "web" in enabled and plan.use_web:
        def web_job():
            items, queries = web(plan.web_query or question, client)
            r.web_queries = queries
            return items
        jobs["web"] = web_job

    t = clock()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, fut in futures.items():
            try:
                r.candidates += fut.result()
            except Exception as e:  # one broken source must not sink the answer
                r.errors[name] = f"{type(e).__name__}: {e}"
    r.timings["gather"] = clock() - t

    # 3 ── Rank & select ─────────────────────────────────────────────────────
    t = clock()
    r.kept = rank(question, r.candidates, client=client, use_judge=use_judge, budget=budget)
    r.context = render_context(r.kept)
    r.timings["rank"] = clock() - t

    # 4 ── Answer ────────────────────────────────────────────────────────────
    t = clock()
    r.answer = synthesize(question, r.kept, client)
    r.timings["answer"] = clock() - t

    # 5 ── Remember (only real answers, not "found nothing" messages) ────────
    if r.kept and r.answer.cited:
        memory.remember(question, r.answer.text)
    return r
