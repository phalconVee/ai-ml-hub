"""Recall: which memories matter for *this* message?

No embeddings and no LLM call: the graph itself is the index. Three steps,
all plain Python, all explainable (the UI shows exactly which facts were used):

  1. seed     find the entities the message mentions (name, alias, or words
              of their description), each with a match score. The User is
              always a seed: in a personal assistant, the person is context.
  2. spread   walk outwards from the seeds through the graph, up to `hops`
              steps, halving the relevance at every step ("spreading
              activation"). Ask about "my sister" → User → Ada → Ada's job.
  3. rank     score every edge between activated nodes by its endpoints'
              relevance plus a little for recency and repetition, and keep
              the top `max_facts`. That's the context budget for the reply.

Why a graph beats "search the chat log": the answer to "what should I get
my sister for her birthday?" lives two hops away (sister → Ada → likes →
pottery), in a sentence that shares no words with the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import networkx as nx

from .config import SETTINGS
from .graph import Fact, KnowledgeGraph, norm_key

STOPWORDS = set("""a an the and or but if of to in on at for with about from by as is are was
were be been being do does did have has had i me my mine myself you your yours we our it its
this that these those what which who whom whose when where why how can could should would will
just so than then there here also very really tell know think please some any all""".split())

DECAY = 0.5          # relevance multiplier per hop
USER_BASE = 1.0      # the User is always somewhat relevant...
USER_PERSONAL = 2.5  # ...and very relevant when the message is about "me/my/I"
PERSONAL = {"i", "me", "my", "mine", "myself", "i'm", "im", "i've"}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def content_words(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS and len(t) > 1}


@dataclass
class Recall:
    """What retrieval found, kept whole so the UI can show its working."""
    facts: list[Fact] = field(default_factory=list)
    seeds: dict[str, float] = field(default_factory=dict)       # entity → match score
    activation: dict[str, float] = field(default_factory=dict)  # entity → relevance after spreading

    def as_context(self) -> str:
        """The block of text Claude reads. Empty memory says so explicitly."""
        if not self.facts:
            return "(nothing relevant in memory yet)"
        return "\n".join(f"- {f.sentence()}" for f in self.facts)


def match_score(query: str, name: str, aliases: list[str], description: str) -> float:
    """How strongly does the message mention this entity?"""
    q_norm = " " + " ".join(tokens(query)) + " "
    q_words = content_words(query)
    best = 0.0
    for label in [name, *aliases]:
        key = norm_key(label)
        if not key:
            continue
        if f" {key} " in q_norm:
            best = max(best, 3.0)                      # the whole name appears
        else:
            words = content_words(key)
            if words:
                overlap = len(words & q_words) / len(words)
                best = max(best, 2.0 * overlap)        # "Rust" in "the Rust book"
    # Weak signal: the message shares words with what we know about it.
    desc_hits = len(content_words(description) & q_words)
    return best + min(desc_hits, 3) * 0.3


def recall(graph: KnowledgeGraph, query: str, *, hops: int = SETTINGS.hops,
           max_facts: int = SETTINGS.max_facts) -> Recall:
    g = graph.to_networkx()
    out = Recall()
    if g.number_of_edges() == 0:
        return out

    # ── 1. seed ──────────────────────────────────────────────────────────────
    for node, data in g.nodes(data=True):
        s = match_score(query, node, data.get("aliases", []), data.get("description", ""))
        if s > 0:
            out.seeds[node] = s
    user = SETTINGS.user_node
    personal = bool(PERSONAL & set(tokens(query)))
    out.seeds[user] = max(out.seeds.get(user, 0.0), USER_PERSONAL if personal else USER_BASE)

    # ── 2. spread (breadth-first, ignoring edge direction) ───────────────────
    undirected = g.to_undirected(as_view=True)
    for seed, score in out.seeds.items():
        # Shortest hop distance from this seed to every node within reach.
        dist = nx.single_source_shortest_path_length(undirected, seed, cutoff=hops)
        for node, d in dist.items():
            out.activation[node] = max(out.activation.get(node, 0.0), score * DECAY ** d)

    # ── 3. rank edges ────────────────────────────────────────────────────────
    latest = max((f for _, _, f in g.edges(data="last_turn")), default=1) or 1
    scored: list[tuple[float, Fact]] = []
    for u, v, data in g.edges(data=True):
        a = max(out.activation.get(u, 0.0), out.activation.get(v, 0.0))
        if a == 0:
            continue                                   # not reached from any seed
        fact: Fact = data["fact"]
        recency = 0.3 * fact.last_turn / latest        # newest fact gets +0.3
        repetition = 0.1 * min(fact.weight - 1, 3)     # said 4+ times caps at +0.3
        penalty = 0.0 if fact.active else 0.4          # old facts: kept, but ranked lower
        scored.append((a + recency + repetition - penalty, fact))
    scored.sort(key=lambda sf: sf[0], reverse=True)
    out.facts = [f for _, f in scored[:max_facts]]
    return out
