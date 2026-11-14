"""The three agents and the routing, with Claude replaced by a script.

`FakeClient` stands in for `client.beta.messages.create(...)`: each call pops
the next canned response and records the request, so we can assert both what
the crew *did* and what it *sent*.
"""

from types import SimpleNamespace as NS

from counsel import NOT_LEGAL_ADVICE, ClauseIndex, chunk_document, run_crew
from counsel.crew import needs_research
from sample_docs.make_sample import build_sample_pdf

LATE = "sample_lease.pdf#p1-c4"      # 3. Late Payment
EARLY = "sample_lease.pdf#p2-c6"     # 5. Early Termination
URL = "https://courts.example.gov/late-fees"


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        # Snapshot `messages`: the agents keep appending to the same list.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def msg(stop_reason, *blocks):
    return NS(stop_reason=stop_reason, content=list(blocks))


def search(query, id="tu_1"):
    return NS(type="tool_use", id=id, name="search_documents", input={"query": query})


def report(ids, coverage="full", questions=(), summary="Clause 3 sets a 10% late fee."):
    return NS(type="tool_use", id="tu_r", name="file_report",
              input={"clause_ids": list(ids), "coverage": coverage,
                     "legal_questions": list(questions), "summary": summary})


def text(t, citations=()):
    return NS(type="text", text=t, citations=list(citations) or None)


def cite(source, quote, title="t"):
    return NS(type="search_result_location", source=source, cited_text=quote, title=title)


def web_turn():
    """What a web-search response looks like: query, results, cited write-up."""
    return msg("end_turn",
               NS(type="server_tool_use", name="web_search", input={"query": "California late fee limit"}),
               NS(type="web_search_tool_result", content=[NS(url=URL, title="Late fees | Courts")]),
               text("Courts look at whether a late fee is a reasonable estimate of damages.",
                    [NS(url=URL, title="Late fees | Courts", cited_text="a reasonable estimate of damages")]))


def index():
    idx = ClauseIndex()
    idx.add(chunk_document(build_sample_pdf(), "sample_lease.pdf"))
    return idx


# ── routing ─────────────────────────────────────────────────────────────────

def test_full_coverage_skips_the_web():
    client = FakeClient([
        msg("tool_use", search("late fee overdue")),                       # finder: search
        msg("tool_use", report([LATE])),                                   # finder: report
        msg("end_turn", text("The late fee is 10% of the overdue amount.",  # drafter
                             [cite(LATE, "a late fee of ten percent (10%)")])),
    ])
    r = run_crew("What is the late fee?", index(), client)

    assert r.route == ["Clause Finder", "Drafter"]
    assert len(client.requests) == 3
    # The finder's search result went back as a tool_result tied to its id.
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "tu_1" and "ten percent" in tool_result["content"]
    # The drafter got the verified clause as a citable search_result block.
    drafter_content = client.requests[2]["messages"][0]["content"]
    assert drafter_content[0]["type"] == "search_result" and drafter_content[0]["source"] == LATE
    assert drafter_content[0]["citations"] == {"enabled": True}
    assert "tools" not in client.requests[2]                              # drafter has no tools
    # Answer: numbered citation + our disclaimer on top.
    md = r.markdown()
    assert md.startswith(f"> {NOT_LEGAL_ADVICE}")
    assert md.endswith("overdue amount. [1]")
    assert r.draft.citations[0].kind == "clause"


def test_law_question_triggers_research_and_web_is_citable():
    client = FakeClient([
        msg("tool_use", report([LATE], "partial", ["Are 10% late fees enforceable in California?"])),
        web_turn(),                                                         # researcher
        msg("end_turn",
            text("Your lease sets a 10% fee.", [cite(LATE, "ten percent (10%)")]),
            text(" Courts ask whether it is reasonable.", [cite(URL, "a reasonable estimate of damages")])),
    ])
    r = run_crew("Is my late fee legal?", index(), client, jurisdiction="California")

    assert r.route == ["Clause Finder", "Legal Researcher", "Drafter"]
    researcher_req = client.requests[1]
    assert researcher_req["tools"][0]["type"] == "web_search_20260209"
    assert "California" in researcher_req["messages"][0]["content"]
    # Web passages reach the drafter as search_result blocks with the URL as source.
    blocks = client.requests[2]["messages"][0]["content"]
    web = [b for b in blocks if b["type"] == "search_result" and b["source"] == URL]
    assert web and web[0]["content"][0]["text"] == "a reasonable estimate of damages"
    assert [c.kind for c in r.draft.citations] == ["clause", "web"]
    assert r.research.queries == ["California late fee limit"]


def test_allow_web_false_never_researches():
    client = FakeClient([
        msg("tool_use", report([], "none")),
        msg("end_turn", text("Your documents don't address this.")),
    ])
    r = run_crew("Can I sublet?", index(), client, allow_web=False)
    assert r.route == ["Clause Finder", "Drafter"]


def test_no_documents_goes_straight_to_research():
    client = FakeClient([web_turn(), msg("end_turn", text("Generally, ..."))])
    r = run_crew("What is a reasonable late fee?", ClauseIndex(), client)
    assert r.route == ["Legal Researcher", "Drafter"]


def test_needs_research_rule():
    from counsel.clause_finder import ClauseReport
    assert not needs_research(ClauseReport([], "full", [], ""))
    assert needs_research(ClauseReport([], "full", ["Is X legal?"], ""))
    assert needs_research(ClauseReport([], "partial", [], ""))


# ── verification and failure paths ──────────────────────────────────────────

def test_invented_clause_ids_are_dropped():
    client = FakeClient([
        # Only LATE was shown by the seed search; EARLY never was, and "made-up" doesn't exist.
        msg("tool_use", report([LATE, "made-up#p9-c99"], "full")),
        msg("end_turn", text("ok")),
    ])
    r = run_crew("late fee overdue rent", index(), client, allow_web=False)
    assert [c.id for c in r.report.clauses] == [LATE]
    assert r.report.dropped_ids == ["made-up#p9-c99"]


def test_report_with_no_real_clauses_is_downgraded_to_none():
    client = FakeClient([
        msg("tool_use", report(["ghost#p1-c1"], "full")),
        web_turn(),                                  # downgraded → research runs
        msg("end_turn", text("...")),
    ])
    r = run_crew("late fee", index(), client)
    assert r.report.coverage == "none" and "Legal Researcher" in r.route


def test_finder_that_never_reports_falls_back_to_seed_hits():
    client = FakeClient([
        msg("end_turn", text("I think it's clause 3.")),   # forgot to call file_report
        web_turn(),
        msg("end_turn", text("...")),
    ])
    r = run_crew("late fee overdue rent", index(), client)
    assert r.report.coverage == "partial" and r.report.clauses[0].id == LATE


def test_refusal_still_carries_the_notice():
    client = FakeClient([msg("tool_use", report([LATE])), msg("refusal")])
    r = run_crew("late fee", index(), client)
    assert "declined" in r.markdown() and NOT_LEGAL_ADVICE in r.markdown()


def test_every_request_uses_model_effort_and_fallback():
    client = FakeClient([msg("tool_use", report([LATE])), msg("end_turn", text("ok"))])
    run_crew("late fee", index(), client)
    for req in client.requests:
        assert req["model"] == "claude-opus-5-5"
        assert req["output_config"] == {"effort": "medium"}
        assert req["fallbacks"] == "default"
        assert req["betas"] == ["server-side-fallback-2026-07-01"]
