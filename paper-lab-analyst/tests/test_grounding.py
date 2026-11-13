"""The quote checker: verified / wrong page / not found, and PDF-noise tolerance."""

import copy

from conftest import REPORT
from paperlab.grounding import check_quote, ground_report, normalise
from paperlab.pdf_extract import Paper, Page, extract_paper


def test_normalise_undoes_pdf_artifacts():
    assert normalise("phos-\nphatase  ﬁeld “x”") == 'phosphatase field "x"'


def toy_paper():
    return Paper("t", [Page(1, "Alpha beta gamma delta epsilon."),
                       Page(2, "The yield rose by 12 percent in treated plots overall.")],
                 [], total_pages=2)


def test_quote_statuses():
    paper = toy_paper()
    assert check_quote("yield rose by 12 percent", 2, paper).status == "verified"
    wrong = check_quote("yield rose by 12 percent", 1, paper)
    assert (wrong.status, wrong.found_page) == ("wrong_page", 2)
    assert check_quote("yield fell by 40 percent in winter", 2, paper).status == "not_found"


def test_near_verbatim_quote_still_counts():
    # One word differs out of 9 (89% coverage ≥ 0.8 threshold): a typo shouldn't fail it.
    assert check_quote("The yield rose by 12 percent in treated fields", 2, toy_paper()).status == "verified"


def test_ground_report_annotates_and_drops_unknown_figures(sample_pdf):
    paper = extract_paper(sample_pdf)
    report = copy.deepcopy(REPORT)
    summary = ground_report(report, paper)
    assert [f["grounding"]["status"] for f in report["key_findings"]] == ["verified", "wrong_page"]
    assert report["key_findings"][1]["grounding"]["found_page"] == 2
    assert report["limitations"][0]["grounding"]["status"] == "not_found"
    assert (summary["verified"], summary["wrong_page"], summary["not_found"]) == (1, 1, 1)
    assert report["key_findings"][0]["figure_ids"] == ["F2"]           # F9 removed
    assert [i["figure_id"] for i in report["figure_insights"]] == ["F2"]  # F7 removed
    assert sorted(summary["dropped_figure_ids"]) == ["F7", "F9"]
