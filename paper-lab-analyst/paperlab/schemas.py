"""JSON Schemas for every structured answer Claude gives us.

With structured outputs (`output_config.format`) the API guarantees the reply
matches the schema, so no agent ever has to parse free-form prose. Rules the
API needs: every object lists all its keys in `required` and sets
`additionalProperties: false`.

The `obj` / `arr` / `STR` helpers just keep the schemas short enough to read.
"""

STR = {"type": "string"}
INT = {"type": "integer"}


def obj(**props) -> dict:
    """An object whose listed properties are all required, nothing extra allowed."""
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def arr(items: dict) -> dict:
    return {"type": "array", "items": items}


def enum(*values: str) -> dict:
    return {"type": "string", "enum": list(values)}


# A short verbatim quote from the paper plus the page it's on. This is the unit
# of evidence: grounding.py later checks that the quote really is in the PDF.
EVIDENCE = obj(quote=STR, page=INT)

# ── Figure reader (vision.py): one per figure ────────────────────────────────
FIGURE_NOTE = obj(
    figure_type=STR,             # "bar chart", "micrograph", "schematic", ...
    what_it_shows=STR,           # axes, groups, panels — a literal description
    key_takeaway=STR,            # the one thing a reader should conclude
    quantitative_details=arr(STR),  # numbers read off the figure (values, n, p, error bars)
    caveats=arr(STR),            # what the figure doesn't show / could mislead
)

# ── Methods analyst (analysts.py) ────────────────────────────────────────────
METHODS_NOTE = obj(
    research_question=STR,
    study_type=STR,              # "randomised controlled trial", "in vitro", "benchmark", ...
    design_summary=STR,
    data_and_samples=STR,        # what was studied and how many (n)
    techniques=arr(STR),
    statistics=STR,
    reproducibility_notes=STR,   # code/data availability, missing details
    evidence=arr(EVIDENCE),
)

# ── Findings analyst (analysts.py) ───────────────────────────────────────────
FINDINGS_NOTE = obj(
    findings=arr(obj(
        claim=STR,
        evidence=EVIDENCE,
        figure_ids=arr(STR),     # our ids ("F2"), so claims link to figures
        strength=enum("strong", "moderate", "weak"),
    )),
    stated_limitations=arr(EVIDENCE),  # limitations the authors admit, quoted
)

# ── Final report, written by the reviewer agent (reviewer.py) ────────────────
REPORT = obj(
    title=STR,
    field=STR,
    one_line_summary=STR,
    executive_summary=STR,
    research_question=STR,
    methods=obj(study_design=STR, data=STR, techniques=arr(STR), statistics=STR),
    key_findings=arr(obj(
        claim=STR,
        evidence=EVIDENCE,
        figure_ids=arr(STR),
        confidence=enum("high", "medium", "low"),
    )),
    figure_insights=arr(obj(figure_id=STR, insight=STR)),
    limitations=arr(obj(
        limitation=STR,
        source=enum("stated_by_authors", "reviewer_inferred"),
        evidence=EVIDENCE,       # for inferred ones: the passage that prompted it
    )),
    open_questions=arr(STR),
    practical_implications=arr(STR),
    reviewer_notes=arr(STR),     # what the reviewer changed vs. the analysts' drafts
)
