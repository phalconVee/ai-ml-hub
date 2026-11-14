"""The decision layer: typed answers + YOUR thresholds -> what happens next.

This is the point of a decision model. Jev returns calibrated numbers; this
file, in plain Python, decides which answers the app acts on by itself and
which go to a person. The model never decides its own trustworthiness.

Every answer ends up in one of three states:

    auto     confident enough to act on with no human
    review   in the "not sure" band: goes to the human review queue
    skipped  deliberately ignored (a box for an object Jev says isn't there)

What gets thresholded, per type (following Jev's Confidence page):

    Choice  the probability of the winning label (a readable risk statement)
    Noul    the value itself: >= yes -> "yes", <= no -> "no", between -> review
    Box     its presence Noul first, then the box's own confidence
    Score   the expected level; a poor-quality frame demotes every "auto" to review
"""

from dataclasses import dataclass
from typing import Literal

from sight import config
from sight.jev import BoxAnswer, ChoiceAnswer, JevResult, NoulAnswer, ScoreAnswer
from sight.questions import Question

Status = Literal["auto", "review", "skipped"]


@dataclass
class Thresholds:
    choice_accept: float = config.CHOICE_ACCEPT
    noul_yes: float = config.NOUL_YES
    noul_no: float = config.NOUL_NO
    box_accept: float = config.BOX_ACCEPT
    quality_min: float = config.QUALITY_MIN_LEVEL


@dataclass
class Decision:
    question_id: str
    kind: str
    role: str
    subject: str
    answer: str                 # human-readable answer ("mug", "yes", "found", ...)
    value: float | None         # the number that was compared to a threshold
    status: Status
    reason: str                 # one line a reviewer can read
    box: dict | None = None     # edges as fractions, for drawing


def noul_verdict(p: float, t: Thresholds) -> tuple[str, Status]:
    """The three-way split for a yes/no probability."""
    if p >= t.noul_yes:
        return "yes", "auto"
    if p <= t.noul_no:
        return "no", "auto"
    return "unsure", "review"


def route(questions: list[Question], result: JevResult, t: Thresholds) -> list[Decision]:
    """Turn one request's answers into decisions, in question order."""
    answers = result.answers
    decisions: list[Decision] = []

    for q in questions:
        a = answers.get(q.id)
        if a is None:  # defensive: the API should answer every question
            decisions.append(Decision(q.id, q.kind, q.role, q.subject, "no answer",
                                      None, "review", "Jev returned no answer"))
            continue

        if isinstance(a, ChoiceAnswer):
            p = a.top_probability
            ok = p >= t.choice_accept
            decisions.append(Decision(
                q.id, q.kind, q.role, q.subject, a.choice, p,
                "auto" if ok else "review",
                f"P({a.choice}) = {p:.2f} {'≥' if ok else '<'} {t.choice_accept:.2f}"))

        elif isinstance(a, NoulAnswer):
            verdict, status = noul_verdict(a.noul, t)
            decisions.append(Decision(
                q.id, q.kind, q.role, q.subject, verdict, a.noul, status,
                f"P(yes) = {a.noul:.2f} (yes ≥ {t.noul_yes:.2f}, no ≤ {t.noul_no:.2f})"))

        elif isinstance(a, BoxAnswer):
            # Its partner Noul shares the slug: box_mug <-> present_mug.
            presence = answers.get("present_" + q.id.removeprefix("box_"))
            decisions.append(_route_box(q, a, presence, t))

        elif isinstance(a, ScoreAnswer):
            level = a.legend.get(round(a.score), f"level {a.score:.1f}")
            decisions.append(Decision(
                q.id, q.kind, q.role, q.subject, level, a.score, "auto",
                f"score {a.score:.2f} on 0..{max(a.legend or {0: ''})}, "
                f"confidence {a.confidence:.2f}"))

    # A poor frame (dark, blurred, covered lens) makes every other answer less
    # trustworthy, however confident each looks on its own. Demote them all.
    quality = next((d for d in decisions if d.role == "quality"), None)
    if quality and quality.value is not None and quality.value < t.quality_min:
        quality.status = "review"
        quality.reason += f" < {t.quality_min:.1f}: frame too poor to trust"
        for d in decisions:
            if d.status == "auto":
                d.status = "review"
                d.reason += " · demoted: low image quality"
    return decisions


def _route_box(q: Question, box: BoxAnswer, presence, t: Thresholds) -> Decision:
    """A Box always returns four edges, even for something that isn't there,
    so it is only as trustworthy as the presence Noul asked beside it."""
    edges = {"left": box.left, "top": box.top, "right": box.right, "bottom": box.bottom}
    conf = box.confidence

    if isinstance(presence, NoulAnswer):
        verdict, _ = noul_verdict(presence.noul, t)
        if verdict == "no":
            return Decision(q.id, q.kind, q.role, q.subject, "not present", conf, "skipped",
                            f"presence P = {presence.noul:.2f}: box ignored", edges)
        if verdict == "unsure":
            return Decision(q.id, q.kind, q.role, q.subject, "maybe", conf, "review",
                            f"presence P = {presence.noul:.2f}: unsure it's there", edges)

    if conf is None:
        return Decision(q.id, q.kind, q.role, q.subject, "found", None, "review",
                        "box has no confidence value: a person checks it", edges)
    ok = conf >= t.box_accept
    return Decision(q.id, q.kind, q.role, q.subject, "found", conf,
                    "auto" if ok else "review",
                    f"box confidence {conf:.2f} {'≥' if ok else '<'} {t.box_accept:.2f}", edges)
