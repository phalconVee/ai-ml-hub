"""The decision layer, the drawing, the log, and the whole pipeline in demo mode."""

from PIL import Image

from sight.decisions import Thresholds, route
from sight.demo import DemoJevClient
from sight.draw import COLOURS, draw_boxes, to_pixels
from sight.jev import parse_response
from sight.log import log_review, read_log
from sight.pipeline import analyze
from sight.questions import Plan, build_questions, parse_labels

T = Thresholds(choice_accept=0.8, noul_yes=0.7, noul_no=0.3, box_accept=0.6, quality_min=1.0)
PLAN = Plan(labels=parse_labels("mug, phone"), objects=["mug"], checks=["Is a person present?"])
QS = build_questions(PLAN)


def answers(choice_p=0.9, present=0.95, box_conf=0.9, person=0.02, quality=1.8):
    """A raw response with knobs for each number the thresholds look at."""
    return parse_response({"model": "m", "usage": {}, "answers": {
        "classify": {"type": "choice", "choice": "mug", "confidence": 0.8,
                     "probabilities": {"mug": choice_p, "phone": round(1 - choice_p, 3)}},
        "present_mug": {"type": "noul", "noul": present},
        "box_mug": {"type": "box", "confidence": box_conf,
                    "box": {"left": 0.1, "top": 0.2, "right": 0.5, "bottom": 0.9}},
        "check_0_is_a_person_present": {"type": "noul", "noul": person},
        "quality": {"type": "score", "score": quality, "confidence": 0.7,
                    "legend": {"0": "dark", "1": "ok", "2": "clear"},
                    "probabilities": {"0": 0.0, "1": 0.2, "2": 0.8}},
    }})


def by_id(decisions):
    return {d.question_id: d for d in decisions}


def test_confident_frame_is_fully_automatic():
    d = by_id(route(QS, answers(), T))
    assert {x.status for x in d.values()} == {"auto"}
    assert d["classify"].answer == "mug"
    assert d["check_0_is_a_person_present"].answer == "no"     # 0.02 <= 0.3
    assert d["box_mug"].box["left"] == 0.1


def test_choice_below_threshold_goes_to_review():
    assert by_id(route(QS, answers(choice_p=0.7), T))["classify"].status == "review"
    # ...and the same answer is accepted by someone more tolerant of risk.
    lenient = Thresholds(choice_accept=0.6)
    assert by_id(route(QS, answers(choice_p=0.7), lenient))["classify"].status == "auto"


def test_noul_has_three_outcomes():
    for p, verdict, status in [(0.9, "yes", "auto"), (0.1, "no", "auto"),
                               (0.5, "unsure", "review")]:
        d = by_id(route(QS, answers(person=p), T))["check_0_is_a_person_present"]
        assert (d.answer, d.status) == (verdict, status)


def test_box_for_absent_object_is_skipped_not_drawn():
    d = by_id(route(QS, answers(present=0.05), T))
    assert d["present_mug"].answer == "no"
    assert d["box_mug"].status == "skipped"


def test_box_review_when_presence_unsure_or_confidence_low_or_null():
    assert by_id(route(QS, answers(present=0.5), T))["box_mug"].status == "review"
    assert by_id(route(QS, answers(box_conf=0.4), T))["box_mug"].status == "review"
    assert by_id(route(QS, answers(box_conf=None), T))["box_mug"].status == "review"


def test_poor_image_quality_demotes_everything_to_review():
    d = by_id(route(QS, answers(quality=0.4), T))
    assert d["quality"].status == "review"
    assert d["classify"].status == "review" and "low image quality" in d["classify"].reason
    # Skipped stays skipped: there is nothing to review about an ignored box.
    d2 = by_id(route(QS, answers(quality=0.4, present=0.05), T))
    assert d2["box_mug"].status == "skipped"


def test_missing_answer_is_reviewed_not_dropped():
    r = answers()
    del r.answers["classify"]
    assert by_id(route(QS, r, T))["classify"].status == "review"


def test_box_fractions_become_pixels_from_top_left():
    # Docs example: left .44 right .84 top .25 bottom .82 on a 1000x500 image.
    assert to_pixels({"left": .44, "right": .84, "top": .25, "bottom": .82}, 1000, 500) == \
        (440, 125, 840, 410)
    assert to_pixels({"left": -0.1, "right": 1.2, "top": 0, "bottom": 1}, 100, 100) == \
        (0, 0, 100, 100)                                     # clamped inside the image


def test_draw_boxes_colours_by_status_and_hides_skipped():
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    ds = route(QS, answers(), T)
    out = draw_boxes(img, ds)
    assert out is not img and img.getpixel((40, 150)) == (255, 255, 255)   # original untouched
    assert out.getpixel((40, 150)) == COLOURS["auto"]       # left edge of the box at x=0.1
    hidden = draw_boxes(img, route(QS, answers(present=0.05), T))
    assert hidden.getpixel((40, 150)) == (255, 255, 255)    # skipped box not drawn


def test_pipeline_end_to_end_in_demo_mode(tmp_path):
    log = tmp_path / "log.jsonl"
    plan = Plan(labels=parse_labels("mug, phone, laptop, book, nothing"),
                objects=["mug", "laptop", "book", "phone"], checks=["Is a person present?"])
    client = DemoJevClient()
    from sight.frames import make_sample_image
    a = analyze(client, make_sample_image(), plan, Thresholds(), log_path=log)

    assert len(client.calls) == 1                            # every question, ONE request
    assert len(client.calls[0]["questions"]) == 1 + 2 * 4 + 1 + 1
    d = by_id(a.decisions)
    assert d["classify"].answer == "laptop" and d["classify"].status == "auto"
    assert d["box_book"].status == "review"                  # low box confidence (0.48)
    assert d["box_phone"].status == "skipped"                # phone isn't on the desk
    assert d["check_0_is_a_person_present"].answer == "no"

    rows = read_log(log)
    assert len(rows) == len(a.decisions) and rows[0]["frame"] == a.frame_id


def test_review_verdicts_are_appended_not_overwritten(tmp_path):
    log = tmp_path / "log.jsonl"
    log_review("f1", "classify", "label set", "mug", "mug", path=log)
    log_review("f1", "box_book", "book", "found", "wrong place", path=log)
    newest, oldest = read_log(log)
    assert oldest["status"] == "confirmed" and newest["status"] == "corrected"
    assert newest["reason"] == "Jev said 'found'"
