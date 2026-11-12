"""
Sanity checks for the grader. Run with:  uv run pytest

If a reward function is wrong, training silently optimises the wrong thing,
and you won't find out until hours later. These tests take milliseconds.
Each one is also a worked example of what a grader pays for.
"""

from reasoning.data import extract_hash_answer
from reasoning.rewards import extract_xml_answer, score

PERFECT = "<reasoning>\n3 pens cost $2, so 12 pens is 4 lots.\n4 x 2 = 8\n</reasoning>\n<answer>\n8\n</answer>"
WRONG_NUMBER = PERFECT.replace("\n8\n</answer>", "\n24\n</answer>")
NO_TAGS = "12 pens cost 8 dollars."
WORDY_ANSWER = PERFECT.replace("\n8\n</answer>", "\n$8 dollars\n</answer>")


def test_answer_key_extraction():
    assert extract_hash_answer("lots of working... #### 72") == "72"
    assert extract_hash_answer("#### 1,000") == "1000"


def test_extract_answer_box():
    assert extract_xml_answer(PERFECT) == "8"
    # No tags: returns the whole text, which won't equal the key.
    assert extract_xml_answer(NO_TAGS) == NO_TAGS


def test_perfect_attempt_gets_full_marks():
    s = score([PERFECT], "8")
    assert s == {"xmlcount": [0.5], "soft_fmt": [0.5], "strict_fmt": [0.5], "int": [0.5], "correct": [2.0]}


def test_multiline_reasoning_still_counts_as_formatted():
    # The reference's regexes (no re.DOTALL) would give 0 here: the reasoning spans two lines.
    assert score([PERFECT], "8")["soft_fmt"] == [0.5]


def test_right_format_wrong_number():
    s = score([WRONG_NUMBER], "8")
    assert s["correct"] == [0.0]
    assert s["strict_fmt"] == [0.5]   # format credit is independent of correctness


def test_no_tags_scores_nothing():
    s = score([NO_TAGS], "8")
    assert sum(sum(v) for v in s.values()) == 0.0


def test_dollar_sign_is_forgiven_but_words_are_not():
    s = score([WORDY_ANSWER], "8")
    assert s["correct"] == [0.0]      # "8 dollars" != "8"
    assert s["int"] == [0.0]
    s = score([PERFECT.replace("\n8\n</answer>", "\n$8\n</answer>")], "8")
    assert s["correct"] == [2.0]      # "$8" -> "8"


def test_group_gets_one_score_per_attempt():
    s = score([PERFECT, WRONG_NUMBER, NO_TAGS], "8")
    assert s["correct"] == [2.0, 0.0, 0.0]
