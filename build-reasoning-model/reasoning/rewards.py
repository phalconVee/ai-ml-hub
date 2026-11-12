"""
rewards.py: the grader (Brick 3).

A reward function takes the model's text and returns a number. That's the
*only* signal the model ever gets. Whatever we pay for is what it learns to
produce, so these functions quietly define the whole task.

We use five small graders rather than one, and add their scores together:

    grader                 max   pays for
    ---------------------  ----  ----------------------------------------------
    xmlcount_reward        0.5   each format tag present exactly once (partial credit)
    soft_format_reward     0.5   <reasoning>…</reasoning> then <answer>…</answer>, loosely
    strict_format_reward   0.5   exact layout, newlines and all
    int_reward             0.5   the answer box holds a plain whole number
    correctness_reward     2.0   the answer box matches the answer key
                           ----
                           4.0   best possible total

Why so many, and why so much format credit? Because of how GRPO learns: it
compares attempts within a group (Brick 0). If every attempt scores 0, the
attempts are all "average" and nothing is learned. A beginner model rarely gets
the maths right *and* in the right box, so correctness alone is usually 0
across the board. The format graders are a ladder of easier wins. The model
first learns to fill the form in; then the answers inside become gradable, and
correctness takes over (it's worth 2.0 for a reason).

What is NOT graded: the reasoning. Not one function below reads it for
quality. Good reasoning gets rewarded only indirectly, because it tends to
produce correct answers. That's the emergence from Brick 0, and it's also a
loophole: nonsense reasoning that stumbles onto the right number gets full
marks (reward hacking).

Signature used throughout:
    fn(completions: list[str], answer: str) -> list[float]
completions = the group of attempts for ONE question; answer = its answer key.
"""

import re


def extract_xml_answer(text: str) -> str:
    """
    Pull out what's inside the LAST <answer>…</answer> box.

        "<reasoning>\n2+2\n</reasoning>\n<answer>\n4\n</answer>"  ->  "4"

    If the tags are missing, split() doesn't fail: it just returns the whole
    text. So a tagless reply gets compared as one long string and (correctly)
    doesn't match "4".
    """
    answer = text.split("<answer>")[-1]     # everything after the last <answer>
    answer = answer.split("</answer>")[0]   # everything before the next </answer>
    return answer.strip()


def normalize_number(text: str) -> str:
    """
    Make harmless formatting differences disappear before comparing:
        "$1,000."  ->  "1000"
        "72.0"     ->  "72"

    The reference compares raw strings, so "$18" != "18" scores 0. That punishes
    a correct answer for cosmetics and makes the reward noisier. Keep this
    narrow: it strips symbols, it doesn't do maths.
    """
    t = text.strip().replace(",", "").replace("$", "").rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", t):      # "72.0" -> "72"
        t = t.split(".")[0]
    return t


# --------------------------------------------------------------- CORRECTNESS


def correctness_reward(completions: list[str], answer: str) -> list[float]:
    """2.0 if the answer box matches the key, else 0.0. The reward that matters most."""
    return [
        2.0 if normalize_number(extract_xml_answer(c)) == normalize_number(answer) else 0.0
        for c in completions
    ]


def int_reward(completions: list[str], answer: str) -> list[float]:
    """
    0.5 if the answer box contains only digits, e.g. "18", not "18 dollars".

    A stepping stone: it teaches "put just a number in the box" even before
    the number is right. Every GSM8K answer is a whole number, so this is
    always the right shape. (isdigit() rejects "-5" and "3.5", which is fine here.)
    """
    return [0.5 if normalize_number(extract_xml_answer(c)).isdigit() else 0.0 for c in completions]


# -------------------------------------------------------------------- FORMAT

# re.DOTALL makes "." also match newlines. Without it, ".*?" stops at the end
# of the line, so any reasoning longer than one line fails the check. The
# reference omits DOTALL. That's a real bug there: its format rewards almost
# never fire on multi-line reasoning, which is exactly what we want.

STRICT_PATTERN = re.compile(
    r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n?$",
    #  ^ start   ^ tags each on their own line              ^ optional final newline, then end
    # The reference requires a trailing "\n", but the model's reply ends at the
    # end-of-text token and decoding strips it, so a perfect reply would never pass.
    re.DOTALL,
)

SOFT_PATTERN = re.compile(
    r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>",
    re.DOTALL,
)


def strict_format_reward(completions: list[str], answer: str) -> list[float]:
    """0.5 for exactly the requested layout (each tag on its own line, nothing before or after)."""
    return [0.5 if STRICT_PATTERN.match(c.strip()) else 0.0 for c in completions]


def soft_format_reward(completions: list[str], answer: str) -> list[float]:
    """0.5 for the right tags in the right order, whatever the whitespace (re.search = anywhere)."""
    return [0.5 if SOFT_PATTERN.search(c) else 0.0 for c in completions]


def count_xml(text: str) -> float:
    """
    Partial credit, 0.125 per tag that appears exactly once, with a small
    penalty for text trailing after </answer>.

    This is the gentlest grader: even a half-formatted attempt scores
    something. That gives GRPO a difference to work with on day one, when
    nobody in the group gets the full format right.
    """
    score = 0.0
    if text.count("<reasoning>\n") == 1:
        score += 0.125
    if text.count("\n</reasoning>\n") == 1:
        score += 0.125
    if text.count("\n<answer>\n") == 1:
        score += 0.125
    if text.count("\n</answer>") == 1:
        score += 0.125
        # Rambling after the answer box costs 0.001 per character. The answer
        # should be the last thing said.
        trailing = text.split("\n</answer>")[-1]
        score -= len(trailing.strip()) * 0.001
    return score


def xmlcount_reward(completions: list[str], answer: str) -> list[float]:
    return [count_xml(c) for c in completions]


# ----------------------------------------------------------------- ALL TOGETHER

# Order matters only for logging (columns print in this order).
REWARD_FUNCS = {
    "xmlcount": xmlcount_reward,
    "soft_fmt": soft_format_reward,
    "strict_fmt": strict_format_reward,
    "int": int_reward,
    "correct": correctness_reward,
}


def score(completions: list[str], answer: str) -> dict[str, list[float]]:
    """
    Run every grader on a group of attempts.

    Returns {grader_name: [score for attempt 1, attempt 2, …]}.
    The training loop adds these up into one total per attempt.
    """
    return {name: fn(completions, answer) for name, fn in REWARD_FUNCS.items()}
