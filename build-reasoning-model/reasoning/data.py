"""
data.py: the homework and the answer key (Brick 2).

GSM8K is ~7,500 grade-school maths word problems. Each comes with a worked
solution written by a person, ending in a line like "#### 72".

We deliberately throw the worked solution away and keep only the number after
"####". The model must never see *how* to solve the problem, only find out
whether it got the right answer (Brick 0). That is what separates this from
ordinary fine-tuning.

Each training example becomes two things:
    prompt  ->  [system message with the format rules, user message with the question]
    answer  ->  "72"   (a bare string, so the grader can compare it exactly)
"""

from datasets import load_dataset

# The "form with labelled boxes". The grader (rewards.py) only ever looks inside
# <answer>...</answer>. The <reasoning> box is space for the model to think,
# and nobody grades it.
#
# Asking for this layout doesn't mean the model will follow it. Small models
# often ignore it at first, which is why rewards.py also pays for the layout itself.
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""


def extract_hash_answer(solution: str) -> str | None:
    """
    Pull the final answer out of a GSM8K worked solution.

        "Natalia sold 48/2 = 24 clips in May.\n... #### 72"  ->  "72"

    Returns None if there's no "####" (it never happens in GSM8K, but the type
    says so honestly).
    """
    if "####" not in solution:
        return None
    # Everything after "####" is the answer. GSM8K writes big numbers with
    # commas ("1,000"), so remove them; the grader compares bare digits.
    return solution.split("####")[1].strip().replace(",", "")


# A one-shot example: one finished exchange shown to the model before the real
# question. It's from the reference notebook (commented out there).
#
# Why we switch it ON: the cold-start problem. In testing, the 0.5B model
# ignored the system prompt completely. 0 of 24 attempts used the tags, so
# every attempt scored 0, every advantage was 0, and GRPO learned nothing.
# With this one example in front of it: 4/24 used the format, and all 4 test
# groups had score differences to learn from. Seeing the form filled in once
# is enough to get the ladder started.
#
# This doesn't break the "no worked solutions" rule: it shows the LAYOUT on a
# trivial non-GSM8K question, not how to solve maths word problems.
ONE_SHOT = [
    {"role": "user", "content": "What is the largest single-digit prime number?"},
    {
        "role": "assistant",
        "content": "<reasoning>\n9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.\n</reasoning>\n<answer>\n7\n</answer>",
    },
]


def build_prompt(question: str, one_shot: bool = True) -> list[dict]:
    """
    Wrap a question in chat format: a list of {role, content} messages.

        system:     the format rules
        user:       example question        (one_shot only)
        assistant:  example filled-in form  (one_shot only)
        user:       the real question

    The model was pretrained to expect this structure. Later, the tokenizer's
    chat template turns it into the exact special-token layout this model family
    uses (e.g. "<|im_start|>system ... <|im_end|>").
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(ONE_SHOT if one_shot else []),
        {"role": "user", "content": question},
    ]


def load_gsm8k(split: str = "train", one_shot: bool = True):
    """
    Download (first time only; cached afterwards) and prepare one GSM8K split.

    split: "train" (7,473 problems, used for learning) or
           "test"  (1,319 problems, kept unseen and used only to measure)

    Returns a Hugging Face Dataset whose rows have:
        question  original text (kept for logging)
        prompt    chat messages for the model
        answer    the bare answer string
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    # .map() runs the function on every row. Returning a key that already
    # exists ("answer") *overwrites* that column, so the worked solution is
    # replaced by the bare number and can't leak into training.
    return ds.map(
        lambda row: {
            "prompt": build_prompt(row["question"], one_shot),
            "answer": extract_hash_answer(row["answer"]),
        }
    )
