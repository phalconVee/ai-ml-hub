"""
reasoning: a small reasoning model trained with GRPO, written to be read.

Read the files in this order. Each one builds on the one before:

    config.py    every knob in one place
    data.py      the homework (GSM8K) and the answer key
    rewards.py   the grader: text in, score out
    model.py     the student: a pretrained model plus a small trainable LoRA adapter
    grpo.py      the core: sample a group, score it, compare, nudge
    train.py     the loop that repeats grpo.py thousands of times
    evaluate.py  did it work? Before vs after
"""
