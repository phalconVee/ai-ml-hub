# Brick 0: The Big Picture

## In my own words

**1. How is this different from regular chatbot training?**
There's no worked solution for the model to copy. It isn't told *how* to solve a problem, only whether it *got* the right answer. Regular training grades the process. This grades only the outcome.

**2. Why try each problem several times?**
Trying once doesn't tell the model much. Trying several times gives it a basis to compare: what to keep and what to push away. It's usually about 6 attempts, each running separately.

**3. What is the model rewarded for?**
Its answer: whether the answer is correct and whether it's in the expected layout. It isn't graded on the reasoning.

## Gaps found and fixed

- **"Thrown into the dark."** Not quite. The model already knows language and maths from pretraining, and it has an answer key. What it lacks is a demonstration of the *method*.
- **"Learning from past mistakes between attempts."** Wrong. The attempts happen side by side and don't see each other. Learning happens only *after* the whole group is scored, by comparing each attempt to the group average. That's the "G" (Group) in GRPO.
- **"The model observes that reasoning helps."** The model doesn't observe anything. The training maths makes the patterns from high-scoring attempts more likely. Reasoning is rewarded only *indirectly*, because careful reasoning tends to produce correct answers. It *emerges*.

## Fine print

- **6 is a setting, not a rule.** It's `num_generations = 6` in the training config, and we can change it.
- **The model doesn't "decide" what to keep.** The training update does that for it.

## Loose thread for later

Nonsense reasoning that stumbles onto the right answer still gets the full reward. This is **reward hacking** (comes back in Brick 3).
