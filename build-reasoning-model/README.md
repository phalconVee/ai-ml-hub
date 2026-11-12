# Build a Reasoning Model, Brick by Brick

A small reasoning model, built to learn how it works. It uses **GRPO** (the reinforcement-learning method behind DeepSeek-R1) to teach a language model to think step by step before answering grade-school maths problems (GSM8K).

The GRPO algorithm is written **from scratch** in about 100 lines ([`reasoning/grpo.py`](reasoning/grpo.py)) instead of hidden inside a library trainer, so every step can be read line by line. Every file is heavily commented to be walked through.

## The idea in one paragraph

Give the model a maths question. Let it try **6 times**, rolling dice so each attempt differs. **Grade** each attempt only on its final answer and its layout, never on its reasoning. Compare each attempt to the **group's average**. Nudge the model toward the above-average attempts and away from the below-average ones. Repeat thousands of times. Nobody shows the model how to reason; it reasons because reasoning earns more reward.

## Code map: read in this order

| File | Role | Brick |
|------|------|-------|
| [`reasoning/config.py`](reasoning/config.py) | Every knob in one place | – |
| [`reasoning/data.py`](reasoning/data.py) | GSM8K, the answer key, the `<reasoning>/<answer>` form, the one-shot example | 2 |
| [`reasoning/rewards.py`](reasoning/rewards.py) | The five graders: text in, score out | 3 |
| [`reasoning/model.py`](reasoning/model.py) | The pretrained model plus a small trainable LoRA adapter | 1, 5 |
| [`reasoning/grpo.py`](reasoning/grpo.py) | **The core:** sample a group, compare, compute the loss | 4 |
| [`reasoning/train.py`](reasoning/train.py) | The loop that repeats GRPO steps, with logs and checkpoints | 6 |
| [`reasoning/evaluate.py`](reasoning/evaluate.py) | Before vs after on unseen test questions, plus 9.11 vs 9.9 | 7 |
| [`tests/test_rewards.py`](tests/test_rewards.py) | Worked examples of what each grader pays for | 3 |
| [`bricks/01_base_model.py`](bricks/01_base_model.py) | Exploration: tokens, probabilities, temperature | 1 |

## Run it

Setup (once). Needs [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Test the graders (takes seconds):

```bash
uv run pytest
```

Measure the original model before any training:

```bash
uv run python -m reasoning.evaluate --n 50
```

Smoke test, 5 steps:

```bash
uv run python -m reasoning.train --max_steps 5 --output_dir outputs/smoke
```

Full training run:

```bash
uv run python -m reasoning.train --output_dir outputs/run
```

Measure after training:

```bash
uv run python -m reasoning.evaluate --adapter outputs/run/final --n 50
```

Any setting in `config.py` can be overridden on the command line: `--num_generations 4`, `--learning_rate 2e-5`, `--model_name Qwen/Qwen2.5-1.5B-Instruct`, and so on.

### Hardware reality

| Where | Speed | Full 250 steps |
|-------|-------|----------------|
| Apple M1, 16 GB (Qwen 0.5B) | ~45–180 s/step | ~6–10 hours |
| Free Colab T4 GPU | much faster | worth it for real runs |

On a Mac, a shorter run still shows the trend: `--max_steps 60 --num_generations 4 --max_completion_length 150`.

## Where this differs from the reference notebook, and why

| Change | Why |
|--------|-----|
| Qwen2.5-0.5B instead of Llama 3.1 8B | Fits on a Mac; works the same way. Swap with `--model_name`. |
| No Unsloth or vLLM | They need an NVIDIA GPU, and they hide the algorithm. Plain PyTorch + PEFT runs anywhere. |
| GRPO written by hand, not TRL's `GRPOTrainer` | So it can be read line by line. |
| One-shot example **on** | Without it, the small model used the format 0/24 times, so every reward was 0 and nothing was learned (the cold-start problem). With it, 4/4 groups had signal. |
| `re.DOTALL` in the format regexes | Bug fix: the reference's patterns fail on any reasoning longer than one line. |
| Answers normalised (`$18` → `18`, `1,000` → `1000`) | A correct answer shouldn't score 0 over cosmetics. |
| Learning rate 1e-5 (ref 5e-6) | Small LoRA, small model, few steps: 5e-6 barely moves. |

## Progress

- [x] Brick 0: The big picture → [notes](notes/00-big-picture.md)
- [x] Brick 1: The base model → [notes](notes/01-base-model.md) · [code](bricks/01_base_model.py)
- [x] Full codebase built and smoke-tested
- [ ] Line-by-line walkthrough: data → rewards → model → grpo → train → evaluate
- [ ] Baseline evaluation (original model)
- [ ] Training run
- [ ] After evaluation and write-up
