"""
train.py: the loop. It repeats one GRPO step (grpo.py) many times.

Run:
    uv run python -m reasoning.train                     # full run, defaults from config.py
    uv run python -m reasoning.train --max_steps 5       # quick smoke test

What to watch in the printed log:
    reward    average total score of the group (max 4.0). Should trend UP.
    correct   fraction of attempts with the right answer. The one that matters.
    format    fraction with the <reasoning>/<answer> layout. Usually rises FIRST.
    kl        distance from the original model. Should stay small (< ~0.5).
    len       average attempt length in tokens.
    loss      hovers near 0 and that's EXPECTED. Advantages average to zero
              within a group by construction, so the loss VALUE is ~0 (plus a
              tiny KL term). Its GRADIENT is not zero, and the gradient is what
              moves the weights. Don't read progress from this column; watch reward.

The first step's learning rate is 0 (warmup starts from zero), so step 1 never changes anything.

One step is noisy (one question, 6 attempts), so judge trends over 10s of steps.

Output (in cfg.output_dir):
    log.jsonl           one line of numbers per step, for plotting later
    checkpoint-N/       LoRA adapter saved every save_every steps
    final/              the adapter at the end; evaluate.py loads this
"""

import json
import math
import time
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from .config import parse_args
from .data import load_gsm8k
from .grpo import generate_group, group_advantages, grpo_loss
from .model import count_parameters, load_for_training
from .rewards import score


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)   # same seed -> same shuffle and same dice -> reproducible run
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(cfg), indent=2))  # record exactly what we ran

    # ---------------------------------------------------------------- setup
    model, tokenizer, device = load_for_training(cfg)
    trainable, total = count_parameters(model)
    print(f"Model: {cfg.model_name} on {device}")
    print(f"Training {trainable:,} of {total:,} weights ({trainable / total:.2%}): only the LoRA adapter\n")

    dataset = load_gsm8k("train", one_shot=cfg.one_shot).shuffle(seed=cfg.seed)

    # The optimiser only gets the trainable (LoRA) weights. The frozen ones
    # aren't even handed to it.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    # Learning rate: 0 -> peak over the warmup, then a cosine curve back to 0.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(cfg.max_steps * cfg.warmup_ratio),
        num_training_steps=cfg.max_steps,
    )

    log_file = open(out_dir / "log.jsonl", "a")
    header = f"{'step':>5} {'reward':>7} {'correct':>8} {'format':>7} {'kl':>7} {'len':>5} {'loss':>8} {'sec':>5}"
    print(header)

    # ----------------------------------------------------------- the loop
    for step in range(1, cfg.max_steps + 1):
        t0 = time.time()
        example = dataset[(step - 1) % len(dataset)]   # one question per step (ref: batch size 1)

        # 1. SAMPLE: G attempts at this question. eval() just switches off
        #    training-only behaviour; with LoRA dropout=0 it changes nothing,
        #    but it's the right habit.
        model.eval()
        prompt_ids, completion_ids, mask, texts = generate_group(
            model, tokenizer, example["prompt"], cfg, device
        )

        # 2. SCORE: every grader on every attempt, then add up per attempt.
        scores = score(texts, example["answer"])                # {grader: [G scores]}
        rewards = torch.tensor([sum(col) for col in zip(*scores.values())], device=device)  # [G]

        # 3. COMPARE: better or worse than the group's average?
        advantages = group_advantages(rewards)

        # 4. NUDGE: compute the loss, backpropagate, take one optimiser step.
        model.train()
        loss, kl = grpo_loss(model, prompt_ids, completion_ids, mask, advantages, cfg)
        optimizer.zero_grad()           # clear last step's gradients (they accumulate otherwise)
        loss.backward()                 # gradients: which way should each LoRA weight move?
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)  # cap the step size
        optimizer.step()                # move the weights
        scheduler.step()                # advance the learning-rate schedule

        # ------------------------------------------------------ bookkeeping
        stats = {
            "step": step,
            "reward": rewards.mean().item(),
            "correct": sum(s > 0 for s in scores["correct"]) / len(texts),
            "format": sum(s > 0 for s in scores["soft_fmt"]) / len(texts),
            "kl": kl.item(),
            "len": mask.sum(dim=1).float().mean().item(),
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
            "sec": time.time() - t0,
            **{f"r_{name}": sum(v) / len(v) for name, v in scores.items()},  # per-grader averages
        }
        log_file.write(json.dumps(stats) + "\n")
        log_file.flush()
        print(
            f"{step:>5} {stats['reward']:>7.2f} {stats['correct']:>8.0%} {stats['format']:>7.0%} "
            f"{stats['kl']:>7.4f} {stats['len']:>5.0f} {stats['loss']:>8.4f} {stats['sec']:>5.1f}"
        )

        # Every so often, show one full attempt so we can see what the numbers mean.
        if step % cfg.sample_every == 0 or step == 1:
            best = int(rewards.argmax())
            print(f"\n----- step {step} | key: {example['answer']} | best attempt (reward {rewards[best]:.2f}) -----")
            print(f"Q: {example['question']}\n")
            print(texts[best].strip())
            print("-" * 70 + "\n" + header)

        if step % cfg.save_every == 0:
            model.save_pretrained(out_dir / f"checkpoint-{step}")   # saves only the adapter (a few MB)

    model.save_pretrained(out_dir / "final")
    log_file.close()
    print(f"\nDone. Adapter saved to {out_dir / 'final'}")
    print(f"Evaluate it:  uv run python -m reasoning.evaluate --adapter {out_dir / 'final'}")


if __name__ == "__main__":
    main()
