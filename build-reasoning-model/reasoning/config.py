"""
config.py: every knob in one place.

Training a model means choosing many numbers: how many attempts per question,
how big each nudge is, how long answers may be, and so on. Scattering them
across files makes them hard to find and easy to get inconsistent, so they
all live here in one dataclass.

Any field can be overridden from the command line, for example:

    uv run python -m reasoning.train --max_steps 20 --num_generations 4

Values marked "(ref)" match the reference notebook. Values that differ say why.
"""

import argparse
from dataclasses import dataclass, fields


@dataclass
class Config:
    # ------------------------------------------------------------------ MODEL
    # Which pretrained model plays "the student" (Brick 1).
    # The reference uses meta-llama/Llama-3.1-8B-Instruct, which needs an NVIDIA
    # GPU with a lot of memory. Qwen2.5-0.5B-Instruct is 16x smaller, so it fits
    # on a Mac, and it works exactly the same way. Bigger options for a Colab
    # GPU: "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct".
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # LoRA (Brick 5): instead of changing all ~500M weights, we freeze them and
    # train a small add-on. lora_rank sets the add-on's size. Bigger rank means
    # more capacity, more memory, and slower training.
    lora_rank: int = 8          # (ref)
    lora_alpha: int = 8         # (ref) scales the add-on's effect; alpha/rank = 1 here

    # ------------------------------------------------------------------ DATA
    # Show one filled-in example before each question (see data.py, ONE_SHOT).
    # Off means the model must discover the format alone. Small models usually
    # can't, so training never gets started. Try --one_shot false to see it.
    one_shot: bool = True

    # ------------------------------------------------------------- GENERATION
    # The "6 attempts" from Brick 0. Each question is answered this many times,
    # independently, and the attempts are compared against each other.
    num_generations: int = 6            # (ref)

    # Hard caps on length, in tokens. The prompt cap is a safety net (GSM8K
    # questions are short). The completion cap bounds memory and time: an
    # answer that hasn't finished by then gets cut off, and is usually wrong
    # and badly formatted, so the reward signal teaches the model to be concise.
    max_prompt_length: int = 256        # (ref)
    max_completion_length: int = 200    # (ref)

    # How loaded the dice are (Brick 1). We need variety between the attempts.
    # At temperature 0 all 6 attempts are identical and there's nothing to compare.
    temperature: float = 0.9

    # ----------------------------------------------------------- OPTIMISATION
    # Step size of each nudge to the weights. The reference uses 5e-6. We use
    # 1e-5 because a small LoRA on a small model over a few hundred steps barely
    # moves at 5e-6. Too high and training goes unstable (rewards collapse).
    learning_rate: float = 1e-5

    # AdamW settings: the optimiser that turns gradients into weight updates.
    adam_beta1: float = 0.9             # (ref) how much gradient "momentum" to keep
    adam_beta2: float = 0.99            # (ref) how quickly to adapt per-weight step sizes
    weight_decay: float = 0.1           # (ref) gentle pull of weights toward zero

    # Learning-rate schedule: ramp up over the first 10% of steps, then decay
    # along a cosine curve to zero. Small careful steps at the start and end.
    warmup_ratio: float = 0.1           # (ref)

    # Gradient clipping: if a single update would be bigger than this, shrink
    # it. RL rewards are noisy, so this guards against one lucky or unlucky
    # group yanking the model too far.
    max_grad_norm: float = 0.1          # (ref)

    # KL penalty strength ("beta" in the GRPO paper). Charges the model for
    # drifting away from the original model, so it can't forget how to write
    # English while chasing rewards. 0.0 turns it off.
    kl_beta: float = 0.04

    # Number of training steps. Each step = one question x num_generations attempts.
    max_steps: int = 250                # (ref)

    # ------------------------------------------------------------ BOOKKEEPING
    seed: int = 42                      # same seed = same shuffle, same dice = reproducible
    output_dir: str = "outputs/run"     # where adapters and logs are written
    save_every: int = 50                # save a checkpoint every N steps
    sample_every: int = 10              # print one full attempt every N steps


def parse_args() -> Config:
    """Build a Config, letting `--field value` on the command line override any default."""
    parser = argparse.ArgumentParser(description="Train a reasoning model with GRPO.")
    for f in fields(Config):
        # f.type is the annotation (int, float, str, bool); argparse uses it to
        # convert the text. bool needs special care: bool("false") is True in
        # Python, because any non-empty string is truthy.
        convert = str_to_bool if f.type is bool else f.type
        parser.add_argument(f"--{f.name}", type=convert, default=f.default)
    return Config(**vars(parser.parse_args()))


def str_to_bool(text: str) -> bool:
    return text.lower() in ("1", "true", "yes", "on")
