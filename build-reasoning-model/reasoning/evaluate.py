"""
evaluate.py: did it work? (Bricks 6 and 7)

Measures a model on GSM8K TEST questions, which training never saw. Getting
training questions right could just be memorising; unseen ones show whether
it learned a skill.

    uv run python -m reasoning.evaluate                                  # original model
    uv run python -m reasoning.evaluate --adapter outputs/run/final      # after training
    uv run python -m reasoning.evaluate --adapter outputs/run/final --n 200

Evaluation uses temperature 0 (no dice, always the top token). Training needed
randomness to explore; measuring needs the opposite, one repeatable answer per
question so before and after are a fair comparison.

Reports:
    accuracy   right answer in the <answer> box
    format     used the <reasoning>/<answer> layout at all
Then asks the classic trick question, "which is bigger, 9.11 or 9.9?"
"""

import argparse

import torch
from peft import PeftModel

from .data import build_prompt, load_gsm8k
from .rewards import correctness_reward, soft_format_reward
from .model import load_base_model, load_tokenizer, pick_device

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@torch.no_grad()
def answer_batch(model, tokenizer, questions: list[str], device, max_new_tokens: int, one_shot: bool = True) -> list[str]:
    """Greedy answers to several questions at once (left padding lines up their ends)."""
    texts = [
        tokenizer.apply_chat_template(build_prompt(q, one_shot), tokenize=False, add_generation_prompt=True)
        for q in questions
    ]
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    out = model.generate(
        **batch,
        do_sample=False,             # temperature 0: always the most likely token
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on GSM8K test questions.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="base model (must match training)")
    parser.add_argument("--adapter", default=None, help="path to a trained LoRA adapter; omit for the original model")
    parser.add_argument("--n", type=int, default=50, help="how many test questions")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--one_shot", type=int, default=1, help="1 = same one-shot prompt as training (default), 0 = none")
    args = parser.parse_args()
    one_shot = bool(args.one_shot)

    device = pick_device()
    tokenizer = load_tokenizer(args.model_name)
    model = load_base_model(args.model_name, device)
    if args.adapter:
        # Re-attach the trained A/B matrices to the frozen original.
        model = PeftModel.from_pretrained(model, args.adapter).to(device)
    model.eval()
    label = f"adapter {args.adapter}" if args.adapter else "original model (no training)"
    print(f"Evaluating {args.model_name} + {label} on {args.n} test questions, {device}\n")

    test = load_gsm8k("test", one_shot=one_shot).select(range(args.n))   # always the same first n, so runs are comparable
    correct = formatted = 0
    for start in range(0, args.n, args.batch_size):
        rows = test.select(range(start, min(start + args.batch_size, args.n)))
        replies = answer_batch(model, tokenizer, rows["question"], device, args.max_new_tokens, one_shot)
        for reply, key in zip(replies, rows["answer"]):
            # Reuse the training graders, so "correct" means the same thing here as there.
            correct += correctness_reward([reply], key)[0] > 0
            formatted += soft_format_reward([reply], key)[0] > 0
        done = start + len(replies)
        print(f"  {done:>4}/{args.n}   accuracy {correct / done:6.1%}   format {formatted / done:6.1%}")

    print(f"\nRESULT  accuracy {correct / args.n:.1%}   format {formatted / args.n:.1%}")

    trick = "Which is bigger, 9.11 or 9.9?"
    print(f"\n----- {trick} -----")
    print(answer_batch(model, tokenizer, [trick], device, args.max_new_tokens, one_shot)[0].strip())


if __name__ == "__main__":
    main()
