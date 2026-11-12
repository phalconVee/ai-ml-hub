"""
Brick 1: The base model.

Run it with:  uv run bricks/01_base_model.py

It makes three ideas visible:
  A. Tokens: how text gets chopped up before the model sees it.
  B. One step: the model's output is a probability list over the menu of tokens, not a sentence.
  C. Dice and temperature: why the same question can give different answers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # small stand-in for the reference's Llama 3.1 8B
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading {MODEL} on {device} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
model.eval()

n_weights = sum(p.numel() for p in model.parameters())
print(f"Menu size (vocabulary): {len(tokenizer):,} tokens")
print(f"Weights (the model itself): {n_weights:,} numbers\n")


def banner(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------- A. Tokens
banner("A. TOKENS: how text is chopped before the model sees it")
for text in ["unbelievable", "Which is bigger, 9.11 or 9.9?", "The capital of France is"]:
    ids = tokenizer.encode(text)
    pieces = [tokenizer.decode([i]) for i in ids]
    print(f"\n{text!r}")
    print(f"  pieces: {pieces}")
    print(f"  ids:    {ids}")


# ------------------------------------------------------- B. One single step
banner("B. ONE STEP: what the model actually outputs")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**inputs).logits[0, -1]  # one score per token on the menu
probs = torch.softmax(logits, dim=-1)

print(f"\nInput: {prompt!r}")
print(f"Output: a list of {probs.shape[0]:,} probabilities (one per token). Top 10:\n")
top = torch.topk(probs, 10)
for p, i in zip(top.values, top.indices):
    print(f"  {p.item():7.2%}  {tokenizer.decode([i.item()])!r}")
print(f"\n  ...plus {probs.shape[0] - 10:,} more, all summing to {probs.sum().item():.3f}")


# ------------------------------------------- C. Temperature reshapes the dice
banner("C1. TEMPERATURE: same probabilities, differently loaded dice")
for T in [0.2, 1.0, 2.0]:
    p_T = torch.softmax(logits / T, dim=-1)
    top = torch.topk(p_T, 5)
    row = "  ".join(f"{tokenizer.decode([i.item()])!r}={p.item():.0%}" for p, i in zip(top.values, top.indices))
    print(f"\n  T={T}:  {row}")
print("\n  T=0 is the extreme: no dice at all, always take the #1 token.")


banner("C2. SIX ATTEMPTS at the same question")
question = "A shop sells 3 pens for $2. How much do 12 pens cost? Explain briefly."
chat = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
)
chat_inputs = tokenizer(chat, return_tensors="pt").to(device)
n_prompt = chat_inputs["input_ids"].shape[1]


def attempts(**gen_kwargs):
    # Six separate runs: no attempt sees another (same as in GRPO).
    results = []
    for _ in range(6):
        out = model.generate(
            **chat_inputs, max_new_tokens=70, pad_token_id=tokenizer.eos_token_id, **gen_kwargs
        )
        results.append(tokenizer.decode(out[0][n_prompt:], skip_special_tokens=True).strip())
    return results


print(f"\nQuestion: {question}")
for label, kwargs in [
    ("TEMPERATURE 0 (no dice: always pick the top token)", dict(do_sample=False)),
    ("TEMPERATURE 1.0 (roll the dice)", dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)),
]:
    print(f"\n--- {label} ---")
    for n, text in enumerate(attempts(**kwargs), 1):
        print(f"\n[{n}] {' '.join(text.split())[:260]}")
