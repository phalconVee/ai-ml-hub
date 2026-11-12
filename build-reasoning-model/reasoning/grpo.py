"""
grpo.py: the heart of the project. Group Relative Policy Optimization (Brick 4).

One GRPO step, in plain words:

    1. SAMPLE    Ask one question. Roll the dice num_generations times (6)
                 to get 6 independent attempts.                    generate_group()
    2. SCORE     Grade each attempt with rewards.py.               (done in train.py)
    3. COMPARE   Turn raw scores into "better or worse than this
                 group's average". That's the advantage.           group_advantages()
    4. NUDGE     Change the weights so better-than-average attempts
                 become more likely, and worse ones less likely.   grpo_loss()

Step 3 is what makes it GRPO. Older methods (PPO) train a whole second neural
network, a "critic", just to estimate what score is normal for a question.
GRPO's trick is that the group itself is the yardstick: the average of 6
attempts at the same question is the baseline. No critic, half the memory.

Shapes used below (G = num_generations, P = prompt length, C = completion length):
    prompt_ids      [1, P]
    completion_ids  [G, C]
    mask            [G, C]   1 for real answer tokens, 0 for padding after the end
    logps           [G, C]   log-probability the model gave each token it wrote
"""

import torch

from .config import Config


# ------------------------------------------------------------- 1. SAMPLE


@torch.no_grad()  # generating is just "running" the model; no learning happens here
def generate_group(model, tokenizer, prompt_messages, cfg: Config, device):
    """
    Generate cfg.num_generations attempts at one question.

    Returns:
        prompt_ids      [1, P]  the question as token ids
        completion_ids  [G, C]  the attempts as token ids (padded to the longest)
        mask            [G, C]  which completion tokens are real
        texts           list of G strings, decoded for the graders
    """
    # Chat messages -> one string in the model's own format, ending with the
    # "assistant, your turn" marker (add_generation_prompt=True).
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_ids = prompt_ids[:, -cfg.max_prompt_length:]   # keep the end if it's ever too long

    output = model.generate(
        input_ids=prompt_ids,
        attention_mask=torch.ones_like(prompt_ids),
        do_sample=True,                        # roll the dice (Brick 1)
        temperature=cfg.temperature,
        # Turn OFF the model's built-in sampling tweaks. Qwen ships with
        # top_k=20, top_p=0.8 and a repetition penalty. These change the odds
        # the dice actually use, but grpo_loss() computes log-probs from the
        # plain model. If the two disagree, we'd be training on probabilities
        # that weren't the real ones. Pure temperature sampling keeps them in sync.
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=cfg.max_completion_length,
        num_return_sequences=cfg.num_generations,  # the group: G independent attempts
        pad_token_id=tokenizer.pad_token_id,
    )
    # generate() returns prompt + answer glued together; keep just the answer part.
    completion_ids = output[:, prompt_ids.shape[1]:]

    mask = completion_mask(completion_ids, end_token_ids(model, tokenizer))
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return prompt_ids, completion_ids, mask, texts


def end_token_ids(model, tokenizer) -> list[int]:
    """
    All token ids that mean "I'm done". Chat models often have more than one
    (Qwen: <|im_end|> ends a chat turn, <|endoftext|> ends a document).
    """
    ids = model.generation_config.eos_token_id
    ids = ids if isinstance(ids, list) else [ids]
    return list({*ids, tokenizer.eos_token_id})


def completion_mask(completion_ids: torch.Tensor, end_ids: list[int]) -> torch.Tensor:
    """
    1 for every token up to and including the first "done" token, 0 after.

    The attempts have different lengths but sit in one rectangular tensor, so
    short ones are padded at the end. Padding isn't something the model chose
    to write, so it must not be rewarded or punished. We keep the "done" token
    itself: deciding WHEN to stop is a real choice the model should learn.

        tokens: [ The, answer, is, 8, <end>, pad, pad ]
        mask:   [  1,    1,     1, 1,   1,    0,   0  ]
    """
    G, C = completion_ids.shape
    is_end = torch.isin(completion_ids, torch.tensor(end_ids, device=completion_ids.device))
    # Position of the first end token per row. Rows that never finished
    # (hit max_completion_length) get C, which keeps every token.
    has_end = is_end.any(dim=1)
    first_end = torch.where(has_end, is_end.int().argmax(dim=1), torch.full((G,), C, device=completion_ids.device))
    positions = torch.arange(C, device=completion_ids.device).expand(G, C)
    return (positions <= first_end.unsqueeze(1)).float()


# ------------------------------------------------------------ 3. COMPARE


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Raw scores -> "how much better than this group's average", in units of
    the group's spread.

        rewards     [2.5, 0.5, 0.5, 0.0, 2.5, 0.0]   mean 1.0
        advantages  [+1.3, -0.4, -0.4, -0.9, +1.3, -0.9]

    Subtracting the mean answers "better or worse than usual for THIS
    question?". A hard question where everyone scores 0.5 and an easy one
    where everyone scores 2.5 both give advantages of zero: nothing stood out,
    so there's nothing to learn.

    Dividing by the standard deviation keeps the nudge size similar whether a
    group's scores are close together or far apart. eps avoids dividing by zero
    when all scores are identical. Then every advantage is 0 and the step
    teaches nothing. (This is the T=0 thought experiment from Brick 1.)
    """
    return (rewards - rewards.mean()) / (rewards.std() + eps)


# --------------------------------------------------------------- 4. NUDGE


def token_logprobs(model, prompt_ids, completion_ids, temperature: float) -> torch.Tensor:
    """
    For each token the model actually wrote, the log-probability it assigned
    to writing it.  -> [G, C]

    We feed "prompt + completion" back through the model in a single pass.
    Every position predicts the next token (Brick 1), so we can read off,
    all at once, how likely each token of the answer was.

    No attention mask is needed: the prompt has no padding, and padding
    after the end can only be "seen" by later padding (a token never looks
    ahead), which the loss mask zeroes out anyway.
    """
    G, C = completion_ids.shape
    input_ids = torch.cat([prompt_ids.expand(G, -1), completion_ids], dim=1)

    # logits_to_keep=C+1: only compute scores for the last C+1 positions.
    # Scoring all 151k vocabulary entries for the prompt positions too would
    # waste a lot of memory.
    logits = model(input_ids=input_ids, logits_to_keep=C + 1, use_cache=False).logits

    # Position i's output predicts token i+1. The last kept position predicts
    # past the end, so drop it; what's left lines up with completion tokens 0..C-1.
    logits = logits[:, :-1, :] / temperature   # same temperature the dice used

    logps = torch.log_softmax(logits.float(), dim=-1)   # scores -> log-probabilities
    # gather: from each position's 151k-long list, pick the entry for the token actually written.
    return logps.gather(dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)


def grpo_loss(model, prompt_ids, completion_ids, mask, advantages, cfg: Config):
    """
    The number whose gradient, when we step "downhill", makes good attempts
    more likely and bad ones less likely.

    Per token t of attempt i:

        objective = ratio · A_i  -  β · KL_t
        loss      = -objective   (optimisers minimise, and we want to MAXIMISE)

    ratio = exp(logp - logp.detach()).
        Its value is exactly 1. Its gradient is the gradient of logp. It's
        written this way to match the paper's formula, where the ratio
        compares the new model to the one that generated the attempts. We
        take one update per batch of attempts, so the two are the same model
        and the ratio is 1, but its gradient still points toward "make this
        token more likely". Multiplying by A_i:
            A_i > 0  -> raise the probability of every token in attempt i
            A_i < 0  -> lower it
        That's the whole learning rule.

    KL_t (k3 estimator: e^(r-l) - (r-l) - 1, with r = reference log-prob, l = current):
        Always >= 0, and zero only when the current model agrees with the
        original. Penalising it is a leash: chase reward, but don't wander so
        far from the original model that you forget how to write.
        The reference model is just the same model with the LoRA adapter
        switched off (see model.py). No second copy in memory.

    Averaging: first over each attempt's real tokens (so long and short
    attempts count equally), then over the group.

    Returns (loss, mean_kl).
    """
    logps = token_logprobs(model, prompt_ids, completion_ids, cfg.temperature)

    ratio = torch.exp(logps - logps.detach())                 # == 1, carries the gradient
    per_token_objective = ratio * advantages.unsqueeze(1)     # broadcast A_i across attempt i's tokens

    if cfg.kl_beta > 0:
        with torch.no_grad(), model.disable_adapter():        # temporarily the original model
            ref_logps = token_logprobs(model, prompt_ids, completion_ids, cfg.temperature)
        diff = ref_logps - logps
        kl = torch.exp(diff) - diff - 1
        per_token_objective = per_token_objective - cfg.kl_beta * kl
    else:
        kl = torch.zeros_like(logps)

    per_token_loss = -per_token_objective
    per_attempt_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    loss = per_attempt_loss.mean()

    mean_kl = ((kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).mean()
    return loss, mean_kl.detach()
