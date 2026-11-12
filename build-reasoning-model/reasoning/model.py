"""
model.py: the student (Brick 1) plus the small trainable add-on (Brick 5).

Two ideas live here.

1. The pretrained model. Hundreds of millions of weights, already able to
   read, write and do some maths. We load it and FREEZE it: those weights
   never change.

2. LoRA (Low-Rank Adaptation). Next to some of the frozen weight matrices we
   attach two thin trainable matrices, A and B. A big frozen matrix W (say
   896x896 ≈ 800k numbers) gets a companion B·A where A is 8x896 and B is 896x8
   (~14k numbers). The layer's output becomes  W·x + (alpha/rank)·B·A·x.

   Training changes only A and B, about 0.2% of the model. Why:
     - memory: no gradients or optimiser state for the frozen 99.8%
     - safety: the original model is still there, untouched
     - a free "reference model": switch the adapter off and you're back to
       the original, which grpo.py uses for its KL penalty
     - tiny checkpoints: we save a few MB of adapter, not GBs of model

   B starts at zero, so at step 0 the adapter adds nothing and the model
   behaves exactly like the original.
"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config

# The attention projections inside every transformer layer: Query, Key, Value,
# Output. These decide what each token pays attention to. Same choice as the reference.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def pick_device() -> torch.device:
    """NVIDIA GPU ("cuda") if present, else Apple Silicon GPU ("mps"), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """
    How many bits per number. bfloat16 (16-bit) halves memory and is fast on
    modern NVIDIA GPUs. Elsewhere we use float32 (32-bit): slower but numerically
    safe. Small RL updates can vanish in rounding error at 16 bits on hardware
    that doesn't handle bf16 well.
    """
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_tokenizer(model_name: str):
    """
    The tokenizer turns text into token ids and back (Brick 1).

    padding_side="left": when several prompts of different lengths are
    batched (evaluate.py does this), the shorter ones are padded on the
    LEFT. The model writes its answer on the right-hand end, so every
    prompt must finish at the same position, right before the first new token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:          # some models (e.g. Llama) ship without one
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name: str, device: torch.device):
    """The pretrained model as downloaded, frozen."""
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=pick_dtype(device))
    return model.to(device)


def load_for_training(cfg: Config):
    """
    Base model + fresh LoRA adapter, ready to train.

    Returns (model, tokenizer, device). `model` is a PeftModel: it behaves like
    the original, but only the adapter's weights have requires_grad=True.
    """
    device = pick_device()
    tokenizer = load_tokenizer(cfg.model_name)
    model = load_base_model(cfg.model_name, device)

    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,        # no dropout: RL signal is noisy enough already
        bias="none",             # leave the original biases frozen too
        task_type="CAUSAL_LM",   # "predict the next token" models
    )
    # get_peft_model freezes every original weight and inserts A/B pairs.
    model = get_peft_model(model, lora)
    return model, tokenizer, device


def count_parameters(model) -> tuple[int, int]:
    """(trainable, total). Shows how tiny the LoRA part is."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
