# Brick 1: The Base Model

Code: [`bricks/01_base_model.py`](../bricks/01_base_model.py), using Qwen2.5-0.5B-Instruct as a small local stand-in for Llama 3.1 8B.

## Key ideas (what the run showed)

- **Tokens are pieces of text; the model is weights.** The model has a menu of ~151k tokens and is made of ~494M weights.
- **One step = one probability list over the whole menu.** For "The capital of France is", the model gives " Paris" 30%, " ______" 12%, and so on. It assigns odds; it doesn't *know* the answer.
- **Numbers get chopped digit by digit.** `9.11` becomes `9` `.` `1` `1`, so the model never sees "eleven".
- **Generation is a loop of dice rolls**, one roll per token.
- **Temperature reshapes the odds, not the number of rolls.** At T=0.2 Paris has 99%; at T=2.0 it drops to 2%. T=0 means no dice: always take the top token.
- **T=0 gave 6 identical answers to the pen problem, all wrong.** T=1.0 gave 6 different paths: some off the rails, some on the right track.

## Explain-back: open

I moved on before writing my explain-back. Check these again at Brick 4 (GRPO):

1. Six identical answers at T=0: what's the group average, and what does GRPO learn from them?
2. When training changes the weights, what's the effect on attempts like #4 and #6?
