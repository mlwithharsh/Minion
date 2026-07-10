"""Minion AI inference module.

Standalone inference package — no training dependencies required.
Load quantized checkpoints and run generation on an RTX 3050 (4 GB VRAM).

Usage:
    from inference import load_model, generate

    model, tokenizer, cfg = load_model("path/to/ckpt.pt")
    text = generate(model, tokenizer, cfg, prompt="Once upon a time")
    print(text)
"""

from inference.quantize import load_quantized_model
from inference.generate import generate_text

__all__ = ["load_quantized_model", "generate_text"]
