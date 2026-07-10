"""
Minion AI — inference/generate.py
CLI generation script for the local RTX 3050 (4 GB VRAM).

Usage:
    # Basic generation from a checkpoint:
    python inference/generate.py --checkpoint out/ckpt.pt --prompt "The future of AI is"

    # With sampling parameters:
    python inference/generate.py \\
        --checkpoint out/ckpt.pt \\
        --prompt "Once upon a time" \\
        --max_new_tokens 200 \\
        --temperature 0.8 \\
        --top_k 50 \\
        --top_p 0.95 \\
        --num_samples 3

    # Without quantization (more VRAM, faster, for debugging):
    python inference/generate.py --checkpoint out/ckpt.pt --no-quantize --prompt "Hello"

Expected performance on RTX 3050 (4 GB VRAM):
    125M model, 4-bit, fp16 compute: ~40–80 tokens/sec
    350M model, 4-bit, fp16 compute: ~20–40 tokens/sec
    GPT-2 XL (Mode B), 4-bit:       ~15–25 tokens/sec
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def encode(tokenizer, text: str) -> list[int]:
    """Tokenize text with either HF tokenizers or transformers tokenizer."""
    enc = tokenizer.encode(text)
    if hasattr(enc, "ids"):
        return enc.ids       # HF tokenizers Encoding object
    if hasattr(enc, "tolist"):
        return enc.tolist()  # torch.Tensor
    return list(enc)


def decode(tokenizer, ids: list[int]) -> str:
    """Decode token ids to text."""
    if hasattr(tokenizer, "decode"):
        result = tokenizer.decode(ids)
        # Some tokenizers return a string directly; some return an object
        if hasattr(result, "tokens"):
            return result.tokens
        return str(result)
    return str(ids)


def generate_text(
    model,
    tokenizer,
    cfg: dict,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    """Generate text from a prompt.

    Args:
        model:          Loaded model (from load_quantized_model).
        tokenizer:      Loaded tokenizer.
        cfg:            Config dict from the checkpoint.
        prompt:         Input text prompt.
        max_new_tokens: Number of new tokens to generate.
        temperature:    Sampling temperature (higher = more random).
        top_k:          Keep only top-k logits (0 = disabled).
        top_p:          Nucleus sampling threshold (1.0 = disabled).
        device:         'cuda' or 'cpu'.

    Returns:
        Generated text string (not including the prompt).
    """
    model.eval()
    input_ids = encode(tokenizer, prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    block_size = cfg.get("block_size", 1024)

    # Check if model is our Minion class or HF wrapper
    with torch.no_grad():
        if hasattr(model, "generate"):
            out = model.generate(
                idx,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                top_k          = top_k if top_k > 0 else None,
                top_p          = top_p if top_p < 1.0 else None,
            )
        else:
            # HF model generate
            out = model.generate(
                idx,
                max_new_tokens    = max_new_tokens,
                temperature       = temperature,
                top_k             = top_k if top_k > 0 else None,
                top_p             = top_p if top_p < 1.0 else None,
                do_sample         = temperature > 0,
                pad_token_id      = getattr(tokenizer, "eos_token_id", 0),
            )

    # Decode only the newly generated tokens (exclude prompt)
    new_ids = out[0, len(input_ids) :].tolist()
    return decode(tokenizer, new_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minion AI — local generation on RTX 3050 (4 GB VRAM)"
    )
    parser.add_argument("--checkpoint",    required=True,      help="Path to .pt checkpoint")
    parser.add_argument("--prompt",        default="",         help="Input prompt text")
    parser.add_argument("--prompt_file",   default=None,       help="Read prompt from this file")
    parser.add_argument("--max_new_tokens",type=int, default=100)
    parser.add_argument("--temperature",   type=float, default=0.8)
    parser.add_argument("--top_k",         type=int,   default=50,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top_p",         type=float, default=0.95,
                        help="Nucleus sampling (1.0 = disabled)")
    parser.add_argument("--num_samples",   type=int,   default=1,
                        help="Number of independent samples to generate")
    parser.add_argument("--device",        default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--no-quantize",   action="store_true",
                        help="Load in fp16 instead of 4-bit (uses more VRAM)")
    parser.add_argument("--seed",          type=int, default=42)

    args = parser.parse_args()

    # Device check
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU (will be slow)")
        args.device = "cpu"

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(args.device)
        print(f"GPU: {props.name}  ({props.total_memory / 1024**3:.1f} GB VRAM)")

    torch.manual_seed(args.seed)

    # Load model
    # Add repo root to path for model.py imports
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    from inference.quantize import load_quantized_model

    model, tokenizer, cfg = load_quantized_model(
        ckpt_path = args.checkpoint,
        quantize  = not args.no_quantize,
        device    = args.device,
    )

    if tokenizer is None:
        print("ERROR: No tokenizer found. Run tokenizer_train.py first.")
        sys.exit(1)

    # Prompt
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()

    if not prompt:
        prompt = input("Enter prompt: ")

    print(f"\n{'='*60}")
    print(f"Prompt: {prompt!r}")
    print(f"{'='*60}\n")

    # Generate
    for i in range(args.num_samples):
        t0 = time.perf_counter()

        text = generate_text(
            model          = model,
            tokenizer      = tokenizer,
            cfg            = cfg,
            prompt         = prompt,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
            device         = args.device,
        )

        dt = time.perf_counter() - t0
        tps = args.max_new_tokens / dt

        if args.num_samples > 1:
            print(f"--- Sample {i+1}/{args.num_samples} "
                  f"({tps:.1f} tok/sec, {dt:.2f}s) ---")
        else:
            print(f"({tps:.1f} tokens/sec on {args.device})\n")

        print(prompt + text)
        print()

    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated(args.device) / 1024 ** 2
        print(f"Peak VRAM: {vram_mb:.0f} MB")


if __name__ == "__main__":
    main()
