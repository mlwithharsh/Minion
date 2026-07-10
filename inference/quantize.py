"""
Minion AI — inference/quantize.py
Helpers to load checkpoints in 4-bit quantized form for RTX 3050 (4 GB VRAM).

Two paths:
  Path A (default): Load Minion or HF model in 4-bit NF4 via bitsandbytes.
                    Requires bitsandbytes, works with any CUDA GPU.
                    Expected VRAM for 125M model: ~100 MB.
                    Expected VRAM for GPT-2 XL (1.5B): ~800 MB.

  Path B (optional): Export to GGUF for llama.cpp / ctransformers.
                     Requires llama-cpp-python (compiled with CUDA support).
                     Useful if you want to run entirely without PyTorch.
                     See export_gguf() below.

Typical usage:
    model, tokenizer, cfg = load_quantized_model("out/ckpt.pt")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml


def load_quantized_model(
    ckpt_path: str,
    quantize: bool = True,
    device: str = "cuda",
) -> tuple:
    """Load a Minion checkpoint (Mode A) or LoRA checkpoint (Mode B) quantized.

    Args:
        ckpt_path: Path to .pt checkpoint saved by train.py.
        quantize:  If True, load model in 4-bit NF4 (bitsandbytes).
                   If False, load in the original dtype (fp16/fp32).
        device:    'cuda' or 'cpu'.

    Returns:
        (model, tokenizer, cfg)  where model is ready for inference.

    VRAM estimates (4-bit, RTX 3050 4 GB):
        125M params:   ~100 MB model  +  ~200 MB KV cache  →  <1 GB total
        350M params:   ~250 MB model  +  ~400 MB KV cache  →  ~1–2 GB total
        GPT-2 XL 1.5B: ~800 MB model +  ~1 GB KV cache    →  ~2–3 GB total
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})
    mode = ckpt.get("mode", cfg.get("mode", "pretrain"))

    print(f"Mode: {mode}")

    if mode == "pretrain":
        model, tokenizer = _load_pretrain_model(ckpt, cfg, quantize, device)
    elif mode == "qlora_finetune":
        model, tokenizer = _load_qlora_model(ckpt, cfg, quantize, device)
    else:
        raise ValueError(f"Unknown checkpoint mode: {mode!r}")

    model.eval()
    print(f"Model ready on {device}.")

    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated(device) / 1024 ** 2
        print(f"VRAM allocated: {vram_mb:.0f} MB")

    return model, tokenizer, cfg


def _load_pretrain_model(ckpt: dict, cfg: dict, quantize: bool, device: str):
    """Load a Minion (Mode A) pretrain checkpoint."""
    # Add parent dir to path so we can import model.py
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    from model import Minion, MinionConfig

    model_cfg = MinionConfig(
        vocab_size              = cfg.get("vocab_size", 32768),
        block_size              = cfg.get("block_size", 1024),
        n_layer                 = cfg.get("n_layer", 12),
        n_head                  = cfg.get("n_head", 12),
        n_embd                  = cfg.get("n_embd", 768),
        n_kv_heads              = cfg.get("n_kv_heads", 3),
        rope_theta              = cfg.get("rope_theta", 10_000.0),
        rope_scaling            = cfg.get("rope_scaling"),
        swiglu_intermediate_dim = cfg.get("swiglu_intermediate_dim"),
        dropout                 = 0.0,  # Always 0 for inference
    )

    if quantize and device == "cuda":
        model = _quantize_minion(model_cfg, ckpt["model"])
    else:
        model = Minion(model_cfg)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(sd, strict=False)
        model = model.to(device)

    tokenizer = _load_tokenizer(cfg)
    return model, tokenizer


def _quantize_minion(model_cfg, state_dict: dict):
    """Load Minion in 4-bit NF4 via bitsandbytes.

    bitsandbytes quantizes nn.Linear layers to 4-bit NF4 on the fly.
    We load weights in fp16, then let BnB handle quantization.
    """
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        print("bitsandbytes not installed — loading in fp16 instead (more VRAM)")
        from model import Minion
        m = Minion(model_cfg).half().cuda()
        sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        m.load_state_dict(sd, strict=False)
        return m

    try:
        import transformers
        from transformers import BitsAndBytesConfig, AutoConfig
    except ImportError:
        pass

    # Build model in fp16, then replace Linear layers with Linear4bit
    from model import Minion
    import torch.nn as nn

    m = Minion(model_cfg)
    sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    m.load_state_dict(sd, strict=False)

    def _replace_linear(module: nn.Module, name: str = "") -> nn.Module:
        """Recursively replace nn.Linear with Linear4bit (NF4)."""
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                # Skip tiny layers (e.g. final RMSNorm weight is not Linear)
                new_layer = Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    quant_type="nf4",
                    compute_dtype=torch.float16,
                )
                # Copy weight data
                new_layer.weight.data = child.weight.data
                if child.bias is not None:
                    new_layer.bias = child.bias
                setattr(module, child_name, new_layer)
            else:
                _replace_linear(child, child_name)
        return module

    m = _replace_linear(m)
    m = m.cuda()
    print("Minion loaded in 4-bit NF4 (bitsandbytes)")
    return m


def _load_qlora_model(ckpt: dict, cfg: dict, quantize: bool, device: str):
    """Load a Mode B LoRA checkpoint (base model + adapters)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    base_model_name = cfg.get("base_model_name", "openai-community/gpt2-xl")
    bnb_config = None

    if quantize and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit             = True,
            bnb_4bit_quant_type      = "nf4",
            bnb_4bit_compute_dtype   = torch.float16,
            bnb_4bit_use_double_quant= True,
        )

    print(f"Loading base model '{base_model_name}' ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config = bnb_config,
        device_map          = "auto" if device == "cuda" else None,
        torch_dtype         = torch.float16,
    )

    # Save LoRA adapter weights to a temp dir then load via PeftModel
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        # The checkpoint may contain just the LoRA state dict;
        # save as PEFT-compatible adapter_model.bin
        adapter_sd = ckpt["model"]
        # Filter to only LoRA keys (they contain 'lora_')
        lora_sd = {k: v for k, v in adapter_sd.items() if "lora_" in k}
        if lora_sd:
            torch.save(lora_sd, os.path.join(tmpdir, "adapter_model.bin"))
            # Write minimal adapter_config.json
            import json
            adapter_cfg = {
                "base_model_name_or_path": base_model_name,
                "peft_type": "LORA",
                "r": cfg.get("lora_rank", 16),
                "lora_alpha": cfg.get("lora_alpha", 32),
                "lora_dropout": cfg.get("lora_dropout", 0.05),
                "target_modules": cfg.get("lora_target_modules", ["c_attn", "c_proj"]),
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
            with open(os.path.join(tmpdir, "adapter_config.json"), "w") as f:
                json.dump(adapter_cfg, f)
            model = PeftModel.from_pretrained(base, tmpdir)
            model = model.merge_and_unload()  # Merge LoRA weights for faster inference
        else:
            # No LoRA keys found — load as full model state dict
            base.load_state_dict(adapter_sd, strict=False)
            model = base

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer


def _load_tokenizer(cfg: dict):
    """Load tokenizer from checkpoint config."""
    tok_path = cfg.get("tokenizer_path", "tokenizer")
    tok_json = os.path.join(tok_path, "tokenizer.json")

    if os.path.exists(tok_json):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(tok_json)
    elif os.path.exists(tok_path):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tok_path)
    else:
        print(f"Warning: no tokenizer found at {tok_path!r}. Returning None.")
        return None


# ---------------------------------------------------------------------------
# Optional GGUF export path
# ---------------------------------------------------------------------------

def export_gguf(
    ckpt_path: str,
    output_path: str,
    quantize_type: str = "q4_0",
) -> None:
    """Export a Minion checkpoint to GGUF format for llama.cpp inference.

    GGUF allows running the model entirely via llama.cpp with no PyTorch
    dependency — useful for ultra-low-power local inference.

    Requirements:
        pip install llama-cpp-python --extra-index-url ...
        And llama.cpp's convert.py script (from the llama.cpp repo).

    Args:
        ckpt_path:     Path to .pt checkpoint.
        output_path:   Output .gguf file path.
        quantize_type: GGUF quantization type (q4_0, q4_k_m, q8_0, etc.)

    Note: Minion's architecture (GQA, RoPE, SwiGLU) maps closely to
    LLaMA's architecture.  The GGUF export converts weight names to
    the llama.cpp naming convention.
    """
    print(
        "GGUF export: This requires the llama.cpp repository and convert.py script.\n"
        "Steps:\n"
        "  1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp\n"
        "  2. pip install -r llama.cpp/requirements/requirements-convert.txt\n"
        "  3. python llama.cpp/convert.py <model_dir> --outfile model.gguf --outtype q4_0\n"
        "\n"
        "The primary inference path for Minion AI is 4-bit bitsandbytes (see quantize.py).\n"
        "GGUF export is an advanced option for llama.cpp-native deployments."
    )
    raise NotImplementedError(
        "GGUF export requires llama.cpp's convert.py — see instructions printed above."
    )
