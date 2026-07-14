"""
Minion AI — train.py
Training entry point for both Mode A (pretrain) and Mode B (qlora_finetune).

Can be called as:
    python train.py config/minion_pretrain_colab.yaml          # from CLI
    python train.py config/smoke_test.yaml --max_iters=20      # with overrides

Or imported into a Colab notebook cell:
    from train import train
    train("config/minion_pretrain_colab.yaml", wandb_log=False)

Mode A (pretrain):
    Trains a Minion model from scratch using the modernized architecture.
    Default config targets ~125M params on a Colab T4 (16 GB).

Mode B (qlora_finetune):
    Loads a pretrained HF causal LM in 4-bit NF4, attaches LoRA adapters,
    and fine-tunes only the adapters.  This is NOT pretraining — it is
    parameter-efficient fine-tuning of an existing model.
    The base model weights are frozen; only LoRA adapters are trained.

Colab-specific features:
    - Detects and logs GPU properties at startup.
    - Auto-detects bf16 support; falls back to fp16 + GradScaler.
    - Saves checkpoints to Google Drive (ckpt_dir config key).
    - Resumes automatically if a checkpoint exists in ckpt_dir.
    - Logs VRAM usage (torch.cuda.max_memory_allocated) every log_interval steps.
    - Warns when wall-clock time approaches session_limit_hours.
"""

from __future__ import annotations

import math
import os
import sys
import time
import pickle
import yaml
import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Default configuration (flat dict — all YAML keys go here too)
# ---------------------------------------------------------------------------

DEFAULTS: dict = {
    # ---- Mode ----
    "mode": "pretrain",            # "pretrain" | "qlora_finetune"

    # ---- I/O ----
    "ckpt_dir":    "/content/drive/MyDrive/minion_ckpts",  # Drive path on Colab
    "out_dir":     "out",           # Local output (also used for smoke tests)
    "eval_interval": 500,
    "log_interval":  10,
    "eval_iters":    50,
    "eval_only":     False,
    "always_save_checkpoint": True,
    "keep_best_checkpoint": True,  # Also save best-val-loss checkpoint separately

    # ---- W&B ----
    "wandb_log":     False,
    "wandb_project": "minion-ai",
    "wandb_run_name":"minion-run",

    # ---- Data ----
    "data_mode":          "streaming",   # "streaming" | "memmap"
    "data_dir":           "data/prepared",   # for memmap mode
    "hf_dataset_name":    "HuggingFaceFW/fineweb-edu",
    "hf_dataset_config":  "sample-10BT",
    "hf_train_split":     "train",
    "hf_val_split":       "train",    # FineWeb-Edu has no official val split
    "text_column":        "text",
    "hf_cache_dir":       None,

    # ---- Tokenizer ----
    "tokenizer_path": "tokenizer",   # Dir containing tokenizer.json

    # ---- Model (Mode A) ----
    "vocab_size":    32768,
    "block_size":    1024,
    "n_layer":       12,
    "n_head":        12,
    "n_embd":        768,
    "n_kv_heads":    3,
    "rope_theta":    10_000.0,
    "rope_scaling":  None,
    "swiglu_intermediate_dim": None,
    "dropout":       0.0,

    # ---- Mode B (qlora_finetune) ----
    "base_model_name":       "openai-community/gpt2-xl",
    "lora_rank":             16,
    "lora_alpha":            32,
    "lora_dropout":          0.05,
    "lora_target_modules":   ["c_attn", "c_proj"],  # GPT-2 attention projections
    "quantization":          "nf4",          # "nf4" | "fp4"
    "bnb_compute_dtype":     "float16",      # compute dtype for 4-bit ops

    # ---- Optimizer ----
    "learning_rate":  6e-4,
    "max_iters":      50_000,
    "weight_decay":   0.1,
    "beta1":          0.9,
    "beta2":          0.95,
    "grad_clip":      1.0,
    "use_8bit_adam":  True,   # 8-bit AdamW via bitsandbytes
    "use_lion":       False,  # Lion optimizer (lower memory, needs lower LR)

    # ---- LR schedule ----
    "decay_lr":       True,
    "warmup_iters":   1000,
    "lr_decay_iters": 50_000,
    "min_lr":         6e-5,

    # ---- Training efficiency ----
    "batch_size":                  8,    # micro-batch
    "gradient_accumulation_steps": 8,   # effective batch = 8*8 = 64
    "gradient_checkpointing":      True,
    "compile":                     True,  # torch.compile (may spike VRAM briefly)

    # ---- Precision ----
    "dtype": "auto",    # "auto" | "bf16" | "fp16" | "fp32"
                        # "auto" detects bf16 support at runtime

    # ---- Colab session management ----
    "session_limit_hours":  11.5,   # Warn when wall-clock approaches this
    "session_warn_fraction": 0.85,  # Warn at 85% of session_limit_hours

    # ---- System ----
    "device":  "cuda",
    "seed":    1337,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str], overrides: dict) -> dict:
    """Merge DEFAULTS ← YAML file ← CLI overrides."""
    cfg = dict(DEFAULTS)

    if config_path is not None:
        with open(config_path) as f:
            file_cfg = yaml.safe_load(f)
        if file_cfg:
            cfg.update(file_cfg)

    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# ---------------------------------------------------------------------------
# Precision helpers
# ---------------------------------------------------------------------------

def resolve_dtype(cfg: dict) -> tuple[str, torch.dtype, bool]:
    """Return (dtype_name, ptdtype, use_amp).

    T4 reports bf16 support in some Colab/PyTorch builds, but compute 7.5
    does not run bf16 tensor cores natively.  For T4, fp16 is faster.
    """
    dtype_pref = cfg.get("dtype", "auto")

    if dtype_pref == "auto":
        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            and torch.cuda.get_device_capability()[0] >= 8
        ):
            dtype_name = "bfloat16"
        elif torch.cuda.is_available():
            dtype_name = "float16"
        else:
            dtype_name = "float32"
    else:
        dtype_name = dtype_pref
    dtype_name = {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }.get(dtype_name, dtype_name)

    ptdtype = {
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
    }[dtype_name]

    use_amp = dtype_name in ("bfloat16", "float16")
    return dtype_name, ptdtype, use_amp


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: dict) -> float:
    """Cosine decay with linear warmup."""
    lr = cfg["learning_rate"]
    min_lr = cfg["min_lr"]
    warmup = cfg["warmup_iters"]
    decay  = cfg["lr_decay_iters"]

    if step < warmup:
        return lr * (step + 1) / (warmup + 1)
    if step > decay:
        return min_lr
    ratio = (step - warmup) / (decay - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (lr - min_lr)


# ---------------------------------------------------------------------------
# VRAM + wall-clock logging
# ---------------------------------------------------------------------------

def log_vram(device: str) -> float:
    """Return and print current max VRAM usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    mb = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    torch.cuda.reset_peak_memory_stats(device)
    return mb / 1024


def check_session_time(
    session_start: float,
    cfg: dict,
) -> None:
    """Warn (not crash) if we're approaching the Colab session wall-clock limit."""
    limit_h   = cfg.get("session_limit_hours", 11.5)
    warn_frac = cfg.get("session_warn_fraction", 0.85)
    elapsed_h = (time.time() - session_start) / 3600
    warn_at   = limit_h * warn_frac

    if elapsed_h >= warn_at:
        remaining_m = (limit_h - elapsed_h) * 60
        print(
            f"\n⚠️  SESSION WARNING: {elapsed_h:.1f}h elapsed of ~{limit_h}h limit. "
            f"~{remaining_m:.0f} min remaining. "
            f"Checkpoint saved to {cfg['ckpt_dir']} — resume after reconnect.\n"
        )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_path(ckpt_dir: str, name: str = "ckpt.pt") -> str:
    return os.path.join(ckpt_dir, name)


def save_checkpoint(
    raw_model,
    optimizer,
    cfg: dict,
    iter_num: int,
    best_val_loss: float,
    tokens_consumed: int,
    is_best: bool = False,
) -> None:
    """Save checkpoint to Drive (ckpt_dir) and optionally to local out_dir."""
    state = {
        "model":          raw_model.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "iter_num":       iter_num,
        "best_val_loss":  best_val_loss,
        "tokens_consumed":tokens_consumed,
        "config":         cfg,
        "mode":           cfg["mode"],
    }

    for out in [cfg["ckpt_dir"], cfg.get("out_dir", "out")]:
        os.makedirs(out, exist_ok=True)
        path = checkpoint_path(out, "ckpt.pt")
        torch.save(state, path)

    if is_best and cfg.get("keep_best_checkpoint", True):
        best_path = checkpoint_path(cfg["ckpt_dir"], "best_ckpt.pt")
        torch.save(state, best_path)
        print(f"  → best checkpoint updated (val_loss={best_val_loss:.4f}): {best_path}")

    print(f"  checkpoint saved: iter={iter_num}, val_loss={best_val_loss:.4f}")


def load_checkpoint(
    ckpt_dir: str,
    device: str,
) -> Optional[dict]:
    """Try to load a checkpoint from ckpt_dir.  Returns None if not found."""
    path = checkpoint_path(ckpt_dir, "ckpt.pt")
    if os.path.exists(path):
        print(f"Found checkpoint: {path}")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        return ckpt
    return None


# ---------------------------------------------------------------------------
# Mode A — Pretrain setup
# ---------------------------------------------------------------------------

def setup_pretrain(cfg: dict, device: str, device_type: str) -> tuple:
    """Initialize Minion model, optimizer, tokenizer for Mode A."""
    from model import Minion, MinionConfig

    # Load tokenizer
    tokenizer = _load_tokenizer(cfg)

    # Override vocab_size from tokenizer if present
    if tokenizer is not None and hasattr(tokenizer, "get_vocab_size"):
        cfg["vocab_size"] = tokenizer.get_vocab_size()
    elif tokenizer is not None and hasattr(tokenizer, "__len__"):
        cfg["vocab_size"] = len(tokenizer)

    model_cfg = MinionConfig(
        vocab_size              = cfg["vocab_size"],
        block_size              = cfg["block_size"],
        n_layer                 = cfg["n_layer"],
        n_head                  = cfg["n_head"],
        n_embd                  = cfg["n_embd"],
        n_kv_heads              = cfg["n_kv_heads"],
        rope_theta              = cfg["rope_theta"],
        rope_scaling            = cfg.get("rope_scaling"),
        swiglu_intermediate_dim = cfg.get("swiglu_intermediate_dim"),
        dropout                 = cfg["dropout"],
        mode                    = "pretrain",
    )
    model = Minion(model_cfg).to(device)

    if cfg.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()

    optimizer = model.configure_optimizers(
        weight_decay  = cfg["weight_decay"],
        learning_rate = cfg["learning_rate"],
        betas         = (cfg["beta1"], cfg["beta2"]),
        device_type   = device_type,
        use_8bit      = cfg.get("use_8bit_adam", True),
        use_lion      = cfg.get("use_lion", False),
    )

    return model, optimizer, tokenizer


# ---------------------------------------------------------------------------
# Mode B — QLoRA fine-tune setup
# ---------------------------------------------------------------------------

def setup_qlora(cfg: dict, device: str, device_type: str) -> tuple:
    """Initialize a quantized HF model + LoRA adapters for Mode B.

    NOTE: This is fine-tuning, not pretraining.  The base model weights
    are frozen in 4-bit NF4; only LoRA adapter parameters are trained.
    Training a model from scratch at GPT2-XL scale on free-tier Colab
    is not feasible (would require many GPU-days of continuous compute).
    QLoRA lets you adapt a powerful pretrained model with <1% of its parameters.
    """
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    base_model   = cfg["base_model_name"]
    compute_dtype = {
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
        "float32":  torch.float32,
    }[cfg.get("bnb_compute_dtype", "float16")]

    print(f"Loading base model '{base_model}' in 4-bit NF4 ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit             = True,
        bnb_4bit_quant_type      = cfg.get("quantization", "nf4"),
        bnb_4bit_compute_dtype   = compute_dtype,
        bnb_4bit_use_double_quant= True,   # nested quantization for extra VRAM saving
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_config,
        device_map          = "auto",
        torch_dtype         = compute_dtype,
    )

    # Enable gradient checkpointing on the HF model
    if cfg.get("gradient_checkpointing", True):
        hf_model.gradient_checkpointing_enable()

    # Prepare for k-bit training (freezes base weights, casts layer norms to fp32)
    hf_model = prepare_model_for_kbit_training(hf_model)

    lora_config = LoraConfig(
        task_type       = TaskType.CAUSAL_LM,
        r               = cfg.get("lora_rank", 16),
        lora_alpha      = cfg.get("lora_alpha", 32),
        lora_dropout    = cfg.get("lora_dropout", 0.05),
        target_modules  = cfg.get("lora_target_modules", ["c_attn", "c_proj"]),
        bias            = "none",
    )
    model = get_peft_model(hf_model, lora_config)
    model.print_trainable_parameters()

    # Tokenizer for Mode B MUST match the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})

    # Optimizer: use paged AdamW if available for further VRAM savings
    try:
        optimizer = bnb.optim.PagedAdam8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["learning_rate"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        print("Optimizer: Paged 8-bit AdamW (bitsandbytes)")
    except AttributeError:
        optimizer = bnb.optim.Adam8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg["learning_rate"],
            betas=(cfg["beta1"], cfg["beta2"]),
        )
        print("Optimizer: 8-bit AdamW (bitsandbytes)")

    # Wrap HF model so it exposes the same (logits, loss) interface as Minion
    model = _HFModelWrapper(model)
    return model, optimizer, tokenizer


class _HFModelWrapper(torch.nn.Module):
    """Thin adapter so HF CausalLM models expose Minion's (logits, loss) interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        if targets is not None:
            out = self.model(input_ids=idx, labels=targets)
            return out.logits, out.loss
        else:
            out = self.model(input_ids=idx)
            return out.logits[:, [-1], :], None

    def state_dict(self, **kwargs):
        return self.model.state_dict(**kwargs)

    def load_state_dict(self, sd, **kwargs):
        return self.model.load_state_dict(sd, **kwargs)

    def parameters(self, **kwargs):
        return self.model.parameters(**kwargs)

    def train(self, mode=True):
        return self.model.train(mode)

    def eval(self):
        return self.model.eval()


# ---------------------------------------------------------------------------
# Tokenizer loading helper
# ---------------------------------------------------------------------------

def _load_tokenizer(cfg: dict):
    """Load tokenizer from tokenizer_path or return None.

    For memmap mode, a tokenizer is NOT required — the .bin files are already
    tokenized.  vocab_size is read from the config directly.
    Only streaming mode needs a live tokenizer to encode documents on-the-fly.
    """
    tok_path = cfg.get("tokenizer_path", None)
    if not tok_path:
        return None

    tok_json = os.path.join(tok_path, "tokenizer.json")

    # Path A: our custom HF tokenizers .json
    if os.path.exists(tok_json):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_json)
        print(f"Loaded custom tokenizer: {tok_json}  (vocab={tok.get_vocab_size()})")
        return tok

    # Path B: a saved HF transformers tokenizer directory (has tokenizer_config.json)
    hf_config = os.path.join(tok_path, "tokenizer_config.json")
    if os.path.exists(hf_config):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tok_path)
        print(f"Loaded HF tokenizer: {tok_path}")
        return tok

    # Nothing found — fine for memmap mode, fatal for streaming mode
    if cfg.get("data_mode", "memmap") == "streaming":
        raise FileNotFoundError(
            f"Streaming mode requires a tokenizer but none was found at '{tok_path}'.\n"
            "Run: python tokenizer_train.py --mode train ..."
        )
    print(
        f"No tokenizer at '{tok_path}' — OK for memmap mode "
        "(vocab_size taken from config)."
    )
    return None


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model,
    data_loader,
    cfg: dict,
    ctx,
) -> dict[str, float]:
    """Estimate loss on train and val splits (eval_iters batches each)."""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg["eval_iters"])
        for k in range(cfg["eval_iters"]):
            X, Y = data_loader.get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Main train() function
# ---------------------------------------------------------------------------

def train(config_path: Optional[str] = None, **overrides) -> None:
    """Entry point callable both from CLI and from a Colab notebook cell.

    Args:
        config_path: Path to a YAML config file.
        **overrides: Key-value pairs that override config file values.

    Example (Colab cell):
        from train import train
        train("config/minion_pretrain_colab.yaml", wandb_log=False)
    """
    session_start = time.time()

    # ---- Config ----
    cfg = load_config(config_path, overrides)
    mode = cfg["mode"]

    # ---- Device ----
    device      = cfg.get("device", "cuda")
    # Auto-fallback: if CUDA requested but not available, switch to CPU
    if "cuda" in device and not torch.cuda.is_available():
        print("CUDA not available — falling back to device=cpu")
        device = "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"

    # ---- GPU properties (never hardcode assumptions) ----
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        total_vram_gb = props.total_memory / 1024 ** 3
        print(f"GPU: {props.name}  |  VRAM: {total_vram_gb:.1f} GB  "
              f"|  SM count: {props.multi_processor_count}  "
              f"|  Compute: {props.major}.{props.minor}")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("No CUDA GPU available — running on CPU (very slow, use Colab for real training)")

    # ---- Precision ----
    dtype_name, ptdtype, use_amp = resolve_dtype(cfg)
    print(f"Precision: {dtype_name}  |  AMP: {use_amp}")
    # autocast: only use CUDA backend when CUDA is actually available
    _use_amp_ctx = use_amp and torch.cuda.is_available() and device_type == "cuda"
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=ptdtype)
        if _use_amp_ctx
        else nullcontext()
    )
    # GradScaler: use new API (torch.amp) to avoid FutureWarning in PyTorch >= 2.4
    _scaler_enabled = dtype_name == "float16" and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=_scaler_enabled)
    except TypeError:
        # Older PyTorch fallback
        scaler = torch.cuda.amp.GradScaler(enabled=_scaler_enabled)

    # ---- Reproducibility ----
    torch.manual_seed(cfg["seed"])
    # TF32: use new API if available (PyTorch >= 2.9), fall back to old for compat
    try:
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision  = 'tf32'
    except AttributeError:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True

    # ---- Model + optimizer ----
    print(f"\nMode: {mode}")
    if mode == "pretrain":
        model, optimizer, tokenizer = setup_pretrain(cfg, device, device_type)
    elif mode == "qlora_finetune":
        model, optimizer, tokenizer = setup_qlora(cfg, device, device_type)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # ---- Data loader ----
    cfg["device"]      = device
    cfg["device_type"] = device_type

    # EOS token id for streaming mode
    if tokenizer is not None:
        if hasattr(tokenizer, "token_to_id"):
            cfg.setdefault("eos_token_id", tokenizer.token_to_id("<|endoftext|>") or 1)
        elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            cfg.setdefault("eos_token_id", tokenizer.eos_token_id)
        else:
            cfg.setdefault("eos_token_id", 1)

    from data import make_data_loader
    data_loader = make_data_loader(cfg, tokenizer)

    # ---- Resume from checkpoint ----
    iter_num       = 0
    best_val_loss  = float("inf")
    tokens_consumed = 0

    ckpt = load_checkpoint(cfg["ckpt_dir"], device)
    if ckpt is None:
        # Also check local out_dir
        ckpt = load_checkpoint(cfg.get("out_dir", "out"), device)

    if ckpt is not None:
        # Restore model weights
        sd = ckpt["model"]
        # Strip compiled-model prefix if present
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model_obj = model.model if isinstance(model, _HFModelWrapper) else model
        model_obj.load_state_dict(sd, strict=False)
        # Restore optimizer
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            print(f"Warning: could not restore optimizer state: {e}")
        iter_num        = ckpt.get("iter_num", 0)
        best_val_loss   = ckpt.get("best_val_loss", float("inf"))
        tokens_consumed = ckpt.get("tokens_consumed", 0)
        print(f"Resumed from iter {iter_num} (best_val_loss={best_val_loss:.4f})")

    # ---- torch.compile ----
    # Note: torch.compile compilation is ephemeral on Colab — the compiled
    # kernel cache is stored in /tmp and lost on disconnect.  Each new session
    # will recompile (~1–2 min for a 125M model).  Disable with compile=False
    # if compilation overhead is unacceptable for short sessions.
    raw_model = model
    if cfg.get("compile", True) and device_type == "cuda":
        print("Compiling model with torch.compile() ... (~1 min, cache is lost on Colab restart)")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without compilation")

    # ---- W&B ----
    if cfg.get("wandb_log", False):
        import wandb
        wandb.init(project=cfg["wandb_project"], name=cfg["wandb_run_name"], config=cfg)

    # ---- Training loop ----
    tokens_per_iter = (
        cfg["gradient_accumulation_steps"] * cfg["batch_size"] * cfg["block_size"]
    )
    print(f"\nTokens per iteration: {tokens_per_iter:,}")
    print(f"Effective batch size: {cfg['gradient_accumulation_steps'] * cfg['batch_size']}")
    print(f"Max iters: {cfg['max_iters']:,}")
    print(f"Checkpoint dir: {cfg['ckpt_dir']}\n")

    model.train()
    X, Y = data_loader.get_batch("train")
    t0 = time.time()
    running_mfu = -1.0
    local_iter  = 0

    while True:
        # ---- LR update ----
        lr = get_lr(iter_num, cfg) if cfg.get("decay_lr", True) else cfg["learning_rate"]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- Eval + checkpoint ----
        if iter_num % cfg["eval_interval"] == 0:
            losses = estimate_loss(model, data_loader, cfg, ctx)
            vram_gb = log_vram(device)
            elapsed_h = (time.time() - session_start) / 3600

            print(
                f"step {iter_num:6d}: "
                f"train {losses['train']:.4f}  val {losses['val']:.4f}  "
                f"lr {lr:.2e}  vram {vram_gb:.2f}GB  "
                f"elapsed {elapsed_h:.2f}h"
            )
            check_session_time(session_start, cfg)

            is_best = losses["val"] < best_val_loss
            if is_best:
                best_val_loss = losses["val"]

            if cfg.get("always_save_checkpoint", True) or is_best:
                if iter_num > 0:
                    save_checkpoint(
                        raw_model     = raw_model.model if isinstance(raw_model, _HFModelWrapper) else raw_model,
                        optimizer     = optimizer,
                        cfg           = cfg,
                        iter_num      = iter_num,
                        best_val_loss = best_val_loss,
                        tokens_consumed = getattr(data_loader, "tokens_consumed", 0),
                        is_best       = is_best,
                    )

            if cfg.get("wandb_log", False):
                import wandb
                wandb.log({
                    "iter":        iter_num,
                    "train/loss":  losses["train"],
                    "val/loss":    losses["val"],
                    "val/ppl":     math.exp(min(losses["val"], 20)),
                    "lr":          lr,
                    "mfu":         running_mfu * 100,
                    "vram_gb":     vram_gb,
                    "elapsed_h":   elapsed_h,
                })

        if iter_num == 0 and cfg.get("eval_only", False):
            break

        # ---- Forward + backward with gradient accumulation ----
        for micro_step in range(cfg["gradient_accumulation_steps"]):
            with ctx:
                _, loss = model(X, Y)
                loss = loss / cfg["gradient_accumulation_steps"]
            scaler.scale(loss).backward()
            X, Y = data_loader.get_batch("train")

        # ---- Gradient clipping ----
        if cfg.get("grad_clip", 1.0) > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in (raw_model.model if isinstance(raw_model, _HFModelWrapper) else raw_model).parameters()
                 if p.requires_grad],
                cfg["grad_clip"],
            )

        # ---- Optimizer step ----
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ---- Logging ----
        t1   = time.time()
        dt   = t1 - t0
        t0   = t1

        if iter_num % cfg["log_interval"] == 0:
            lossf = loss.item() * cfg["gradient_accumulation_steps"]
            if local_iter >= 5:
                model_obj = raw_model.model if isinstance(raw_model, _HFModelWrapper) else raw_model
                if hasattr(model_obj, "estimate_mfu"):
                    mfu = model_obj.estimate_mfu(
                        cfg["batch_size"] * cfg["gradient_accumulation_steps"], dt
                    )
                    running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
            vram_gb = log_vram(device)
            print(
                f"iter {iter_num:5d}: loss {lossf:.4f}  "
                f"{dt*1000:.1f}ms/iter  mfu {running_mfu*100:.1f}%  "
                f"vram {vram_gb:.2f}GB"
            )

        iter_num  += 1
        local_iter += 1

        if iter_num >= cfg["max_iters"]:
            print(f"Training complete at iter {iter_num}.")
            break

    # Final checkpoint
    save_checkpoint(
        raw_model     = raw_model.model if isinstance(raw_model, _HFModelWrapper) else raw_model,
        optimizer     = optimizer,
        cfg           = cfg,
        iter_num      = iter_num,
        best_val_loss = best_val_loss,
        tokens_consumed = getattr(data_loader, "tokens_consumed", 0),
        is_best       = False,
    )

    if cfg.get("wandb_log", False):
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Minion AI (Mode A: pretrain | Mode B: qlora_finetune)"
    )
    parser.add_argument("config", nargs="?", default=None, help="Path to YAML config file")
    parser.add_argument("--mode",         default=None)
    parser.add_argument("--max_iters",    type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--grad_accum",   type=int,   default=None, dest="gradient_accumulation_steps")
    parser.add_argument("--lr",           type=float, default=None, dest="learning_rate")
    parser.add_argument("--ckpt_dir",     default=None)
    parser.add_argument("--out_dir",      default=None)
    parser.add_argument("--wandb_log",    action="store_true", default=None)
    parser.add_argument("--compile",      type=lambda x: x.lower() != "false", default=None)
    parser.add_argument("--device",       default=None)
    parser.add_argument("--data_mode",    default=None)
    parser.add_argument("--n_layer",      type=int, default=None)
    parser.add_argument("--n_head",       type=int, default=None)
    parser.add_argument("--n_embd",       type=int, default=None)
    parser.add_argument("--block_size",   type=int, default=None)
    parser.add_argument("--eval_only",    action="store_true", default=None)

    args, unknown = parser.parse_known_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}

    # Parse any additional --key=value flags
    for arg in unknown:
        if arg.startswith("--") and "=" in arg:
            key, val = arg[2:].split("=", 1)
            try:
                from ast import literal_eval
                val = literal_eval(val)
            except (ValueError, SyntaxError):
                pass
            overrides[key] = val

    train(args.config, **overrides)


if __name__ == "__main__":
    main()
