"""
Minion AI — model.py
Full definition of the modernized GPT language model.

Changes vs. nanoGPT:
  - RMSNorm  (no mean-subtract, no bias) replaces LayerNorm
  - RoPE     applied to Q/K per layer   replaces learned absolute wpe
  - GQA      (n_kv_heads < n_head)      replaces full MHA
  - SwiGLU   (3 matrices, 2.67× expand) replaces 4×GELU MLP
  - F.scaled_dot_product_attention       replaces manual attention math
  - Configurable vocab_size              removes GPT-2 BPE dependency
  - torch.utils.checkpoint               for optional gradient checkpointing

References:
  RoPE:    https://arxiv.org/abs/2104.09864
  GQA:     https://arxiv.org/abs/2305.13245
  SwiGLU:  https://arxiv.org/abs/2002.05202
  RMSNorm: https://arxiv.org/abs/1910.07467
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Compared to LayerNorm: removes the mean-subtract step and the bias term.
    ~10% faster, identical quality at scale. No bias saves a small amount of
    memory/params in every block (2 × n_embd per layer).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for stability, then back to input dtype
        return self._norm(x.float()).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    scaling: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cosine/sine frequency tables.

    Args:
        head_dim:    Per-head embedding dimension (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        theta:       RoPE base frequency. Default 10 000 (original paper).
                     Increase to e.g. 500 000 for long-context extension
                     without retraining (NTK-aware approach: higher theta
                     spreads frequencies so existing rotations still work
                     at longer positions).
        scaling:     Optional linear position scaling for context extension.
                     position = token_index / scaling.  E.g. scaling=2.0
                     doubles effective context.  For best quality at very
                     long context, prefer increasing theta (NTK-aware)
                     over linear scaling.

    Returns:
        (cos, sin) each of shape (max_seq_len, head_dim // 2)
    """
    # Frequencies: θ_i = 1 / theta^(2i / head_dim), for i in [0, head_dim/2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    if scaling is not None:
        # Tradeoff: linear scaling is simple but can degrade quality at very
        # long context vs. NTK-aware (different theta per frequency band).
        t = t / scaling
    freqs = torch.outer(t, inv_freq)   # (seq_len, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE rotation in-place to Q or K.

    Args:
        x:       (B, n_heads, T, head_dim)
        cos/sin: (T, head_dim//2) — already sliced to current seq length

    Returns:
        Rotated tensor, same shape as x.

    Implementation uses the split-half form (equivalent to complex rotation):
        [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    where x1 = x[..., :d/2], x2 = x[..., d/2:].
    """
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    # Broadcast: (T, hd//2) -> (1, 1, T, hd//2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """Causal self-attention with Grouped Query Attention and RoPE.

    GQA uses n_kv_heads < n_head key/value heads.  KV is expanded
    (repeat_interleave) to match n_head before SDPA.

    Memory tradeoff:
      n_kv_heads = n_head  → full MHA, no KV savings.
      n_kv_heads = 1       → Multi-Query Attention (MQA), maximum KV savings
                             but slight quality loss at small model scale.
      n_kv_heads = n_head//4  → good balance (LLaMA-3 default).

    KV cache size scales as n_kv_heads / n_head relative to full MHA,
    so n_head//4 gives 4× smaller KV cache — significant on a 4 GB GPU.
    """

    def __init__(self, config: "MinionConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "n_embd must be divisible by n_head"
        assert config.n_head % config.n_kv_heads == 0, \
            "n_head must be divisible by n_kv_heads (GQA requirement)"

        self.n_head    = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.n_rep     = config.n_head // config.n_kv_heads   # Q heads per KV head
        self.head_dim  = config.n_embd // config.n_head
        self.dropout   = config.dropout

        # Q projects to n_head * head_dim; KV each project to n_kv_heads * head_dim
        self.q_proj = nn.Linear(config.n_embd, config.n_head    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # Project and reshape
        q = self.q_proj(x).view(B, T, self.n_head,     self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, nkv, T, hd)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, nkv, T, hd)

        # Apply RoPE to Q and K (position-dependent rotation; V is not rotated)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: expand K/V heads to match Q heads via repeat_interleave
        # (B, n_kv_heads, T, hd) → (B, n_heads, T, hd)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Flash Attention / memory-efficient SDPA backend — no manual masking
        # is_causal=True handles the causal mask automatically
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # Reassemble heads and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y))


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """Gated feedforward block: down(silu(gate(x)) * up(x)).

    Uses three linear layers vs. nanoGPT's two (fc + proj).  The intermediate
    dimension is reduced to ~2/3 × 4 × n_embd so total parameter count stays
    comparable to a 4×GELU MLP despite the extra gate matrix.

    Tradeoff: SwiGLU is empirically better than GELU per token at the same
    parameter count (LLaMA, PaLM, etc. all use it), at no throughput cost.
    The 2/3 factor is from the SwiGLU paper; some implementations round to
    the nearest multiple of 64 or 256 for GPU alignment — we do the former.
    """

    def __init__(self, config: "MinionConfig"):
        super().__init__()
        if config.swiglu_intermediate_dim is not None:
            # Explicit override: useful when you want exact param count control
            intermediate = config.swiglu_intermediate_dim
        else:
            # Default: 2/3 of 4 × n_embd, rounded up to nearest multiple of 64
            intermediate = int(2 / 3 * 4 * config.n_embd)
            intermediate = ((intermediate + 63) // 64) * 64

        self.gate_proj = nn.Linear(config.n_embd, intermediate, bias=False)
        self.up_proj   = nn.Linear(config.n_embd, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate,  config.n_embd, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU (swish) gate × up projection, then project back down
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Single transformer block: pre-norm, GQA attention, pre-norm, SwiGLU FFN."""

    def __init__(self, config: "MinionConfig"):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn      = GroupedQueryAttention(config)
        self.ffn_norm  = RMSNorm(config.n_embd)
        self.ffn       = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MinionConfig:
    # ---- Vocabulary & sequence ----
    vocab_size: int = 32768   # 32k default: smaller embedding/output matrix vs. GPT-2's 50k.
                               # Saves ~25 MB on a 4 GB GPU.  Larger = richer subword coverage.
    block_size: int = 1024    # Context length. RoPE allows extending later (NTK/linear scaling)
                               # without retraining from scratch.

    # ---- Model dimensions ----
    n_layer: int = 12
    n_head:  int = 12
    n_embd:  int = 768

    # ---- GQA ----
    # n_kv_heads = n_head // 4 gives 4× smaller KV cache vs full MHA.
    # Tradeoff: lower n_kv_heads → less KV memory, marginally lower quality.
    # Set n_kv_heads = n_head to get full MHA (no memory benefit, no quality loss).
    n_kv_heads: int = 3

    # ---- RoPE ----
    rope_theta:   float = 10_000.0   # Base frequency. Raise to e.g. 500_000 for long context.
    rope_scaling: Optional[float] = None  # Linear scaling factor; None = disabled.

    # ---- SwiGLU ----
    # None = auto (2/3 × 4 × n_embd rounded to 64).  Set explicitly for exact param budgets.
    swiglu_intermediate_dim: Optional[int] = None

    # ---- Regularization ----
    dropout: float = 0.0   # 0 for pretraining; 0.1+ can help for fine-tuning small datasets.

    # ---- Training mode (informational; used by train.py) ----
    mode: str = "pretrain"   # "pretrain" | "qlora_finetune"


# ---------------------------------------------------------------------------
# Top-level Model
# ---------------------------------------------------------------------------

class Minion(nn.Module):
    """Minion AI language model.

    Key design choices vs. nanoGPT:
      - No wpe (learned positional embedding) — positions handled by RoPE in attention.
      - Weight tying: wte.weight == lm_head.weight (saves vocab_size × n_embd params).
      - RoPE cos/sin tables registered as non-parameter buffers; auto-moves with .to(device).
      - Gradient checkpointing: call enable_gradient_checkpointing() before training.
    """

    def __init__(self, config: MinionConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.n_head % config.n_kv_heads == 0, (
            f"n_head ({config.n_head}) must be divisible by n_kv_heads ({config.n_kv_heads})"
        )
        self.config = config
        self._gradient_checkpointing = False

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            norm = RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (lm_head shares wte's embedding matrix)
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute RoPE frequency tables as non-trainable buffers
        head_dim = config.n_embd // config.n_head
        cos, sin = precompute_rope_freqs(
            head_dim, config.block_size, config.rope_theta, config.rope_scaling
        )
        self.register_buffer("rope_cos", cos)  # (block_size, head_dim//2)
        self.register_buffer("rope_sin", sin)

        # Weight initialization
        self.apply(self._init_weights)
        # Scaled init for residual projections: std = 0.02 / sqrt(2 * n_layer)
        # This prevents variance blow-up through the residual stream (GPT-2 paper).
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = self.get_num_params()
        print(f"Minion: {n_params / 1e6:.2f}M parameters")

    def get_num_params(self) -> int:
        """Total parameter count.  lm_head weight is tied to wte, counted once."""
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self) -> None:
        """Activate gradient checkpointing on all transformer blocks.

        Recomputes block activations during the backward pass instead of
        caching them.  Reduces activation memory by roughly sqrt(n_layer)
        at the cost of ~33% extra compute.  Essential for large models on
        Colab's 16 GB budget.
        """
        self._gradient_checkpointing = True
        print(f"Gradient checkpointing enabled ({self.config.n_layer} blocks)")

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Sequence length {T} > block_size {self.config.block_size}"
        )

        # Token embeddings only — no positional embedding (RoPE handles positions)
        x = self.transformer.drop(self.transformer.wte(idx))  # (B, T, n_embd)

        # Slice RoPE tables to current sequence length (saves broadcast overhead)
        cos = self.rope_cos[:T]  # (T, head_dim//2)
        sin = self.rope_sin[:T]

        for block in self.transformer.h:
            if self._gradient_checkpointing and self.training:
                # Recompute activations during backward; saves ~sqrt(n_layer)× memory.
                # use_reentrant=False is the recommended modern API.
                x = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, use_reentrant=False
                )
            else:
                x = block(x, cos, sin)

        x = self.transformer.norm(x)

        if targets is not None:
            # Training: compute loss over all positions
            logits = self.lm_head(x)                                            # (B, T, V)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only forward the last position (avoids computing whole vocab over T)
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, V)
            loss   = None

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
        use_8bit: bool = True,
        use_lion: bool = False,
    ) -> torch.optim.Optimizer:
        """Return configured optimizer (8-bit AdamW, Lion, or fp32 AdamW fallback).

        8-bit AdamW (bitsandbytes):
            Quantizes the two Adam moment buffers from fp32 to int8.
            Saves ~50% optimizer state memory — significant for large models
            where optimizer state can exceed model size.

        Lion (bitsandbytes):
            Only one momentum buffer (vs. Adam's two).  Lower memory than Adam,
            but needs lower LR (~ 1/3 of AdamW's LR) and higher weight decay.
            Less battle-tested than AdamW for language model pretraining.

        Decay rule: weight decay on 2D+ parameters (weights), not 1D (norms, biases).
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params,   "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        n_decay   = sum(p.numel() for p in decay_params)
        n_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"  decayed params:   {n_decay:,}  ({len(decay_params)} tensors)")
        print(f"  no-decay params:  {n_nodecay:,}  ({len(nodecay_params)} tensors)")

        if use_lion and device_type == "cuda":
            try:
                from bitsandbytes.optim import Lion as BnbLion
                opt = BnbLion(optim_groups, lr=learning_rate, betas=betas)
                print("Optimizer: Lion (bitsandbytes)")
                return opt
            except (ImportError, AttributeError):
                print("Lion not found in bitsandbytes, falling back to AdamW")

        if use_8bit and device_type == "cuda":
            try:
                import bitsandbytes as bnb
                opt = bnb.optim.Adam8bit(optim_groups, lr=learning_rate, betas=betas)
                print("Optimizer: 8-bit AdamW (bitsandbytes) — ~50% optimizer memory saving")
                return opt
            except ImportError:
                print("bitsandbytes not installed, falling back to torch AdamW")

        # Fallback: standard PyTorch AdamW with fused CUDA kernel when available
        import inspect
        fused = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
            and device_type == "cuda"
        )
        opt = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas,
            **({"fused": True} if fused else {}),
        )
        print(f"Optimizer: torch AdamW (fused={fused})")
        return opt

    def estimate_mfu(self, tokens_per_iter: int, dt: float) -> float:
        """Estimate model flops utilization (MFU) relative to T4 fp16 peak (65 TFLOPS)."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_achieved  = flops_per_token * tokens_per_iter / dt
        return flops_achieved / 65e12   # T4 fp16 peak

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with temperature, top-k, and top-p (nucleus) sampling."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens once cumulative probability exceeds top_p
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, idx_next], dim=1)
        return idx
