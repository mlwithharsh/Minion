# Minion AI

A modernized, VRAM-efficient GPT training stack built on top of [nanoGPT](https://github.com/karpathy/nanoGPT).

**Train on Google Colab (T4, 16 GB) → Run locally on an RTX 3050 (4 GB)**

---

## What changed vs. nanoGPT (and why)

| Component | nanoGPT | Minion AI | Why |
|-----------|---------|-----------|-----|
| Positional encoding | Learned absolute `wpe` | **RoPE** (Rotary Position Embeddings) | No extra parameters; context can be extended via NTK/linear scaling without retraining |
| Normalization | LayerNorm (with bias) | **RMSNorm** (no bias, no mean-subtract) | ~10% faster; identical quality at scale (LLaMA, Mistral, Gemma all use it) |
| Attention | Manual MHA + causal mask | **GQA** + `F.scaled_dot_product_attention` | GQA shrinks KV cache 4× on default config; SDPA uses Flash Attention automatically |
| MLP | 4×GELU (2 matrices) | **SwiGLU** (2.67×, 3 matrices) | Better loss-per-token at same parameter count (LLaMA, PaLM, Mistral) |
| Tokenizer | Hardcoded GPT-2 BPE | **Custom BPE or Unigram** (32k default) | Smaller vocab = smaller embedding matrix = less VRAM on 4 GB inference GPU |
| Optimizer | fp32 AdamW | **8-bit AdamW** (bitsandbytes) | ~50% optimizer state memory saving — frees 2–4 GB on Colab T4 |
| Dataset | Static `.bin` memmap only | **Streaming** (HF datasets) + memmap | No full corpus download; training starts immediately on Colab |
| Config | Python `exec()` overrides | **YAML** (`config/*.yaml`) | Cleaner, type-checked, editor-friendly |
| Checkpoints | Local only | **Google Drive** + local | Survives Colab session disconnects |

---

## Two training modes

### Mode A — `pretrain` (default)
Train a Minion model fully from scratch with the modernized architecture.

- **Default config**: 12 layers, 12 heads (3 KV heads), 768 embed → **~125M parameters**
- **Wall-clock**: ~5k–10k steps per Colab session at the default config. Reach first-pass coherence in ~3–5 sessions.
- **VRAM budget on T4 (16 GB)**:
  - Model (fp16):         ~0.5 GB
  - Optimizer (8-bit):    ~0.5 GB
  - Activations + grad:   ~8–10 GB (with gradient checkpointing on)
  - Total:                ~10–12 GB → comfortable headroom

Scale to ~350M (24 layers, 1024 embed) if you have Colab Pro with an A100.

### Mode B — `qlora_finetune`
> **This is fine-tuning, not pretraining.** Load GPT-2 XL (1.558B params) in 4-bit NF4, freeze base weights, train LoRA adapters only.

Why not pretrain at 1.5B scale? Pretraining a 1.5B-param model to convergence requires hundreds of GPU-days. Free-tier Colab cannot do this. QLoRA lets you **adapt an already-converged model** to your domain in a few sessions.

- **VRAM on T4**: ~4–5 GB total (model at 4-bit + activations)
- **Trainable params**: ~20M LoRA adapters out of 1.558B total
- **Default base model**: `openai-community/gpt2-xl`

---

## Quickstart

### Step 1 — Train on Colab

Open the notebook in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/minion/blob/main/notebooks/minion_colab.ipynb)

Or manually:
1. Upload `notebooks/minion_colab.ipynb` to Colab
2. Runtime → Change runtime type → **GPU (T4)**
3. Run all cells (mount Drive, install deps, clone repo, train)

Checkpoints save automatically to `Google Drive/minion_ckpts/`.

### Step 2 — Smoke test (verify pipeline, < 60 seconds)

Before committing to a real run, verify the pipeline works:

```bash
# Prepare tiny Shakespeare data
python tokenizer_train.py --mode train \
  --corpus_path data/shakespeare_char/input.txt \
  --vocab_size 512 --algorithm bpe \
  --tokenizer_dir tokenizer_smoke

python tokenizer_train.py --mode prepare \
  --tokenizer_path tokenizer_smoke \
  --corpus_path data/shakespeare_char/input.txt \
  --data_dir data/smoke_prepared

# Run 10 training steps (~30 seconds on GPU, ~2 min on CPU)
python train.py config/smoke_test.yaml
```

### Step 3 — Real Mode A pretraining

```bash
# 1. Train a custom tokenizer from FineWeb-Edu (streaming, ~10 min)
python tokenizer_train.py \
  --mode train \
  --hf_dataset HuggingFaceFW/fineweb-edu \
  --hf_dataset_config sample-10BT \
  --vocab_size 32768 \
  --tokenizer_dir tokenizer

# 2. Start training (run in Colab via the notebook)
python train.py config/minion_pretrain_colab.yaml
```

### Step 4 — Sync checkpoint from Drive to local

```bash
# Install rclone (one time): https://rclone.org/install/
# Configure Drive remote (one time): rclone config

python inference/sync_checkpoint.py \
  --method rclone \
  --remote "gdrive:minion_ckpts" \
  --local_dir ./checkpoints

# OR with gdown (no rclone needed):
python inference/sync_checkpoint.py \
  --method gdown \
  --drive_folder_url "https://drive.google.com/drive/folders/YOUR_FOLDER_ID" \
  --local_dir ./checkpoints
```

### Step 5 — Run inference on RTX 3050

```bash
# Install inference deps (lighter than training deps)
pip install -r requirements-infer.txt

# CLI generation (4-bit quantized, ~60 tok/sec on 3050 for 125M)
python inference/generate.py \
  --checkpoint checkpoints/ckpt.pt \
  --prompt "The meaning of intelligence is" \
  --max_new_tokens 200 \
  --temperature 0.8

# Local API server (http://localhost:8000)
python inference/server.py --checkpoint checkpoints/ckpt.pt
```

---

## File structure

```
minion/
├── model.py              # Modernized GPT model (RoPE, GQA, RMSNorm, SwiGLU)
├── data.py               # Dual-mode data pipeline (streaming + memmap)
├── train.py              # Training loop (Mode A + B, Colab, Drive, VRAM logging)
├── tokenizer_train.py    # Custom BPE/Unigram tokenizer training
│
├── config/
│   ├── minion_pretrain_colab.yaml      # Mode A — 125M pretrain, T4 tuned
│   ├── minion_qlora_gpt2xl_colab.yaml  # Mode B — QLoRA GPT-2 XL
│   └── smoke_test.yaml                 # Tiny sanity check (< 1 min)
│
├── inference/
│   ├── generate.py         # CLI generation (4-bit quantized)
│   ├── server.py           # FastAPI local serving endpoint
│   ├── quantize.py         # 4-bit BnB load + GGUF export (optional)
│   └── sync_checkpoint.py  # Google Drive → local checkpoint sync
│
├── notebooks/
│   └── minion_colab.ipynb  # Runnable Colab training notebook
│
├── requirements-train.txt  # Colab training deps
└── requirements-infer.txt  # Local inference deps (lighter)
```

---

## Architecture details

### RoPE (Rotary Position Embeddings)
Applied to Q and K in every attention layer. No learned parameters — positions encoded as rotations in the complex plane. Extending context length only requires changing `rope_theta` or `rope_scaling` in the config; no retraining from scratch.

```yaml
rope_theta:   10000.0   # Default; increase to 500000 for long context
rope_scaling: null      # Linear scaling factor; null = disabled
```

### GQA (Grouped Query Attention)
`n_kv_heads = n_head // 4` by default (e.g. 12 → 3 KV heads). This gives a **4× smaller KV cache** vs. full MHA — critical for fitting long sequences on a 4 GB inference GPU.

**Tradeoff**: smaller `n_kv_heads` → lower KV memory, marginal quality loss at small model scale. Set `n_kv_heads = n_head` to recover full MHA if you have VRAM to spare.

### SwiGLU MLP
Three matrices (gate, up, down) vs. nanoGPT's two (fc, proj). Intermediate dim is `~2/3 × 4 × n_embd` (auto-rounded to 64) to keep parameter count comparable despite the extra matrix.

**Tradeoff**: SwiGLU is empirically better per-token than GELU at the same parameter count (used in LLaMA, PaLM, Mistral, Gemma).

### 8-bit AdamW
Quantizes the two Adam moment buffers from fp32 to int8. On a 125M model, saves ~500 MB of optimizer state. On a 350M model, saves ~1.5 GB.

```yaml
use_8bit_adam: true   # Recommended; disable only for debugging
use_lion:      false  # Lion: lower memory but needs ~3× lower LR
```

---

## Config reference

All training hyperparameters live in `config/*.yaml`. Override from CLI with `--key=value`:

```bash
python train.py config/minion_pretrain_colab.yaml --max_iters=1000 --wandb_log=true
```

Key parameters:

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `pretrain` | `pretrain` or `qlora_finetune` |
| `data_mode` | `streaming` | `streaming` or `memmap` |
| `n_layer` / `n_head` / `n_embd` | 12/12/768 | Model size (~125M) |
| `n_kv_heads` | 3 | GQA KV heads (≤ n_head) |
| `vocab_size` | 32768 | Tokenizer vocab size |
| `block_size` | 1024 | Context length |
| `gradient_checkpointing` | `true` | Saves ~4 GB activation memory |
| `gradient_accumulation_steps` | 8 | Effective batch = batch_size × accum |
| `use_8bit_adam` | `true` | 8-bit optimizer via bitsandbytes |
| `ckpt_dir` | `/content/drive/...` | Drive path for checkpoints |
| `session_limit_hours` | 11.5 | Colab session limit for warnings |

---

## VRAM and wall-clock expectations

### Mode A — 125M pretrain on Colab T4

| Component | VRAM |
|-----------|------|
| Model (fp16) | ~0.5 GB |
| Optimizer (8-bit AdamW) | ~0.5 GB |
| Activations + gradients | ~8–10 GB |
| **Total** | **~10–12 GB** |

- Throughput: ~80–120 tokens/sec on T4
- Steps per session (~11h): ~30k–40k steps
- Useful coherence: ~20k–50k steps (3–5 sessions)

### Mode B — GPT-2 XL QLoRA on Colab T4

| Component | VRAM |
|-----------|------|
| Base model (4-bit NF4) | ~800 MB |
| LoRA adapters (fp16) | ~20 MB |
| Activations (block_size=512) | ~2–3 GB |
| **Total** | **~4–5 GB** |

- Throughput: ~40–60 tokens/sec on T4
- Visible improvement after: 500–2000 steps

### Inference on RTX 3050 (4 GB)

| Model | Quantization | VRAM | Tokens/sec |
|-------|-------------|------|-----------|
| 125M  | 4-bit BnB   | ~0.5 GB | ~60–90 |
| 350M  | 4-bit BnB   | ~1.0 GB | ~25–40 |
| GPT-2 XL (Mode B) | 4-bit BnB | ~2 GB | ~15–25 |

---

## Troubleshooting

**`torch.compile` fails or is very slow on Colab**
The compiled kernel cache is stored in `/tmp` and lost on every Colab session restart. Each session will recompile (~1–2 min). Disable with `--compile=false` if startup time matters.

**bitsandbytes import error on Windows**
bitsandbytes does not have official Windows support. Use WSL2 or run training exclusively on Colab. Inference on Windows works via CPU or via a WSL2 CUDA environment.

**Streaming dataset is slow to start**
The first few batches may be slow as HuggingFace tokenizes incoming documents. After a warm-up period it reaches steady-state throughput. Use `--data_mode=memmap` with pre-tokenized `.bin` files if you need deterministic startup.

**OOM (out of memory) on Colab**
1. Reduce `batch_size` (try 4 or 2)
2. Reduce `block_size` (try 512)
3. Enable `gradient_checkpointing: true`
4. Disable `compile: false` (compilation can spike VRAM briefly)
5. Reduce model size (`n_layer`, `n_embd`)

---

## Acknowledgements

Built on top of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.
Architecture improvements from [LLaMA](https://arxiv.org/abs/2302.13971), [GQA](https://arxiv.org/abs/2305.13245), [RoPE](https://arxiv.org/abs/2104.09864), [SwiGLU](https://arxiv.org/abs/2002.05202), [QLoRA](https://arxiv.org/abs/2305.14314).
