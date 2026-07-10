"""
Minion AI — inference/server.py
Tiny FastAPI local serving endpoint for the RTX 3050.

Starts a local HTTP server on localhost:8000 with streamed generation.
No authentication, no rate limiting — local use only.

Usage:
    python inference/server.py --checkpoint out/ckpt.pt

    # Then query it:
    curl -X POST http://localhost:8000/generate \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "Hello world", "max_new_tokens": 100}'

    # Streamed response:
    curl -X POST http://localhost:8000/generate/stream \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "The future of", "max_new_tokens": 200}'

Expected VRAM usage on RTX 3050 (4 GB):
    125M model (4-bit):  ~0.5 GB model + ~0.5 GB overhead  → ~1 GB total
    350M model (4-bit):  ~1.0 GB model + ~0.5 GB overhead  → ~1.5 GB total
    Leaves plenty of headroom for long-context generation.

Expected throughput on RTX 3050:
    125M: ~60–90 tokens/sec
    350M: ~25–40 tokens/sec
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import torch

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt:         str
    max_new_tokens: int   = 100
    temperature:    float = 0.8
    top_k:          int   = 50
    top_p:          float = 0.95
    seed:           Optional[int] = None


class GenerateResponse(BaseModel):
    prompt:      str
    text:        str
    tokens_sec:  float
    vram_mb:     float


# ---------------------------------------------------------------------------
# Global model state (loaded once on startup)
# ---------------------------------------------------------------------------

_model    = None
_tokenizer = None
_cfg      = None
_device   = "cuda"


def _get_model():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model, _tokenizer, _cfg


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title        = "Minion AI — Local Inference Server",
    description  = "Local generation endpoint for RTX 3050. "
                   "No internet required after model load.",
    version      = "1.0.0",
)


@app.get("/health")
async def health():
    """Check if the model is loaded and ready."""
    if _model is None:
        return {"status": "loading", "model": None}

    vram_mb = (
        torch.cuda.memory_allocated(_device) / 1024 ** 2
        if torch.cuda.is_available() else 0.0
    )
    return {
        "status":   "ready",
        "device":   _device,
        "vram_mb":  round(vram_mb, 1),
        "block_size": _cfg.get("block_size", 1024),
        "vocab_size": _cfg.get("vocab_size", 32768),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text from a prompt. Returns full completion in one response."""
    model, tokenizer, cfg = _get_model()

    if req.seed is not None:
        torch.manual_seed(req.seed)

    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    from inference.generate import generate_text, encode

    # Validate prompt length
    prompt_ids = encode(tokenizer, req.prompt)
    block_size = cfg.get("block_size", 1024)
    if len(prompt_ids) >= block_size:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long ({len(prompt_ids)} tokens, max {block_size - 1})"
        )

    t0 = time.perf_counter()

    # Run in a thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        None,
        lambda: generate_text(
            model          = model,
            tokenizer      = tokenizer,
            cfg            = cfg,
            prompt         = req.prompt,
            max_new_tokens = req.max_new_tokens,
            temperature    = req.temperature,
            top_k          = req.top_k,
            top_p          = req.top_p,
            device         = _device,
        )
    )

    dt = time.perf_counter() - t0
    vram_mb = (
        torch.cuda.memory_allocated(_device) / 1024 ** 2
        if torch.cuda.is_available() else 0.0
    )

    return GenerateResponse(
        prompt     = req.prompt,
        text       = text,
        tokens_sec = round(req.max_new_tokens / dt, 1),
        vram_mb    = round(vram_mb, 1),
    )


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Generate text with server-sent events (token-by-token streaming).

    Streams each new token as it is generated.  The client receives a
    'data: <token>\\n\\n' line for each token, followed by 'data: [DONE]'.
    """
    model, tokenizer, cfg = _get_model()

    if req.seed is not None:
        torch.manual_seed(req.seed)

    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    from inference.generate import encode, decode

    prompt_ids = encode(tokenizer, req.prompt)
    block_size = cfg.get("block_size", 1024)
    if len(prompt_ids) >= block_size:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long ({len(prompt_ids)} tokens, max {block_size - 1})"
        )

    async def token_stream() -> AsyncGenerator[str, None]:
        import torch.nn.functional as F

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=_device)

        for _ in range(req.max_new_tokens):
            idx_cond = idx[:, -block_size:]

            with torch.no_grad():
                logits, _ = model(idx_cond)

            logits = logits[:, -1, :] / max(req.temperature, 1e-6)

            if req.top_k > 0:
                v, _ = torch.topk(logits, min(req.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if req.top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cp - F.softmax(sl, dim=-1) > req.top_p] = float("-inf")
                logits = torch.scatter(logits, 1, si, sl)

            probs    = F.softmax(logits, dim=-1)
            tok      = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, tok], dim=1)

            token_text = decode(tokenizer, [tok.item()])
            yield f"data: {token_text}\n\n"

            # Yield control to event loop between tokens
            await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minion AI — local FastAPI server for RTX 3050"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--device",     default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--host",       default="127.0.0.1")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--no-quantize",action="store_true",
                        help="Load in fp16 instead of 4-bit (more VRAM)")
    args = parser.parse_args()

    global _model, _tokenizer, _cfg, _device
    _device = args.device

    if _device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        _device = "cpu"

    # Load model at startup
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    from inference.quantize import load_quantized_model

    _model, _tokenizer, _cfg = load_quantized_model(
        ckpt_path = args.checkpoint,
        quantize  = not args.no_quantize,
        device    = _device,
    )

    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated(_device) / 1024 ** 2
        vram_total = torch.cuda.get_device_properties(_device).total_memory / 1024 ** 2
        print(f"VRAM: {vram_mb:.0f} / {vram_total:.0f} MB used after model load")

    print(f"\nMinion AI server running at http://{args.host}:{args.port}")
    print(f"  POST /generate       → full completion")
    print(f"  POST /generate/stream → streamed completion")
    print(f"  GET  /health         → status check\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
