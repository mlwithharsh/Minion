"""
Minion AI — data.py
Dual-mode data pipeline.

Mode 'streaming': HuggingFace datasets with streaming=True.
    - No full corpus download needed — starts yielding batches immediately.
    - Packed-sequence collator: concatenates tokenized documents and chunks
      to block_size (identical semantics to nanoGPT's .bin approach).
    - Saves tokens_consumed in checkpoint for approximate resume.

Mode 'memmap': numpy memmap from pre-tokenized .bin files.
    - Identical to nanoGPT's get_batch() — fast random-access, low overhead.
    - Requires running tokenizer_train.py + prepare first.
    - Preferred when the full dataset fits on disk (local runs, Colab Pro).

The choice is a config option (data_mode: streaming | memmap); both modes
expose the same get_batch(split) -> (x, y) interface to train.py.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Streaming mode
# ---------------------------------------------------------------------------

class StreamingDataset:
    """Packed-sequence dataset backed by a HuggingFace streaming iterator.

    Documents are tokenized on-the-fly, concatenated with an EOS separator,
    and chunked into fixed block_size+1 windows.  Each window yields an
    (x, y) pair where y = x shifted left by one (standard next-token LM target).

    Design notes:
        - EOS token between documents prevents the model from learning cross-
          document dependencies.  Set eos_token_id to your tokenizer's EOS.
        - The collator accumulates a rolling buffer and yields chunks lazily,
          so memory use is O(block_size) regardless of corpus size.
        - tokens_consumed tracks how many tokens have been yielded.  On
          checkpoint resume, the iterator is fast-forwarded by skipping
          documents until ~tokens_consumed tokens have been seen.  This is
          approximate (document boundaries != token boundaries) but close
          enough for practical purposes.
    """

    def __init__(
        self,
        hf_dataset_name: str,
        hf_dataset_config: Optional[str],
        hf_split: str,
        tokenizer,
        block_size: int,
        eos_token_id: int,
        text_column: str = "text",
        tokens_consumed: int = 0,
        cache_dir: Optional[str] = None,
    ):
        from datasets import load_dataset

        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.tokens_consumed = tokens_consumed

        print(f"Loading streaming dataset '{hf_dataset_name}' (split={hf_split})")
        self.ds = load_dataset(
            hf_dataset_name,
            hf_dataset_config,
            split=hf_split,
            streaming=True,
            cache_dir=cache_dir,
        )
        self._buf: list[int] = []
        self._iter: Iterator = iter(self.ds)

        # Fast-forward past already-consumed tokens on resume
        if tokens_consumed > 0:
            self._fast_forward(tokens_consumed)

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize a single document, appending EOS."""
        ids = self.tokenizer.encode(text)
        if hasattr(ids, "ids"):
            ids = ids.ids  # HF tokenizers returns an Encoding object
        return ids + [self.eos_token_id]

    def _fast_forward(self, target_tokens: int) -> None:
        """Skip documents until approximately target_tokens have been consumed."""
        seen = 0
        t0 = time.time()
        print(f"Resuming streaming dataset: fast-forwarding ~{target_tokens:,} tokens...")
        while seen < target_tokens:
            try:
                doc = next(self._iter)
                ids = self._tokenize(doc[self.text_column])
                seen += len(ids)
            except StopIteration:
                self._iter = iter(self.ds)
        elapsed = time.time() - t0
        print(f"Fast-forward complete ({seen:,} tokens skipped in {elapsed:.1f}s)")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (x, y) tensor pairs of length block_size indefinitely."""
        while True:
            # Fill buffer until we have at least block_size+1 tokens
            while len(self._buf) < self.block_size + 1:
                try:
                    doc = next(self._iter)
                except StopIteration:
                    # Restart iterator at epoch boundary
                    self._iter = iter(self.ds)
                    continue
                self._buf.extend(self._tokenize(doc[self.text_column]))

            # Yield one chunk
            chunk = self._buf[: self.block_size + 1]
            self._buf = self._buf[self.block_size + 1 :]
            self.tokens_consumed += self.block_size

            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:],  dtype=torch.long)
            yield x, y


class StreamingDataLoader:
    """Batched loader around StreamingDataset.

    Collects batch_size (x, y) pairs into a single (B, T) batch tensor.
    Supports train and val splits independently.
    """

    def __init__(
        self,
        train_dataset: StreamingDataset,
        val_dataset: StreamingDataset,
        batch_size: int,
        device: str,
        device_type: str,
    ):
        self.batch_size  = batch_size
        self.device      = device
        self.device_type = device_type
        self._iters = {
            "train": iter(train_dataset),
            "val":   iter(val_dataset),
        }
        self._datasets = {"train": train_dataset, "val": val_dataset}

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        it = self._iters[split]
        for _ in range(self.batch_size):
            x, y = next(it)
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs)  # (B, T)
        y = torch.stack(ys)
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    @property
    def tokens_consumed(self) -> int:
        return self._datasets["train"].tokens_consumed


# ---------------------------------------------------------------------------
# Memmap mode
# ---------------------------------------------------------------------------

class MemmapDataLoader:
    """Fast random-access loader from pre-tokenized .bin files.

    Identical semantics to nanoGPT's get_batch():
        data = np.memmap(path, dtype=np.uint16, mode='r')
        ix   = random indices
        x, y = stacked [i : i+block_size] windows

    Use this mode when you have a pre-tokenized .bin file and want:
        - Fast restart (no re-tokenization on resume)
        - Deterministic sampling (add seed for reproducibility)
        - Local development without internet access

    Create .bin files with tokenizer_train.py --mode prepare, or keep using
    the original nanoGPT data/*/prepare.py scripts for Shakespeare, OWT, etc.
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int,
        batch_size: int,
        device: str,
        device_type: str,
    ):
        self.block_size  = block_size
        self.batch_size  = batch_size
        self.device      = device
        self.device_type = device_type

        train_path = os.path.join(data_dir, "train.bin")
        val_path   = os.path.join(data_dir, "val.bin")

        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"train.bin not found in {data_dir}. "
                "Run: python tokenizer_train.py --mode prepare --data_dir <dir>  "
                "or use data_mode=streaming in your config."
            )

        self._paths = {"train": train_path, "val": val_path}
        print(f"Memmap data loader: {train_path}")

    def _load(self, split: str) -> np.ndarray:
        # Recreate memmap every call to avoid a memory leak (numpy bug workaround):
        # https://stackoverflow.com/a/61472122
        return np.memmap(self._paths[split], dtype=np.uint16, mode="r")

    def get_batch(self, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        data = self._load(split)
        ix   = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([
            torch.from_numpy(data[i     : i + self.block_size    ].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(data[i + 1 : i + self.block_size + 1].astype(np.int64))
            for i in ix
        ])
        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    @property
    def tokens_consumed(self) -> int:
        # Memmap is random-access; no sequential position to track
        return 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_data_loader(cfg: dict, tokenizer=None) -> MemmapDataLoader | StreamingDataLoader:
    """Create a data loader based on config.

    Config keys consumed:
        data_mode:           'streaming' | 'memmap'
        data_dir:            path for memmap .bin files (memmap mode)
        hf_dataset_name:     HF dataset name (streaming mode)
        hf_dataset_config:   HF dataset config/subset (streaming mode, can be None)
        hf_train_split:      split name for train (default 'train')
        hf_val_split:        split name for val   (default 'validation')
        text_column:         column with document text (default 'text')
        block_size:          sequence length
        batch_size:          micro-batch size
        device:              torch device string
        device_type:         'cuda' | 'cpu'
        eos_token_id:        EOS token id (streaming mode)
        tokens_consumed:     resume offset (streaming mode, from checkpoint)
        hf_cache_dir:        optional HF cache dir override
    """
    mode = cfg.get("data_mode", "memmap")

    if mode == "memmap":
        return MemmapDataLoader(
            data_dir    = cfg["data_dir"],
            block_size  = cfg["block_size"],
            batch_size  = cfg["batch_size"],
            device      = cfg["device"],
            device_type = cfg["device_type"],
        )

    elif mode == "streaming":
        if tokenizer is None:
            raise ValueError("streaming mode requires a tokenizer object")

        common = dict(
            hf_dataset_name   = cfg["hf_dataset_name"],
            hf_dataset_config = cfg.get("hf_dataset_config", None),
            tokenizer         = tokenizer,
            block_size        = cfg["block_size"],
            eos_token_id      = cfg.get("eos_token_id", tokenizer.token_to_id("<|endoftext|>") or 0),
            text_column       = cfg.get("text_column", "text"),
            cache_dir         = cfg.get("hf_cache_dir", None),
        )

        train_ds = StreamingDataset(
            hf_split        = cfg.get("hf_train_split", "train"),
            tokens_consumed = cfg.get("tokens_consumed", 0),
            **common,
        )
        val_ds = StreamingDataset(
            hf_split        = cfg.get("hf_val_split", "validation"),
            tokens_consumed = 0,  # val always starts from the beginning
            **common,
        )

        return StreamingDataLoader(
            train_dataset = train_ds,
            val_dataset   = val_ds,
            batch_size    = cfg["batch_size"],
            device        = cfg["device"],
            device_type   = cfg["device_type"],
        )

    else:
        raise ValueError(f"Unknown data_mode: {mode!r}. Choose 'streaming' or 'memmap'.")
