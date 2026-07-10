"""
Minion AI — tokenizer_train.py
Train a custom BPE or Unigram tokenizer from your own corpus,
or prepare pre-tokenized .bin files for the memmap data loader.

Usage:
    # Train a new 32k BPE tokenizer from a text file or HF dataset:
    python tokenizer_train.py --mode train --corpus_path data/mytext.txt --vocab_size 32768

    # Train from a HuggingFace dataset (streams, no full download):
    python tokenizer_train.py --mode train --hf_dataset fineweb-edu --vocab_size 32768

    # Prepare memmap .bin files from a tokenized corpus:
    python tokenizer_train.py --mode prepare --data_dir data/mydata --tokenizer_path tokenizer/

    # Load a pre-trained HF tokenizer (for Mode B / qlora_finetune):
    python tokenizer_train.py --mode load_hf --hf_tokenizer_name gpt2

The output tokenizer is saved as tokenizer/tokenizer.json (HF-compatible format).
For Mode B, the base model's tokenizer is loaded directly by train.py — you do
not need to run this script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Text iterator utilities
# ---------------------------------------------------------------------------

def iter_text_file(path: str, batch_size: int = 1000) -> Iterator[list[str]]:
    """Yield batches of lines from a plain-text file."""
    batch: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                batch.append(line)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def iter_hf_dataset(
    dataset_name: str,
    config_name: Optional[str],
    split: str,
    text_column: str,
    batch_size: int = 1000,
    max_docs: Optional[int] = None,
) -> Iterator[list[str]]:
    """Stream text from a HuggingFace dataset in batches."""
    from datasets import load_dataset

    print(f"Streaming '{dataset_name}' (config={config_name}, split={split})")
    ds = load_dataset(
        dataset_name,
        config_name,
        split=split,
        streaming=True,
    )
    batch: list[str] = []
    n = 0
    for doc in ds:
        text = doc.get(text_column, "")
        if text:
            batch.append(text)
            n += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
        if max_docs is not None and n >= max_docs:
            break
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def train_tokenizer(
    text_iterator: Iterator[list[str]],
    vocab_size: int,
    algorithm: str,
    output_dir: str,
    special_tokens: list[str],
) -> None:
    """Train a BPE or Unigram tokenizer using HuggingFace tokenizers.

    Args:
        text_iterator: Yields batches of text strings.
        vocab_size:    Target vocabulary size.
                       Tradeoff: smaller vocab → smaller embedding matrix (less VRAM),
                       larger vocab → fewer tokens per document (longer effective context).
                       32k is a good balance for a 4 GB inference GPU.
        algorithm:     'bpe' or 'unigram'.
                       BPE: greedy merge-based, widely used (GPT-2, LLaMA use variants).
                       Unigram: probabilistic, slightly better fertility on non-English.
        output_dir:    Directory to save tokenizer files.
        special_tokens: List of special token strings to add (e.g. '<|endoftext|>').
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    os.makedirs(output_dir, exist_ok=True)

    if algorithm == "bpe":
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
            min_frequency=2,
        )
    elif algorithm == "unigram":
        tokenizer = Tokenizer(models.Unigram())
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
        trainer = trainers.UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose 'bpe' or 'unigram'.")

    print(f"Training {algorithm.upper()} tokenizer | vocab_size={vocab_size}")

    # HF tokenizers train_from_iterator expects an iterator of strings or lists
    def flat_iter() -> Iterator[str]:
        for batch in text_iterator:
            yield from batch

    tokenizer.train_from_iterator(flat_iter(), trainer=trainer)

    out_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(out_path)
    print(f"Tokenizer saved to {out_path}")

    # Also save a vocab.txt for inspection
    vocab = tokenizer.get_vocab()
    vocab_path = os.path.join(output_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{repr(token)}\n")
    print(f"Vocabulary ({len(vocab)} tokens) saved to {vocab_path}")


# ---------------------------------------------------------------------------
# Prepare .bin memmap files
# ---------------------------------------------------------------------------

def prepare_bin_files(
    data_dir: str,
    tokenizer_path: str,
    corpus_path: Optional[str] = None,
    hf_dataset_name: Optional[str] = None,
    hf_dataset_config: Optional[str] = None,
    hf_train_split: str = "train",
    hf_val_split: str = "validation",
    text_column: str = "text",
    val_fraction: float = 0.0005,
) -> None:
    """Tokenize a corpus and write train.bin / val.bin for memmap mode.

    The .bin files store token ids as uint16 (supports vocab_size up to 65535;
    if your vocab is larger, change dtype to uint32 and update MemmapDataLoader).

    Args:
        data_dir:       Output directory for .bin files.
        tokenizer_path: Path to tokenizer.json or HF tokenizer directory.
        corpus_path:    Plain-text file with one document per line.
        hf_dataset_name: HF dataset name (alternative to corpus_path).
        val_fraction:   Fraction of tokens held out for validation.
    """
    from tokenizers import Tokenizer

    os.makedirs(data_dir, exist_ok=True)

    # Load tokenizer
    tok_json = os.path.join(tokenizer_path, "tokenizer.json")
    if os.path.exists(tok_json):
        tokenizer = Tokenizer.from_file(tok_json)
    else:
        # Try loading as an HF tokenizer directory
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def encode(text: str) -> list[int]:
        enc = tokenizer.encode(text)
        if hasattr(enc, "ids"):
            return enc.ids
        return list(enc)

    print("Tokenizing corpus...")
    all_ids: list[int] = []

    if corpus_path is not None:
        for batch in iter_text_file(corpus_path):
            for text in batch:
                all_ids.extend(encode(text))
    elif hf_dataset_name is not None:
        for split in (hf_train_split, hf_val_split):
            for batch in iter_hf_dataset(hf_dataset_name, hf_dataset_config, split, text_column):
                for text in batch:
                    all_ids.extend(encode(text))
    else:
        raise ValueError("Provide either corpus_path or hf_dataset_name")

    print(f"Total tokens: {len(all_ids):,}")

    # Split train / val
    val_n   = max(1, int(len(all_ids) * val_fraction))
    train_n = len(all_ids) - val_n

    for name, ids in [("train", all_ids[:train_n]), ("val", all_ids[train_n:])]:
        arr = np.array(ids, dtype=np.uint16)
        path = os.path.join(data_dir, f"{name}.bin")
        arr.tofile(path)
        print(f"Wrote {path}  ({len(arr):,} tokens)")

    # Save meta.pkl for compatibility with original nanoGPT
    import pickle
    meta = {
        "vocab_size": tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size")
                      else len(tokenizer),
        "tokenizer": tokenizer_path,
    }
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("Saved meta.pkl")


# ---------------------------------------------------------------------------
# Load pretrained HF tokenizer (Mode B helper)
# ---------------------------------------------------------------------------

def load_hf_tokenizer(model_name: str, output_dir: str) -> None:
    """Download and save a HuggingFace tokenizer for use in Mode B (qlora_finetune).

    For Mode B, the tokenizer MUST match the base model's tokenizer.
    This script downloads it and saves a local copy in output_dir so
    train.py can load it without internet access during training.
    """
    from transformers import AutoTokenizer

    print(f"Downloading tokenizer for '{model_name}'...")
    tok = AutoTokenizer.from_pretrained(model_name)
    os.makedirs(output_dir, exist_ok=True)
    tok.save_pretrained(output_dir)
    print(f"Saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Minion AI tokenizer tool")
    parser.add_argument("--mode", required=True,
                        choices=["train", "prepare", "load_hf"],
                        help="train: fit tokenizer | prepare: write .bin files | load_hf: download HF tokenizer")

    # --- train mode ---
    parser.add_argument("--corpus_path",      default=None, help="Path to plain-text corpus file")
    parser.add_argument("--hf_dataset",       default=None, help="HF dataset name for streaming training")
    parser.add_argument("--hf_dataset_config",default=None, help="HF dataset config/subset")
    parser.add_argument("--hf_split",         default="train")
    parser.add_argument("--text_column",      default="text")
    parser.add_argument("--max_docs",         type=int, default=None, help="Cap documents (for fast test)")
    parser.add_argument("--vocab_size",       type=int, default=32768)
    parser.add_argument("--algorithm",        default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--tokenizer_dir",    default="tokenizer",
                        help="Output dir for tokenizer.json")
    parser.add_argument("--special_tokens",   nargs="+",
                        default=["<unk>", "<|endoftext|>", "<|pad|>"])

    # --- prepare mode ---
    parser.add_argument("--data_dir",         default="data/prepared",
                        help="Output dir for .bin files")
    parser.add_argument("--tokenizer_path",   default="tokenizer",
                        help="Dir containing tokenizer.json")
    parser.add_argument("--val_fraction",     type=float, default=0.0005)

    # --- load_hf mode ---
    parser.add_argument("--hf_tokenizer_name", default="gpt2",
                        help="HF model name whose tokenizer to download")

    args = parser.parse_args()

    if args.mode == "train":
        if args.corpus_path:
            text_iter = iter_text_file(args.corpus_path)
        elif args.hf_dataset:
            text_iter = iter_hf_dataset(
                args.hf_dataset, args.hf_dataset_config,
                args.hf_split, args.text_column, max_docs=args.max_docs,
            )
        else:
            parser.error("--mode train requires --corpus_path or --hf_dataset")

        train_tokenizer(
            text_iterator = text_iter,
            vocab_size    = args.vocab_size,
            algorithm     = args.algorithm,
            output_dir    = args.tokenizer_dir,
            special_tokens= args.special_tokens,
        )

    elif args.mode == "prepare":
        prepare_bin_files(
            data_dir          = args.data_dir,
            tokenizer_path    = args.tokenizer_path,
            corpus_path       = args.corpus_path,
            hf_dataset_name   = args.hf_dataset,
            hf_dataset_config = args.hf_dataset_config,
            text_column       = args.text_column,
            val_fraction      = args.val_fraction,
        )

    elif args.mode == "load_hf":
        load_hf_tokenizer(args.hf_tokenizer_name, args.tokenizer_dir)


if __name__ == "__main__":
    main()
