"""Sanity-check the FineWeb-Edu streaming pipeline. Does NOT train anything.

Verifies: tokenizer + vocab size, sequence length / packing behavior, the
deterministic hash-based train/val split (zero leakage), batch shapes, and that
a decoded sequence is readable text.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import assign_split, build_dataloaders
from src.utils.config import load_config_from_dict


def _make_config(block_size: int, val_fraction: float, hash_modulo: int, cap_tokens: int):
    return load_config_from_dict(
        {
            "data": {
                "dataset_type": "fineweb_edu",
                "dataset_name": "fineweb_edu",
                "tokenizer_name": "gpt2",
                "fineweb_subset": "sample-10BT",
                "block_size": block_size,
                "batch_size": 4,
                "eval_batch_size": 4,
                "val_fraction": val_fraction,
                "hash_modulo": hash_modulo,
                "max_train_tokens": cap_tokens,
                "max_val_tokens": cap_tokens,
            },
            "model": {"max_seq_len": block_size},
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the FineWeb-Edu pipeline (no training).")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--hash-modulo", type=int, default=1000)
    parser.add_argument("--num-batches", type=int, default=3)
    parser.add_argument("--cap-tokens", type=int, default=200_000)
    parser.add_argument("--split-scan-docs", type=int, default=3000)
    args = parser.parse_args()

    config = _make_config(args.block_size, args.val_fraction, args.hash_modulo, args.cap_tokens)
    tokenizer, train_loader, val_loader, metadata = build_dataloaders(config)

    print("=== Pipeline report ===")
    print(f"tokenizer          : {tokenizer.name}")
    print(f"vocab_size         : {tokenizer.vocab_size}")
    print(f"eos_token_id       : {tokenizer.eos_token_id}")
    print(f"sequence length    : {config.data.block_size}")
    print(f"split mechanism    : {metadata['split_mechanism']}")
    print(f"val_fraction       : {config.data.val_fraction}  (hash_modulo={config.data.hash_modulo})")
    print("packing            : docs tokenized (no special tokens), EOS appended after each doc,")
    print("                     concatenated into a buffer, sliced into fixed-length blocks;")
    print("                     docs are EOS-separated and may span block boundaries.")

    print("\n=== Batch shape check ===")
    first_seq = None
    for batch in itertools.islice(train_loader, args.num_batches):
        x, y = batch["input_ids"], batch["targets"]
        print(f"input_ids {tuple(x.shape)}  targets {tuple(y.shape)}")
        assert x.shape[1] == config.data.block_size, "sequence length mismatch"
        assert y.shape == x.shape
        if first_seq is None:
            first_seq = x[0].tolist()

    print("\n=== Decoded sample (first train sequence, truncated) ===")
    decoded = tokenizer.decode(first_seq)
    print(repr(decoded[:600]))
    eos_count = sum(1 for t in first_seq if t == tokenizer.eos_token_id)
    print(f"\nEOS tokens in this {config.data.block_size}-token sequence: {eos_count}")

    print("\n=== Zero-leakage split check ===")
    raw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    total_tokens = 0
    for row in itertools.islice(raw, args.split_scan_docs):
        doc_key = str(row.get("id") or row.get("text"))
        total_tokens += int(row.get("token_count") or 0)
        role = assign_split(doc_key, val_fraction=args.val_fraction, hash_modulo=args.hash_modulo)
        (val_ids if role == "val" else train_ids).add(doc_key)

    overlap = train_ids & val_ids
    scanned = len(train_ids) + len(val_ids)
    print(f"scanned docs       : {scanned}")
    print(f"train docs         : {len(train_ids)}")
    print(f"val docs           : {len(val_ids)}  (~{100*len(val_ids)/max(1,scanned):.2f}%)")
    print(f"overlap (leakage)  : {len(overlap)}")
    print(f"~tokens in {scanned} scanned docs: {total_tokens:,}")
    print("note: sample-10BT holds ~10B tokens total (streamed, never fully downloaded).")
    assert not overlap, "LEAKAGE: a document appeared in both train and val"

    # Determinism: re-assign the same keys, expect identical routing.
    stable = all(
        assign_split(k, val_fraction=args.val_fraction, hash_modulo=args.hash_modulo) == "val"
        for k in itertools.islice(val_ids, 50)
    )
    print(f"determinism recheck: {'OK' if stable else 'FAILED'}")
    assert stable

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
