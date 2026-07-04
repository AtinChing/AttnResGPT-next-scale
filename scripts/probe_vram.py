"""Probe the largest micro-batch that fits in VRAM, then hold the effective
token batch constant via gradient accumulation.

CUDA-intended: run this on the training GPU (e.g. the Razer RTX box). It builds
the real model from a config and runs a full forward + backward + optimizer step
with synthetic data (no dataset/streaming needed), measuring peak memory for each
candidate micro-batch. It does NOT train.

Example:
    python -m scripts.probe_vram --config configs/fineweb_30m_diag.yaml
"""
from __future__ import annotations

import argparse

import torch
from torch.optim import AdamW

from src.metrics.norms import language_model_loss
from src.models.attnres import build_model
from src.utils.config import load_config
from src.utils.runtime import amp_dtype_from_string


def _try_micro_batch(config, device: torch.device, amp_dtype: torch.dtype, micro_batch: int, steps: int) -> float:
    """Return peak MiB for a micro-batch, or raise on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = build_model(config.model).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.adam_eps,
        weight_decay=config.training.weight_decay,
    )
    block_size = config.data.block_size
    vocab = config.model.vocab_size
    use_amp = config.training.mixed_precision and device.type == "cuda"

    model.train()
    for _ in range(steps):
        input_ids = torch.randint(0, vocab, (micro_batch, block_size), device=device)
        targets = torch.randint(0, vocab, (micro_batch, block_size), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits, _ = model(input_ids, return_aux=False)
            loss = language_model_loss(logits, targets)
        loss.backward()
        optimizer.step()

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    del model, optimizer
    torch.cuda.empty_cache()
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe max micro-batch and derive grad_accum (no training).")
    parser.add_argument("--config", default="configs/fineweb_30m_diag.yaml")
    parser.add_argument("--candidates", default="1,2,4,8,16,32", help="Comma-separated micro-batch sizes to try.")
    parser.add_argument("--target-tokens", type=int, default=None,
                        help="Effective batch in tokens to hold constant (default: micro*accum*block from config).")
    parser.add_argument("--steps", type=int, default=3, help="Iterations per candidate (to include optimizer state).")
    parser.add_argument("--headroom", type=float, default=0.90, help="Fraction of total VRAM to stay under.")
    args = parser.parse_args()

    config = load_config(args.config)

    if not torch.cuda.is_available():
        print("CUDA not available. This probe is meant to run on the training GPU; exiting without measuring.")
        return

    device = torch.device("cuda")
    amp_dtype = amp_dtype_from_string(config.training.amp_dtype)
    total_mib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    block_size = config.data.block_size
    target_tokens = args.target_tokens or (
        config.data.batch_size * config.training.grad_accum_steps * block_size
    )
    budget = args.headroom * total_mib

    print(f"device        : {torch.cuda.get_device_name(device)}")
    print(f"total VRAM     : {total_mib:,.0f} MiB  (usable budget @ {args.headroom:.0%}: {budget:,.0f} MiB)")
    print(f"model          : {config.model.architecture}  block_size={block_size}  precision={config.training.amp_dtype}")
    print(f"target eff batch: {target_tokens:,} tokens\n")

    candidates = [int(c) for c in args.candidates.split(",") if c.strip()]
    largest_fit = None
    for micro_batch in candidates:
        try:
            peak = _try_micro_batch(config, device, amp_dtype, micro_batch, args.steps)
        except torch.cuda.OutOfMemoryError:
            print(f"micro_batch={micro_batch:>3}: OOM")
            torch.cuda.empty_cache()
            break
        fits = peak <= budget
        flag = "OK" if fits else "over budget"
        print(f"micro_batch={micro_batch:>3}: peak {peak:8,.0f} MiB  [{flag}]")
        if fits:
            largest_fit = micro_batch

    print()
    if largest_fit is None:
        print("No candidate fit the budget. Try smaller micro-batches or a smaller block_size.")
        return

    accum = max(1, round(target_tokens / (largest_fit * block_size)))
    eff_seq = largest_fit * accum
    eff_tok = eff_seq * block_size
    print(f"recommended micro_batch : {largest_fit}")
    print(f"recommended grad_accum  : {accum}")
    print(f"=> effective batch      : {eff_seq} seqs = {eff_tok:,} tokens")
    if eff_tok != target_tokens:
        print(f"note: closest achievable to target {target_tokens:,} (accum is integer).")
    print(f"\nApply with:\n  --overrides data.batch_size={largest_fit} training.grad_accum_steps={accum}")


if __name__ == "__main__":
    main()
