from __future__ import annotations

import pytest
import torch

from src.metrics.norms import language_model_loss
from src.models.attnres import build_model
from src.utils.config import AttnResConfig, ModelConfig


def _make_model_config(architecture: str, vocab_size: int) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        size_name="small",
        vocab_size=vocab_size,
        max_seq_len=24,
        d_model=48,
        n_layers=2,
        n_heads=4,
        d_ff=96,
        dropout=0.0,
        attnres=AttnResConfig(enabled=architecture == "attnres", final_readout=True),
    )


def _run_overfit(model: torch.nn.Module, input_ids: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    initial_loss = None
    final_loss = None
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(input_ids, return_aux=False)
        loss = language_model_loss(logits, targets)
        if initial_loss is None:
            initial_loss = float(loss.item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    return float(initial_loss), float(final_loss)


@pytest.mark.slow
def test_tiny_overfit_baseline_and_attnres() -> None:
    token_ids = ([3, 7, 11, 5, 9, 2, 13, 1] * 8)[:40]
    input_ids = torch.tensor(token_ids[:24], dtype=torch.long).unsqueeze(0)
    targets = torch.tensor(token_ids[1:25], dtype=torch.long).unsqueeze(0)

    baseline = build_model(_make_model_config("baseline", 32))
    attnres = build_model(_make_model_config("attnres", 32))

    baseline_initial, baseline_final = _run_overfit(baseline, input_ids, targets)
    attn_initial, attn_final = _run_overfit(attnres, input_ids, targets)

    assert baseline_final < baseline_initial
    assert attn_final < attn_initial
