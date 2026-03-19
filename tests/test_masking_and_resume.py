from __future__ import annotations

import pytest
import torch

from src.models.baseline import CausalSelfAttention
from src.training.train import validate_resume_checkpoint
from src.utils.config import Config
from src.utils.logging import build_run_identity


def test_causal_mask_prevents_future_leakage() -> None:
    torch.manual_seed(0)
    attention = CausalSelfAttention(
        d_model=16,
        n_heads=4,
        dropout=0.0,
        max_seq_len=16,
    )
    attention.eval()

    x = torch.randn(1, 8, 16)
    y = x.clone()
    y[:, 4:, :] = torch.randn_like(y[:, 4:, :])

    out_x, _ = attention(x, return_attention=True)
    out_y, _ = attention(y, return_attention=True)

    assert torch.allclose(out_x[:, :4, :], out_y[:, :4, :], atol=1e-5)


def test_resume_validation_rejects_config_hash_mismatch() -> None:
    config = Config()
    identity = build_run_identity(config)
    checkpoint = {
        "config_hash": "different_hash",
        "model_type": config.model.architecture,
        "size": config.model.size_name,
        "context": config.data.block_size,
        "tokenizer_name": config.data.tokenizer_name,
    }
    with pytest.raises(ValueError):
        validate_resume_checkpoint(checkpoint, config=config, identity=identity, tokenizer_name=config.data.tokenizer_name)
