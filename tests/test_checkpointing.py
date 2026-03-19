from __future__ import annotations

from pathlib import Path

import torch

from src.models.attnres import build_model
from src.training.train import build_checkpoint_payload
from src.utils.config import Config
from src.utils.logging import build_run_identity


def test_checkpoint_metadata_roundtrip(tmp_path: Path) -> None:
    config = Config()
    config.model.vocab_size = 64
    config.model.max_seq_len = 16
    config.data.block_size = 16
    identity = build_run_identity(config)
    model = build_model(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    payload = build_checkpoint_payload(
        config=config,
        identity=identity,
        step=7,
        tokenizer_name="gpt2",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        best_val_loss=1.23,
    )
    checkpoint_path = tmp_path / "step_0000007.pt"
    torch.save(payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu")

    assert loaded["global_step"] == 7
    assert loaded["config_hash"] == identity.config_hash
    assert loaded["model_type"] == config.model.architecture
    assert loaded["size"] == config.model.size_name
    assert loaded["context"] == config.data.block_size
    assert loaded["tokenizer_name"] == "gpt2"
