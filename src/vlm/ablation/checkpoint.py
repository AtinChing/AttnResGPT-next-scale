from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import torch

from src.utils.runtime import get_rng_state, set_rng_state
from src.vlm.ablation.io_utils import atomic_torch_save, atomic_write_json, ensure_dir


def archive_run_dir(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = run_dir.parent / f"{run_dir.name}_archived_{stamp}"
    if run_dir.exists():
        shutil.move(str(run_dir), str(archive))
    return archive


def build_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    batch_index: int,
    global_step: int,
    best_val_accuracy: float,
    early_stopping_bad_epochs: int,
    examples_processed: int,
    elapsed_training_time: float,
    model_config: dict[str, Any],
    dataset_config: dict[str, Any],
    tokenizer_vocab: dict[str, int],
    variant: str,
    seed: int,
    config_hash: str,
    source_code_hash: str,
    status: str = "running",
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler.is_enabled() else None,
        "epoch": epoch,
        "batch_index": batch_index,
        "global_step": global_step,
        "best_val_accuracy": best_val_accuracy,
        "early_stopping_bad_epochs": early_stopping_bad_epochs,
        "examples_processed": examples_processed,
        "elapsed_training_time": elapsed_training_time,
        "rng_state": get_rng_state(),
        "model_config": model_config,
        "dataset_config": dataset_config,
        "tokenizer_vocab": tokenizer_vocab,
        "variant": variant,
        "seed": seed,
        "config_hash": config_hash,
        "source_code_hash": source_code_hash,
        "status": status,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    atomic_torch_save(path, payload)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def restore_training_state(
    checkpoint: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> None:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler") is not None and scaler.is_enabled():
        scaler.load_state_dict(checkpoint["scaler"])
    if "rng_state" in checkpoint:
        set_rng_state(checkpoint["rng_state"])


def validate_checkpoint_compatibility(
    checkpoint: dict[str, Any],
    *,
    variant: str,
    seed: int,
    config_hash: str,
    source_code_hash: str | None = None,
    force_restart: bool = False,
) -> None:
    if force_restart:
        return
    expected = {
        "variant": variant,
        "seed": seed,
        "config_hash": config_hash,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Incompatible checkpoint for {key}: expected {value}, found {checkpoint.get(key)}"
            )
    if source_code_hash is not None and checkpoint.get("source_code_hash") not in {None, source_code_hash}:
        raise ValueError(
            "Checkpoint source_code_hash mismatch. Set FORCE_RESTART=True to archive and restart."
        )


def write_status(run_dir: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(ensure_dir(run_dir) / "status.json", payload)


def mark_completed(run_dir: Path) -> None:
    (ensure_dir(run_dir) / "completed.marker").write_text("ok\n", encoding="utf-8")
