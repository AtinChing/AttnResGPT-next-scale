from __future__ import annotations

import argparse
import json
import math
import shutil
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW

from src.data.dataset import build_dataloaders
from src.metrics.norms import StepNormTracker, language_model_loss, perplexity_from_loss
from src.models.attnres import build_model
from src.training.eval import evaluate_model
from src.utils.config import Config, load_config
from src.utils.logging import (
    ExperimentLogger,
    RunIdentity,
    build_run_identity,
    create_run_paths,
    write_run_snapshot,
)
from src.utils.runtime import (
    amp_dtype_from_string,
    count_parameters,
    cycle,
    get_device,
    get_rng_state,
    overall_grad_norm,
    seed_everything,
    set_rng_state,
)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    min_lr_ratio = min_lr / max(base_lr, 1e-12)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _scaler_for(device: torch.device, mixed_precision: bool, amp_dtype: torch.dtype) -> torch.amp.GradScaler:
    use_scaler = device.type == "cuda" and mixed_precision and amp_dtype == torch.float16
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    return torch.amp.GradScaler(scaler_device, enabled=use_scaler)


def _checkpoint_metadata(config: Config, identity: RunIdentity, tokenizer_name: str, step: int) -> dict[str, Any]:
    return {
        "global_step": step,
        "config_hash": identity.config_hash,
        "model_type": config.model.architecture,
        "size": config.model.size_name,
        "context": config.data.block_size,
        "tokenizer_name": tokenizer_name,
        "run_name": identity.run_name,
    }


def build_checkpoint_payload(
    *,
    config: Config,
    identity: RunIdentity,
    step: int,
    tokenizer_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    best_val_loss: Optional[float],
) -> dict[str, Any]:
    payload = _checkpoint_metadata(config, identity, tokenizer_name, step)
    payload.update(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "best_val_loss": best_val_loss,
            "rng_state": get_rng_state(),
        }
    )
    return payload


def _resolve_resume_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    checkpoints = sorted(path.glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {path}")
    return checkpoints[-1]


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    config: Config,
    identity: RunIdentity,
    tokenizer_name: str,
) -> None:
    expected = {
        "config_hash": identity.config_hash,
        "model_type": config.model.architecture,
        "size": config.model.size_name,
        "context": config.data.block_size,
        "tokenizer_name": tokenizer_name,
    }
    for key, expected_value in expected.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            raise ValueError(
                f"Resume checkpoint mismatch for {key}: expected {expected_value}, found {actual_value}"
            )


def _collect_aux_scalars(aux: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in aux.items()
        if isinstance(value, (int, float))
    }


def _probe_payload(
    *,
    step: int,
    train_payload: dict[str, Any],
    aux: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step": step,
        "train_metrics": train_payload,
        "depth_attention_rows": [row.tolist() for row in aux.get("depth_attention_rows", [])],
        "depth_source_indices": aux.get("depth_source_indices", []),
    }


def _write_metadata(paths, metadata: dict[str, Any]) -> None:
    with paths.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _cleanup_partial_outputs(config: Config, identity: RunIdentity) -> None:
    output_root = Path(config.logging.output_root)
    run_dir = output_root / "runs" / identity.run_name
    checkpoint_dir = output_root / "checkpoints" / identity.run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)


def train_from_config(config: Config) -> dict[str, Any]:
    seed_everything(config.experiment.seed, deterministic=config.experiment.deterministic)

    tokenizer, train_loader, val_loader, data_meta = build_dataloaders(config)
    config.model.vocab_size = tokenizer.vocab_size
    identity = build_run_identity(config)
    paths = create_run_paths(config.logging.output_root, identity)

    if config.training.resume_from is None:
        existing_outputs = [
            paths.train_log_path,
            paths.val_log_path,
            paths.summary_json_path,
        ]
        if any(path.exists() for path in existing_outputs):
            raise FileExistsError(
                f"Run directory already contains outputs for {identity.run_name}. "
                "Use training.resume_from to continue or remove the old outputs first."
            )

    tokenizer.save(paths.tokenizer_dir)
    metadata = {
        "status": "running",
        "run_name": identity.run_name,
        "config_hash": identity.config_hash,
        "model_type": config.model.architecture,
        "size": config.model.size_name,
        "context": config.data.block_size,
        "tokenizer_name": tokenizer.name,
        "dataset": config.data.dataset_name,
        "seed": config.experiment.seed,
        "stage": config.experiment.stage,
    }
    logger = ExperimentLogger(paths, config=config, identity=identity)
    metadata.update(logger.wandb_metadata())
    write_run_snapshot(config, identity, paths, metadata)

    device = get_device(config.training.device)
    amp_dtype = amp_dtype_from_string(config.training.amp_dtype)
    model = build_model(config.model).to(device)
    counts = count_parameters(model)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        total_steps=config.training.max_steps,
        base_lr=config.training.learning_rate,
        min_lr=config.training.min_lr,
    )
    scaler = _scaler_for(device, config.training.mixed_precision, amp_dtype)
    norm_tracker = StepNormTracker()
    norm_tracker.register(model)

    start_step = 0
    best_val_loss: float | None = None
    last_checkpoint_path: str | None = None
    last_train_payload: dict[str, Any] = {}
    last_aux: dict[str, Any] = {}

    if config.training.resume_from is not None:
        checkpoint_path = _resolve_resume_path(config.training.resume_from)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not config.training.allow_resume_mismatch:
            validate_resume_checkpoint(checkpoint, config=config, identity=identity, tokenizer_name=tokenizer.name)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler") is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        if "rng_state" in checkpoint:
            set_rng_state(checkpoint["rng_state"])
        start_step = int(checkpoint["global_step"])
        best_val_loss = checkpoint.get("best_val_loss")
        last_checkpoint_path = str(checkpoint_path)

    train_iterator = cycle(train_loader)
    use_autocast = device.type == "cuda" and config.training.mixed_precision

    try:
        try:
            for step in range(start_step + 1, config.training.max_steps + 1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                norm_tracker.reset_step()
                total_loss = 0.0
                probe_step = config.logging.save_probes and (
                    step == 1 or step == config.training.max_steps or step % config.training.probe_interval == 0
                )

                for _ in range(config.training.grad_accum_steps):
                    batch = next(train_iterator)
                    input_ids = batch["input_ids"].to(device)
                    targets = batch["targets"].to(device)
                    autocast_context = torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=use_autocast,
                    )
                    with autocast_context if use_autocast else nullcontext():
                        logits, aux = model(input_ids, return_aux=probe_step)
                        loss = language_model_loss(logits, targets)
                        scaled_loss = loss / config.training.grad_accum_steps
                    total_loss += float(loss.item())
                    last_aux = aux
                    if scaler.is_enabled():
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
                grad_norm = overall_grad_norm(model)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()

                step_norms = norm_tracker.snapshot()
                train_payload = {
                    "step": step,
                    "train_loss": total_loss / config.training.grad_accum_steps,
                    "train_perplexity": perplexity_from_loss(total_loss / config.training.grad_accum_steps),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "global_grad_norm": grad_norm,
                    **step_norms,
                    **_collect_aux_scalars(last_aux),
                }
                last_train_payload = deepcopy(train_payload)

                if step % config.training.log_interval == 0 or step == 1:
                    logger.log_train(train_payload)

                if probe_step:
                    logger.save_probe(step, _probe_payload(step=step, train_payload=train_payload, aux=last_aux))

                if step % config.training.eval_interval == 0 or step == config.training.max_steps:
                    val_metrics = evaluate_model(
                        model,
                        val_loader,
                        device=device,
                        amp_dtype=amp_dtype,
                        max_batches=config.training.eval_max_batches,
                        collect_artifacts=True,
                    )
                    val_payload = {"step": step, **val_metrics}
                    logger.log_val(val_payload)
                    if best_val_loss is None or val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = float(val_metrics["val_loss"])

                if config.logging.save_checkpoints and (
                    step % config.training.checkpoint_interval == 0 or step == config.training.max_steps
                ):
                    checkpoint_path = paths.checkpoint_dir / f"step_{step:07d}.pt"
                    torch.save(
                        build_checkpoint_payload(
                            config=config,
                            identity=identity,
                            step=step,
                            tokenizer_name=tokenizer.name,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            best_val_loss=best_val_loss,
                        ),
                        checkpoint_path,
                    )
                    last_checkpoint_path = str(checkpoint_path)
                    logger.prune_old_checkpoints(config.logging.keep_last_k_checkpoints)

            final_eval = evaluate_model(
                model,
                val_loader,
                device=device,
                amp_dtype=amp_dtype,
                max_batches=config.evaluation.max_batches or config.training.eval_max_batches,
                collect_artifacts=True,
            )
            wandb_metadata = logger.wandb_metadata()
            summary = {
                "run_name": identity.run_name,
                "config_hash": identity.config_hash,
                "model": config.model.architecture,
                "size": config.model.size_name,
                "context": config.data.block_size,
                "dataset": config.data.dataset_name,
                "tokenizer_name": tokenizer.name,
                "seed": config.experiment.seed,
                "parameter_count_total": counts["total"],
                "parameter_count_trainable": counts["trainable"],
                "best_val_loss": best_val_loss,
                "checkpoint_path": last_checkpoint_path,
                "wandb_enabled": wandb_metadata.get("wandb_enabled"),
                "wandb_mode": wandb_metadata.get("wandb_mode"),
                "wandb_url": wandb_metadata.get("wandb_url"),
                "last_gradient_norms": last_train_payload.get("gradient_norms", {}),
                "last_activation_norms_train": last_train_payload.get("activation_norms", {}),
                **data_meta,
                **final_eval,
            }
            logger.save_summary(summary)

            metadata.update(
                {
                    "status": "completed",
                    "checkpoint_path": last_checkpoint_path,
                    "global_step": config.training.max_steps,
                }
            )
            _write_metadata(paths, metadata)
            logger.close(status='completed')
            return summary
        except Exception as error:
            metadata.update(
                {
                    "status": "failed",
                    "error": str(error),
                    "checkpoint_path": last_checkpoint_path,
                }
            )
            _write_metadata(paths, metadata)
            logger.close(status='failed')
            raise
    finally:
        norm_tracker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline or AttnRes GPT run.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--model", choices=["baseline", "attnres"], default=None)
    parser.add_argument("--overrides", nargs="*", default=[], help="Optional key=value config overrides.")
    args = parser.parse_args()

    overrides = list(args.overrides)
    if args.model is not None:
        overrides.append(f"model.architecture={args.model}")

    config = load_config(args.config, overrides=overrides)
    summary = train_from_config(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
