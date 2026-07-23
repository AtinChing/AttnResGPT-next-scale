from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.vlm_attnres import TinyAttnResVLM
from src.utils.runtime import amp_dtype_from_string, count_parameters, seed_everything
from src.vlm.ablation.checkpoint import (
    archive_run_dir,
    build_checkpoint_payload,
    load_checkpoint,
    mark_completed,
    restore_training_state,
    save_checkpoint,
    validate_checkpoint_compatibility,
    write_status,
)
from src.vlm.ablation.config import (
    AblationExperimentConfig,
    VARIANTS,
    build_decoder_config,
    build_vision_config,
    config_hash,
    run_dir_for,
)
from src.vlm.ablation.eval import evaluate_model
from src.vlm.ablation.init_sync import copy_shared_weights, validate_shared_initialization
from src.vlm.ablation.io_utils import append_jsonl, atomic_write_json, ensure_dir
from src.vlm.ablation.manifest import ExperimentManifest
from src.vlm.ablation.routing import aggregate_routing_rows
from src.vlm.ablation.wandb_logger import AblationWandbLogger
from src.vlm.synthetic_vqa import SyntheticVQADataset, VQATokenizer, collate_vqa_batch


def _select_amp_dtype(requested: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if requested == "auto":
        major, _ = torch.cuda.get_device_capability(device)
        # bf16 is reliable on Ampere+ (sm80+); T4 is sm75 -> float16.
        return torch.bfloat16 if major >= 8 else torch.float16
    return amp_dtype_from_string(requested)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def build_dataloaders(
    config: AblationExperimentConfig,
    tokenizer: VQATokenizer,
) -> dict[str, DataLoader]:
    collate = partial(collate_vqa_batch, pad_token_id=tokenizer.pad_token_id)
    loaders: dict[str, DataLoader] = {}
    split_seeds = {
        "train": config.dataset_seed_offset,
        "validation": config.dataset_seed_offset + 1,
        "test": config.dataset_seed_offset + 2,
    }
    sizes = {
        "train": config.train_size,
        "validation": config.validation_size,
        "test": config.test_size,
    }
    for split, size in sizes.items():
        dataset = SyntheticVQADataset(
            split=split,  # type: ignore[arg-type]
            size=size,
            split_seed=split_seeds[split],
            tokenizer=tokenizer,
            image_size=config.image_size,
            supervise_eos=config.supervise_eos,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(split == "train"),
            num_workers=config.num_workers,
            collate_fn=collate,
            pin_memory=True,
        )
    return loaders


def build_model_for_variant(
    config: AblationExperimentConfig,
    *,
    variant: str,
    vocab_size: int,
    seed: int,
    reference: TinyAttnResVLM | None = None,
) -> tuple[TinyAttnResVLM, dict[str, Any]]:
    residual = VARIANTS[variant]
    seed_everything(seed, deterministic=True)
    vision_config = build_vision_config(config, residual["encoder"])
    decoder_config = build_decoder_config(config, residual["decoder"], vocab_size=vocab_size)
    model = TinyAttnResVLM(
        vision_config=vision_config,
        decoder_config=decoder_config,
        encoder_residual=residual["encoder"],
        decoder_residual=residual["decoder"],
    )
    init_meta: dict[str, Any] = {"copied_shared_tensors": 0, "shared_init": None}
    if reference is not None:
        init_meta["copied_shared_tensors"] = copy_shared_weights(reference, model)
        init_meta["shared_init"] = validate_shared_initialization(reference, model)
    return model, init_meta


def train_variant_seed(
    config: AblationExperimentConfig,
    *,
    variant: str,
    seed: int,
    project_root: Path,
    manifest: ExperimentManifest,
    source_code_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    cfg_hash = config_hash(config)
    run_dir = run_dir_for(project_root, variant, seed, cfg_hash)
    completed_marker = run_dir / "completed.marker"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    if completed_marker.exists() and config.resume and not config.force_restart:
        metrics_path = run_dir / "final_test_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        manifest.upsert(
            variant,
            seed,
            cfg_hash,
            status="completed",
            run_directory=str(run_dir),
            latest_checkpoint=str(last_path),
            best_checkpoint=str(best_path),
            best_validation_accuracy=metrics.get("validation_accuracy"),
        )
        return {"status": "skipped_completed", "run_dir": str(run_dir), "metrics": metrics}

    if config.force_restart and run_dir.exists():
        archive_run_dir(run_dir)

    ensure_dir(run_dir)
    tokenizer = VQATokenizer()
    loaders = build_dataloaders(config, tokenizer)

    seed_everything(seed, deterministic=True)
    reference, _ = build_model_for_variant(
        config,
        variant="baseline",
        vocab_size=tokenizer.vocab_size,
        seed=seed,
    )
    model, init_meta = build_model_for_variant(
        config,
        variant=variant,
        vocab_size=tokenizer.vocab_size,
        seed=seed,
        reference=reference if variant != "baseline" else None,
    )
    if variant == "baseline":
        init_meta["shared_init"] = validate_shared_initialization(reference, model)

    model = model.to(device)
    amp_dtype = _select_amp_dtype(config.amp_dtype, device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )
    steps_per_epoch = max(1, len(loaders["train"]))
    total_steps = steps_per_epoch * config.max_epochs
    warmup_steps = max(1, int(total_steps * config.warmup_fraction))
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=config.min_lr_ratio,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and config.mixed_precision and amp_dtype == torch.float16,
    )

    start_epoch = 0
    global_step = 0
    best_val_accuracy = -1.0
    bad_epochs = 0
    examples_processed = 0
    elapsed_training_time = 0.0
    resumed = False

    if config.resume and last_path.exists() and not config.force_restart:
        checkpoint = load_checkpoint(last_path, map_location=device)
        validate_checkpoint_compatibility(
            checkpoint,
            variant=variant,
            seed=seed,
            config_hash=cfg_hash,
            source_code_hash=source_code_hash,
            force_restart=False,
        )
        restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_val_accuracy = float(checkpoint["best_val_accuracy"])
        bad_epochs = int(checkpoint["early_stopping_bad_epochs"])
        examples_processed = int(checkpoint["examples_processed"])
        elapsed_training_time = float(checkpoint["elapsed_training_time"])
        resumed = True

    param_counts = count_parameters(model)
    baseline_params = count_parameters(reference)
    param_increase_pct = 100.0 * (param_counts["total"] - baseline_params["total"]) / max(1, baseline_params["total"])

    wandb_logger = AblationWandbLogger(
        config=config,
        variant=variant,
        seed=seed,
        config_hash=cfg_hash,
        run_dir=run_dir,
        extra_config={
            "parameter_count": param_counts,
            "parameter_increase_pct": param_increase_pct,
            "amp_dtype": str(amp_dtype),
            "source_code_hash": source_code_hash,
            "resumed": resumed,
        },
    )
    wandb_logger.update_summary(
        {
            "variant": variant,
            "seed": seed,
            "encoder_residual": VARIANTS[variant]["encoder"],
            "decoder_residual": VARIANTS[variant]["decoder"],
            "parameter_count": param_counts["total"],
            "parameter_increase_pct": param_increase_pct,
            "device": str(device),
            "amp_dtype": str(amp_dtype),
            "resumed": resumed,
            **wandb_logger.metadata(),
        }
    )

    atomic_write_json(
        run_dir / "config.json",
        {
            "experiment": config.to_dict(),
            "variant": variant,
            "seed": seed,
            "config_hash": cfg_hash,
            "source_code_hash": source_code_hash,
            "init_validation": init_meta,
            "parameter_count": param_counts,
            "parameter_increase_pct": param_increase_pct,
            "amp_dtype": str(amp_dtype),
            "wandb": wandb_logger.metadata(),
        },
    )

    manifest.upsert(
        variant,
        seed,
        cfg_hash,
        status="running",
        run_directory=str(run_dir),
        latest_checkpoint=str(last_path) if last_path.exists() else None,
        best_checkpoint=str(best_path) if best_path.exists() else None,
        current_epoch=start_epoch,
        global_step=global_step,
        best_validation_accuracy=best_val_accuracy if best_val_accuracy >= 0 else None,
    )

    def _save(path: Path, *, epoch: int, batch_index: int, status: str = "running") -> None:
        payload = build_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            batch_index=batch_index,
            global_step=global_step,
            best_val_accuracy=best_val_accuracy,
            early_stopping_bad_epochs=bad_epochs,
            examples_processed=examples_processed,
            elapsed_training_time=elapsed_training_time,
            model_config={
                "encoder_residual": model.encoder_residual,
                "decoder_residual": model.decoder_residual,
                "vision": model.vision_config.__dict__,
                "decoder": model.decoder_config.__dict__,
            },
            dataset_config={
                "train_size": config.train_size,
                "validation_size": config.validation_size,
                "test_size": config.test_size,
                "image_size": config.image_size,
            },
            tokenizer_vocab=tokenizer.vocab,
            variant=variant,
            seed=seed,
            config_hash=cfg_hash,
            source_code_hash=source_code_hash,
            status=status,
        )
        # Dataclass nested objects are not JSON-safe in torch save metadata; convert lightly.
        payload["model_config"]["vision"] = {
            key: (value if not hasattr(value, "__dict__") else value.__dict__)
            for key, value in model.vision_config.__dict__.items()
        }
        payload["model_config"]["decoder"] = {
            key: (value if not hasattr(value, "__dict__") else value.__dict__)
            for key, value in model.decoder_config.__dict__.items()
        }
        save_checkpoint(path, payload)
        write_status(
            run_dir,
            {
                "status": status,
                "epoch": epoch,
                "global_step": global_step,
                "best_val_accuracy": best_val_accuracy,
                "examples_processed": examples_processed,
            },
        )
        manifest.upsert(
            variant,
            seed,
            cfg_hash,
            status=status if status != "running" else "running",
            run_directory=str(run_dir),
            latest_checkpoint=str(last_path),
            best_checkpoint=str(best_path) if best_path.exists() else None,
            current_epoch=epoch,
            global_step=global_step,
            best_validation_accuracy=best_val_accuracy if best_val_accuracy >= 0 else None,
        )

    peak_alloc = 0
    peak_reserved = 0
    train_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(start_epoch, config.max_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_examples = 0
            epoch_t0 = time.perf_counter()
            for batch_index, batch in enumerate(loaders["train"]):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None)
                    if device.type == "cuda" and config.mixed_precision
                    else nullcontext()
                )
                with autocast:
                    output = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        targets=targets,
                        return_aux=False,
                    )
                    loss = output["loss"]
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                scheduler.step()

                batch_size = input_ids.size(0)
                epoch_loss += float(loss.item()) * batch_size
                epoch_examples += batch_size
                examples_processed += batch_size
                global_step += 1

                if config.wandb_log_interval > 0 and global_step % config.wandb_log_interval == 0:
                    wandb_logger.log(
                        {
                            "train/loss": float(loss.item()),
                            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

                if config.checkpoint_interval > 0 and global_step % config.checkpoint_interval == 0:
                    elapsed_training_time += time.perf_counter() - epoch_t0
                    _save(last_path, epoch=epoch, batch_index=batch_index)
                    epoch_t0 = time.perf_counter()

            elapsed_training_time += time.perf_counter() - epoch_t0
            train_loss = epoch_loss / max(1, epoch_examples)
            throughput = epoch_examples / max(1e-6, time.perf_counter() - (train_start if epoch == start_epoch else epoch_t0))
            append_jsonl(
                run_dir / "train_metrics.jsonl",
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": train_loss,
                    "examples": epoch_examples,
                    "examples_per_sec": throughput,
                },
            )
            wandb_logger.log(
                {
                    "train/epoch_loss": train_loss,
                    "train/examples_per_sec": throughput,
                    "train/epoch": epoch,
                },
                step=global_step,
            )

            _save(last_path, epoch=epoch + 1, batch_index=0)
            val_metrics = evaluate_model(
                model,
                loaders["validation"],
                device=device,
                amp_dtype=amp_dtype if config.mixed_precision else None,
                capture_routing=True,
            )
            append_jsonl(
                run_dir / "validation_metrics.jsonl",
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": val_metrics["loss"],
                    "accuracy": val_metrics["accuracy"],
                    "family_accuracy": val_metrics["family_accuracy"],
                },
            )
            routing_summary = aggregate_routing_rows(val_metrics["routing"])
            append_jsonl(
                run_dir / "routing_metrics.jsonl",
                {"epoch": epoch, "global_step": global_step, **routing_summary},
            )
            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = float(val_metrics["accuracy"])
                bad_epochs = 0
                _save(best_path, epoch=epoch + 1, batch_index=0)
            else:
                bad_epochs += 1

            wandb_logger.log(
                {
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/family_accuracy": val_metrics["family_accuracy"],
                    "val/best_accuracy": best_val_accuracy,
                    "encoder_routing/n_sites": len(routing_summary.get("encoder_routing", [])),
                    "decoder_routing/n_sites": len(routing_summary.get("decoder_routing", [])),
                },
                step=global_step,
            )
            _save(last_path, epoch=epoch + 1, batch_index=0)

            if device.type == "cuda":
                peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated(device)))
                peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))

            if bad_epochs >= config.early_stopping_patience:
                break

        # Final evaluation on best checkpoint when available.
        if best_path.exists():
            best_ckpt = load_checkpoint(best_path, map_location=device)
            model.load_state_dict(best_ckpt["model"])

        val_final = evaluate_model(
            model,
            loaders["validation"],
            device=device,
            amp_dtype=amp_dtype if config.mixed_precision else None,
            capture_routing=True,
        )
        test_final = evaluate_model(
            model,
            loaders["test"],
            device=device,
            amp_dtype=amp_dtype if config.mixed_precision else None,
            capture_routing=True,
        )
        routing_summary = {
            "encoder_routing": aggregate_routing_rows(val_final["routing"])["encoder_routing"],
            "decoder_routing": aggregate_routing_rows(val_final["routing"])["decoder_routing"],
            "test_encoder_routing": aggregate_routing_rows(test_final["routing"])["encoder_routing"],
            "test_decoder_routing": aggregate_routing_rows(test_final["routing"])["decoder_routing"],
        }
        atomic_write_json(run_dir / "routing_summary.json", routing_summary)

        duration = elapsed_training_time
        final_metrics = {
            "variant": variant,
            "seed": seed,
            "config_hash": cfg_hash,
            "encoder_residual": VARIANTS[variant]["encoder"],
            "decoder_residual": VARIANTS[variant]["decoder"],
            "validation_loss": val_final["loss"],
            "validation_accuracy": val_final["accuracy"],
            "test_loss": test_final["loss"],
            "test_accuracy": test_final["accuracy"],
            "family_accuracy_validation": val_final["family_accuracy"],
            "family_accuracy_test": test_final["family_accuracy"],
            "parameter_count": param_counts["total"],
            "parameter_increase_pct": param_increase_pct,
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_reserved,
            "examples_per_sec": examples_processed / max(1e-6, duration),
            "training_duration_sec": duration,
            "best_epoch": start_epoch,  # overwritten below from best checkpoint when present
            "best_validation_accuracy": best_val_accuracy,
            "checkpoint_last": str(last_path),
            "checkpoint_best": str(best_path),
            "resumed": resumed,
            "init_validation": init_meta,
        }
        if best_path.exists():
            best_ckpt = load_checkpoint(best_path, map_location="cpu")
            final_metrics["best_epoch"] = int(best_ckpt["epoch"])
        final_metrics["wandb"] = wandb_logger.metadata()
        atomic_write_json(run_dir / "final_test_metrics.json", final_metrics)
        wandb_logger.update_summary(
            {
                "final/validation_loss": final_metrics["validation_loss"],
                "final/validation_accuracy": final_metrics["validation_accuracy"],
                "final/test_loss": final_metrics["test_loss"],
                "final/test_accuracy": final_metrics["test_accuracy"],
                "final/family_accuracy_test": final_metrics["family_accuracy_test"],
                "final/best_epoch": final_metrics["best_epoch"],
                "final/best_validation_accuracy": final_metrics["best_validation_accuracy"],
                "final/parameter_count": final_metrics["parameter_count"],
                "final/parameter_increase_pct": final_metrics["parameter_increase_pct"],
                "final/peak_allocated_bytes": final_metrics["peak_allocated_bytes"],
                "final/training_duration_sec": final_metrics["training_duration_sec"],
                "checkpoint_best": final_metrics["checkpoint_best"],
                "checkpoint_last": final_metrics["checkpoint_last"],
            }
        )
        wandb_logger.log(
            {
                "test/loss": test_final["loss"],
                "test/accuracy": test_final["accuracy"],
                "test/family_accuracy": test_final["family_accuracy"],
            },
            step=global_step,
        )
        _save(last_path, epoch=config.max_epochs, batch_index=0, status="completed")
        mark_completed(run_dir)
        manifest.upsert(
            variant,
            seed,
            cfg_hash,
            status="completed",
            run_directory=str(run_dir),
            latest_checkpoint=str(last_path),
            best_checkpoint=str(best_path),
            current_epoch=final_metrics["best_epoch"],
            global_step=global_step,
            best_validation_accuracy=best_val_accuracy,
        )
        wandb_logger.finish(status="completed")
        return {"status": "completed", "run_dir": str(run_dir), "metrics": final_metrics, "resumed": resumed}
    except Exception as exc:  # noqa: BLE001 - persist failure into manifest for resume UX
        write_status(run_dir, {"status": "failed", "error": str(exc)})
        if last_path.exists() or global_step > 0:
            try:
                _save(last_path, epoch=start_epoch, batch_index=0, status="interrupted")
            except Exception:  # noqa: BLE001
                pass
        manifest.upsert(
            variant,
            seed,
            cfg_hash,
            status="failed",
            run_directory=str(run_dir),
            latest_checkpoint=str(last_path) if last_path.exists() else None,
            best_checkpoint=str(best_path) if best_path.exists() else None,
            current_epoch=start_epoch,
            global_step=global_step,
            best_validation_accuracy=best_val_accuracy if best_val_accuracy >= 0 else None,
            error_message=str(exc),
        )
        wandb_logger.update_summary({"error_message": str(exc)})
        wandb_logger.finish(status="failed")
        raise
