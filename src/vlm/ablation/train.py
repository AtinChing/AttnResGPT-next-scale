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
from src.vlm.clevr.dataset import CLEVRExampleDataset, collate_clevr_batch
from src.vlm.clevr.prepare import PreparedBenchmark
from src.vlm.clevr.validate import majority_answer_baseline


def _select_amp_dtype(requested: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if requested == "auto":
        major, _ = torch.cuda.get_device_capability(device)
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
    prepared: PreparedBenchmark,
    *,
    control_mode: str = "none",
) -> dict[str, DataLoader]:
    collate = partial(collate_clevr_batch, pad_token_id=prepared.tokenizer.pad_token_id)
    loaders: dict[str, DataLoader] = {}
    for split, examples in prepared.split_examples.items():
        dataset = CLEVRExampleDataset(
            examples=examples,
            image_root=prepared.image_root,
            image_prefix=prepared.image_prefix_by_split[split],
            tokenizer=prepared.tokenizer,
            preprocess=prepared.preprocess,
            supervise_eos=config.supervise_eos,
            allow_unk=(split != "train"),
            control_mode=control_mode if split != "train" else "none",
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


def _run_controls(
    *,
    config: AblationExperimentConfig,
    prepared: PreparedBenchmark,
    model: TinyAttnResVLM,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    controls: dict[str, Any] = {}
    controls["majority_answer"] = {
        "validation": majority_answer_baseline(prepared.split_examples["train"], prepared.split_examples["validation"]),
        "test": majority_answer_baseline(prepared.split_examples["train"], prepared.split_examples["test"]),
    }
    for mode in ("question_only", "blank_question"):
        loaders = build_dataloaders(config, prepared, control_mode=mode)
        controls[mode] = {
            "validation": evaluate_model(
                model,
                loaders["validation"],
                device=device,
                amp_dtype=amp_dtype if config.mixed_precision else None,
                capture_routing=False,
            ),
            "test": evaluate_model(
                model,
                loaders["test"],
                device=device,
                amp_dtype=amp_dtype if config.mixed_precision else None,
                capture_routing=False,
            ),
        }
    return controls


def train_variant_seed(
    config: AblationExperimentConfig,
    *,
    prepared: PreparedBenchmark,
    variant: str,
    seed: int,
    project_root: Path,
    manifest: ExperimentManifest,
    source_code_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    cfg_hash = config_hash(config)
    run_dir = run_dir_for(project_root, config.benchmark, variant, seed, cfg_hash)
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
    prepared.tokenizer.save(run_dir / "tokenizer.json")
    atomic_write_json(run_dir / "subset_manifest.json", prepared.manifest)
    atomic_write_json(run_dir / "preprocess.json", prepared.preprocess.to_dict())

    loaders = build_dataloaders(config, prepared)
    tokenizer = prepared.tokenizer

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
    steps_per_epoch = max(1, math.ceil(len(loaders["train"]) / max(1, config.grad_accum_steps)))
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
    micro_step = 0

    if config.resume and last_path.exists() and not config.force_restart:
        checkpoint = load_checkpoint(last_path, map_location=device)
        validate_checkpoint_compatibility(
            checkpoint,
            variant=variant,
            seed=seed,
            config_hash=cfg_hash,
            source_code_hash=source_code_hash,
            force_restart=config.force_restart,
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
        elapsed_training_time = float(checkpoint.get("elapsed_training_time", 0.0))
        resumed = True

    param_counts = count_parameters(model)
    baseline_params = count_parameters(reference)["total"]
    param_increase_pct = 100.0 * (param_counts["total"] - baseline_params) / max(1, baseline_params)

    wandb_logger = AblationWandbLogger(
        config=config,
        variant=variant,
        seed=seed,
        config_hash=cfg_hash,
        run_dir=run_dir,
        extra_config={
            "benchmark": config.benchmark,
            "dataset_version": config.dataset_version,
            "subset_manifest_hash": config.subset_manifest_hash,
            "vocab_hash": config.vocab_hash,
            "preprocess_hash": config.preprocess_hash,
            "parameter_count": param_counts,
            "parameter_increase_pct": param_increase_pct,
            "init_validation": init_meta,
            "source_code_hash": source_code_hash,
        },
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
                "variant": variant,
                "benchmark": config.benchmark,
                "dataset_version": config.dataset_version,
                "benchmark_mode": config.benchmark_mode,
                "subset_manifest_hash": config.subset_manifest_hash,
                "vocab_hash": config.vocab_hash,
                "preprocess_hash": config.preprocess_hash,
                "source_code_hash": source_code_hash,
            },
            dataset_config=prepared.to_meta(),
            tokenizer_vocab=tokenizer.token_to_id,
            variant=variant,
            seed=seed,
            config_hash=cfg_hash,
            source_code_hash=source_code_hash,
            status=status,
        )
        save_checkpoint(path, payload)
        write_status(
            run_dir,
            {
                "status": status,
                "epoch": epoch,
                "global_step": global_step,
                "best_val_accuracy": best_val_accuracy,
                "checkpoint": str(path),
            },
        )

    peak_alloc = 0
    peak_reserved = 0
    train_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        for epoch in range(start_epoch, config.max_epochs):
            model.train()
            epoch_t0 = time.perf_counter()
            epoch_loss = 0.0
            epoch_examples = 0
            optimizer.zero_grad(set_to_none=True)
            for batch_index, batch in enumerate(loaders["train"]):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                targets = batch["targets"].to(device)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None)
                    if device.type == "cuda" and config.mixed_precision
                    else nullcontext()
                )
                with autocast:
                    output = model(pixel_values=pixel_values, input_ids=input_ids, targets=targets)
                    loss = output["loss"] / max(1, config.grad_accum_steps)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_step += 1
                batch_size = input_ids.size(0)
                epoch_loss += float(output["loss"].item()) * batch_size
                epoch_examples += batch_size
                examples_processed += batch_size

                if micro_step % max(1, config.grad_accum_steps) == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    if config.wandb_log_interval > 0 and global_step % config.wandb_log_interval == 0:
                        wandb_logger.log(
                            {
                                "train/loss": float(output["loss"].item()),
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
                    "answer_token_nll": val_metrics.get("answer_token_nll"),
                    "category_accuracy": val_metrics.get("category_accuracy"),
                    "program_length_accuracy": val_metrics.get("program_length_accuracy"),
                    "dependency_depth_accuracy": val_metrics.get("dependency_depth_accuracy"),
                },
            )
            routing_summary = aggregate_routing_rows(val_metrics["routing"])
            append_jsonl(
                run_dir / "routing_metrics.jsonl",
                {"epoch": epoch, "global_step": global_step, **routing_summary},
            )
            wandb_logger.log(
                {
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/answer_token_nll": val_metrics.get("answer_token_nll"),
                    "val/category_accuracy": val_metrics.get("family_accuracy"),
                },
                step=global_step,
            )
            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = float(val_metrics["accuracy"])
                bad_epochs = 0
                _save(best_path, epoch=epoch + 1, batch_index=0)
            else:
                bad_epochs += 1
            _save(last_path, epoch=epoch + 1, batch_index=0)
            if device.type == "cuda":
                peak_alloc = max(peak_alloc, int(torch.cuda.max_memory_allocated(device)))
                peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
            if bad_epochs >= config.early_stopping_patience:
                break

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
        val_routing = aggregate_routing_rows(val_final["routing"])
        test_routing = aggregate_routing_rows(test_final["routing"])
        routing_summary = {
            "encoder_routing": val_routing["encoder_routing"],
            "decoder_routing": val_routing["decoder_routing"],
            "test_encoder_routing": test_routing["encoder_routing"],
            "test_decoder_routing": test_routing["decoder_routing"],
            "by_program_depth_validation": val_routing.get("by_program_depth", {}),
            "by_program_depth_test": test_routing.get("by_program_depth", {}),
        }
        atomic_write_json(run_dir / "routing_summary.json", routing_summary)

        controls = {}
        if config.run_controls:
            controls = _run_controls(
                config=config,
                prepared=prepared,
                model=model,
                device=device,
                amp_dtype=amp_dtype,
            )
            atomic_write_json(run_dir / "controls.json", controls)

        duration = elapsed_training_time
        a_to_b_drop = None
        if config.benchmark == "clevr_cogent_v1":
            a_to_b_drop = float(val_final["accuracy"]) - float(test_final["accuracy"])

        final_metrics = {
            "variant": variant,
            "seed": seed,
            "benchmark": config.benchmark,
            "dataset_version": config.dataset_version,
            "benchmark_mode": config.benchmark_mode,
            "config_hash": cfg_hash,
            "subset_manifest_hash": config.subset_manifest_hash,
            "vocab_hash": config.vocab_hash,
            "preprocess_hash": config.preprocess_hash,
            "source_code_hash": source_code_hash,
            "encoder_residual": VARIANTS[variant]["encoder"],
            "decoder_residual": VARIANTS[variant]["decoder"],
            "validation_loss": val_final["loss"],
            "validation_accuracy": val_final["accuracy"],
            "validation_answer_token_nll": val_final.get("answer_token_nll"),
            "test_loss": test_final["loss"],
            "test_accuracy": test_final["accuracy"],
            "test_answer_token_nll": test_final.get("answer_token_nll"),
            "test_label": (
                "held_out_official_validation_subset"
                if config.benchmark == "clevr_v1"
                else "condition_B_validation_subset"
            ),
            "condition_A_validation_accuracy": val_final["accuracy"] if config.benchmark == "clevr_cogent_v1" else None,
            "condition_B_test_accuracy": test_final["accuracy"] if config.benchmark == "clevr_cogent_v1" else None,
            "a_to_b_accuracy_drop": a_to_b_drop,
            "category_accuracy_validation": val_final.get("category_accuracy"),
            "category_accuracy_test": test_final.get("category_accuracy"),
            "program_length_accuracy_test": test_final.get("program_length_accuracy"),
            "dependency_depth_accuracy_test": test_final.get("dependency_depth_accuracy"),
            "question_family_accuracy_test": test_final.get("question_family_accuracy"),
            "shape_accuracy_test": test_final.get("shape_accuracy"),
            "family_accuracy_test": test_final.get("family_accuracy"),
            "controls": {
                key: {
                    "validation_accuracy": (value.get("validation") or {}).get("accuracy")
                    if isinstance(value, dict) and "validation" in value
                    else (value.get("validation") or {}).get("accuracy"),
                    "test_accuracy": (value.get("test") or {}).get("accuracy"),
                }
                if key != "majority_answer"
                else value
                for key, value in controls.items()
            },
            "parameter_count": param_counts["total"],
            "parameter_increase_pct": param_increase_pct,
            "peak_allocated_bytes": peak_alloc,
            "peak_reserved_bytes": peak_reserved,
            "examples_per_sec": examples_processed / max(1e-6, duration),
            "training_duration_sec": duration,
            "best_epoch": start_epoch,
            "best_validation_accuracy": best_val_accuracy,
            "checkpoint_last": str(last_path),
            "checkpoint_best": str(best_path),
            "resumed": resumed,
            "init_validation": init_meta,
            "label": "compute_constrained_official_subset",
        }
        if best_path.exists():
            best_ckpt = load_checkpoint(best_path, map_location="cpu")
            final_metrics["best_epoch"] = int(best_ckpt["epoch"])
        final_metrics["wandb"] = wandb_logger.metadata()
        atomic_write_json(run_dir / "final_test_metrics.json", final_metrics)
        wandb_logger.update_summary(
            {
                "final/validation_accuracy": final_metrics["validation_accuracy"],
                "final/test_accuracy": final_metrics["test_accuracy"],
                "final/test_answer_token_nll": final_metrics["test_answer_token_nll"],
                "final/a_to_b_accuracy_drop": final_metrics["a_to_b_accuracy_drop"],
            }
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
    except Exception as exc:  # noqa: BLE001
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
        )
        wandb_logger.finish(status="failed")
        raise
