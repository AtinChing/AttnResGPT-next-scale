from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import torch

from src.data.dataset import build_dataloaders
from src.metrics.depth_metrics import average_depth_artifacts
from src.metrics.norms import (
    LayerInputMagnitudeTracker,
    last_layer_input_magnitude,
    language_model_loss,
    perplexity_from_loss,
    position_wise_language_model_loss,
    second_half_language_model_loss,
)
from src.models.attnres import build_model
from src.utils.config import Config, load_config
from src.utils.runtime import amp_dtype_from_string, count_parameters, get_device


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: Optional[int] = None,
    collect_artifacts: bool = True,
) -> dict[str, Any]:
    model.eval()
    use_autocast = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}
    layer_input_tracker = LayerInputMagnitudeTracker()
    layer_input_tracker.register(model)

    losses: list[float] = []
    second_half_losses: list[float] = []
    depth_rows_batches: list[list[torch.Tensor]] = []
    depth_source_indices_batches: list[list[list[int]]] = []

    try:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_autocast,
            )
            with autocast_context if use_autocast else nullcontext():
                logits, aux = model(input_ids, return_aux=collect_artifacts)
                loss = language_model_loss(logits, targets)
                second_half_loss = second_half_language_model_loss(logits, targets)

            losses.append(float(loss.item()))
            second_half_losses.append(float(second_half_loss.item()))

            if collect_artifacts:
                rows = aux.get("depth_attention_rows", [])
                source_indices = aux.get("depth_source_indices", [])
                if rows:
                    depth_rows_batches.append(rows)
                    depth_source_indices_batches.append(source_indices)
    finally:
        layer_input_magnitudes = layer_input_tracker.snapshot()["layer_input_magnitudes"]
        layer_input_tracker.close()

    mean_loss = sum(losses) / max(1, len(losses))
    mean_second_half_loss = sum(second_half_losses) / max(1, len(second_half_losses))
    metrics: dict[str, Any] = {
        "val_loss": mean_loss,
        "perplexity": perplexity_from_loss(mean_loss),
        "second_half_loss": mean_second_half_loss,
        "mean_layer_input_magnitudes": layer_input_magnitudes,
        "mean_layer_input_magnitude_last_layer": last_layer_input_magnitude(layer_input_magnitudes),
    }

    metrics.update(average_depth_artifacts(depth_rows_batches, depth_source_indices_batches))
    return metrics


@torch.no_grad()
def evaluate_positionwise_loss(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: Optional[int] = None,
) -> dict[str, Any]:
    model.eval()
    use_autocast = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}

    loss_sums: torch.Tensor | None = None
    examples_seen = 0

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        autocast_context = torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_autocast,
        )
        with autocast_context if use_autocast else nullcontext():
            logits, _ = model(input_ids, return_aux=False)
            position_losses = position_wise_language_model_loss(logits, targets)

        position_losses = position_losses.detach().float().cpu()
        if loss_sums is None:
            loss_sums = torch.zeros_like(position_losses)
        batch_size = int(targets.size(0))
        loss_sums += position_losses * batch_size
        examples_seen += batch_size

    if loss_sums is None:
        return {
            "position_losses": [],
            "mean_position_loss": None,
            "first_half_position_loss": None,
            "second_half_position_loss": None,
            "last_token_position_loss": None,
            "max_position_loss": None,
            "max_position_index": None,
        }

    mean_losses = loss_sums / max(1, examples_seen)
    half_start = mean_losses.numel() // 2
    max_index = int(torch.argmax(mean_losses).item())
    return {
        "position_losses": mean_losses.tolist(),
        "mean_position_loss": float(mean_losses.mean().item()),
        "first_half_position_loss": float(mean_losses[:half_start].mean().item()) if half_start > 0 else float(mean_losses.mean().item()),
        "second_half_position_loss": float(mean_losses[half_start:].mean().item()),
        "last_token_position_loss": float(mean_losses[-1].item()),
        "max_position_loss": float(mean_losses[max_index].item()),
        "max_position_index": max_index,
    }


def load_checkpoint_model(config: Config, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    model = build_model(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model


def write_json_output(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".jsonl":
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        return
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained baseline or AttnRes checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file.")
    parser.add_argument("--model", choices=["baseline", "attnres", "block_attnres"], default=None)
    parser.add_argument("--overrides", nargs="*", default=[], help="Optional key=value config overrides.")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path. Appends JSONL when the suffix is .jsonl; otherwise writes one JSON file.",
    )
    args = parser.parse_args()

    overrides = list(args.overrides)
    if args.model is not None:
        overrides.append(f"model.architecture={args.model}")

    config = load_config(args.config, overrides=overrides)
    tokenizer, _, val_loader, data_meta = build_dataloaders(config)
    config.model.vocab_size = tokenizer.vocab_size

    device = get_device(config.training.device)
    amp_dtype = amp_dtype_from_string(config.training.amp_dtype)
    model = load_checkpoint_model(config, args.checkpoint, device)
    max_batches = args.max_batches or config.evaluation.max_batches
    metrics = evaluate_model(
        model,
        val_loader,
        device=device,
        amp_dtype=amp_dtype,
        max_batches=max_batches,
        collect_artifacts=True,
    )
    counts = count_parameters(model)

    summary = {
        "checkpoint_path": str(Path(args.checkpoint)),
        "config_path": str(Path(args.config)),
        "max_batches": max_batches,
        "model": config.model.architecture,
        "size": config.model.size_name,
        "context": config.data.block_size,
        "dataset": data_meta.get("dataset", config.data.dataset_name),
        "tokenizer_name": tokenizer.name,
        "parameter_count_total": counts["total"],
        "parameter_count_trainable": counts["trainable"],
        **metrics,
    }
    for key, value in summary.items():
        print(f"{key}: {value}")
    if args.output_json is not None:
        write_json_output(args.output_json, summary)
        print(f"wrote_json: {args.output_json}")


if __name__ == "__main__":
    main()
