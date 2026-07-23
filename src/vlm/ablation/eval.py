from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.models.vlm_attnres import TinyAttnResVLM
from src.vlm.ablation.routing import collect_routing_batch_stats


@torch.no_grad()
def evaluate_model(
    model: TinyAttnResVLM,
    loader: DataLoader,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
    capture_routing: bool = False,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    if capture_routing:
        model.set_weight_capture(True)

    total_loss = 0.0
    total_examples = 0
    correct = 0
    family_correct: dict[str, int] = defaultdict(int)
    family_total: dict[str, int] = defaultdict(int)
    routing_rows: list[dict[str, Any]] = []

    autocast_enabled = device.type == "cuda" and amp_dtype is not None
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)
        answer_positions = batch["answer_positions"].to(device)
        answer_ids = batch["answer_ids"].to(device)
        families = batch["families"]

        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
            if autocast_enabled
            else nullcontext()
        )
        with autocast:
            output = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                targets=targets,
                return_aux=capture_routing,
            )
        loss = output["loss"]
        logits = output["logits"]
        preds = logits.argmax(dim=-1)
        batch_size = input_ids.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        for row in range(batch_size):
            position = int(answer_positions[row].item())
            pred_id = int(preds[row, position].item())
            target_id = int(answer_ids[row].item())
            family = families[row]
            family_total[family] += 1
            if pred_id == target_id:
                correct += 1
                family_correct[family] += 1

        if capture_routing:
            routing_rows.append(
                collect_routing_batch_stats(
                    model,
                    families=families,
                    prefix_length=int(output["prefix_length"]),
                    text_length=input_ids.size(1),
                )
            )

    if capture_routing:
        model.set_weight_capture(False)

    accuracy = correct / max(1, total_examples)
    family_accuracy = {
        family: family_correct[family] / max(1, family_total[family])
        for family in sorted(family_total)
    }
    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": accuracy,
        "correct": correct,
        "total": total_examples,
        "family_accuracy": family_accuracy,
        "family_total": dict(family_total),
        "routing": routing_rows,
    }
