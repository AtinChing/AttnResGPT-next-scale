from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.vlm_attnres import TinyAttnResVLM
from src.vlm.ablation.routing import collect_routing_batch_stats


def _bucket_accuracy(correct_map: dict[Any, int], total_map: dict[Any, int]) -> dict[str, float]:
    return {
        str(key): correct_map[key] / max(1, total_map[key])
        for key in sorted(total_map, key=lambda item: str(item))
    }


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
    answer_nll_sum = 0.0
    total_examples = 0
    correct = 0

    family_correct: dict[str, int] = defaultdict(int)
    family_total: dict[str, int] = defaultdict(int)
    level_correct: dict[int, int] = defaultdict(int)
    level_total: dict[int, int] = defaultdict(int)
    hops_correct: dict[int, int] = defaultdict(int)
    hops_total: dict[int, int] = defaultdict(int)
    held_out_correct = 0
    held_out_total = 0
    degraded_correct = 0
    degraded_total = 0
    local_detail_correct = 0
    local_detail_total = 0
    compositional_correct = 0
    compositional_total = 0
    multi_hop_correct = 0
    multi_hop_total = 0

    # degradation strength bins: low/mid/high
    deg_correct = {"low": 0, "mid": 0, "high": 0}
    deg_total = {"low": 0, "mid": 0, "high": 0}

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
        difficulty_levels = batch.get("difficulty_levels")
        hops = batch.get("hops")
        degradation = batch.get("degradation_strength")
        held_out = batch.get("held_out")
        visual_degraded = batch.get("visual_degraded")

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
        log_probs = F.log_softmax(logits.float(), dim=-1)
        batch_size = input_ids.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        level_list = (
            difficulty_levels.tolist()
            if isinstance(difficulty_levels, torch.Tensor)
            else [0] * batch_size
        )
        hops_list = hops.tolist() if isinstance(hops, torch.Tensor) else [0] * batch_size
        deg_list = (
            degradation.tolist() if isinstance(degradation, torch.Tensor) else [0.0] * batch_size
        )
        held_list = held_out.tolist() if isinstance(held_out, torch.Tensor) else [False] * batch_size
        degraded_list = (
            visual_degraded.tolist()
            if isinstance(visual_degraded, torch.Tensor)
            else [False] * batch_size
        )

        for row in range(batch_size):
            position = int(answer_positions[row].item())
            pred_id = int(preds[row, position].item())
            target_id = int(answer_ids[row].item())
            family = families[row]
            level = int(level_list[row])
            hop = int(hops_list[row])
            deg = float(deg_list[row])
            is_held = bool(held_list[row])
            is_degraded = bool(degraded_list[row])
            answer_nll_sum += float(-log_probs[row, position, target_id].item())

            family_total[family] += 1
            level_total[level] += 1
            hops_total[hop] += 1
            hit = pred_id == target_id
            if hit:
                correct += 1
                family_correct[family] += 1
                level_correct[level] += 1
                hops_correct[hop] += 1

            if family == "local_detail":
                local_detail_total += 1
                local_detail_correct += int(hit)
            if family == "compositional":
                compositional_total += 1
                compositional_correct += int(hit)
            if family == "multi_hop":
                multi_hop_total += 1
                multi_hop_correct += int(hit)
            if is_held:
                held_out_total += 1
                held_out_correct += int(hit)
            if is_degraded or level >= 3:
                degraded_total += 1
                degraded_correct += int(hit)

            if deg < 0.25:
                bucket = "low"
            elif deg < 0.55:
                bucket = "mid"
            else:
                bucket = "high"
            deg_total[bucket] += 1
            deg_correct[bucket] += int(hit)

        if capture_routing:
            routing_rows.append(
                collect_routing_batch_stats(
                    model,
                    families=families,
                    difficulty_levels=[int(value) for value in level_list],
                    prefix_length=int(output["prefix_length"]),
                    text_length=input_ids.size(1),
                )
            )

    if capture_routing:
        model.set_weight_capture(False)

    accuracy = correct / max(1, total_examples)
    return {
        "loss": total_loss / max(1, total_examples),
        "answer_token_nll": answer_nll_sum / max(1, total_examples),
        "accuracy": accuracy,
        "correct": correct,
        "total": total_examples,
        "family_accuracy": _bucket_accuracy(family_correct, family_total),
        "family_total": dict(family_total),
        "level_accuracy": _bucket_accuracy(level_correct, level_total),
        "level_total": {str(key): value for key, value in level_total.items()},
        "hops_accuracy": _bucket_accuracy(hops_correct, hops_total),
        "hops_total": {str(key): value for key, value in hops_total.items()},
        "held_out_accuracy": held_out_correct / max(1, held_out_total),
        "held_out_total": held_out_total,
        "visual_degraded_accuracy": degraded_correct / max(1, degraded_total),
        "visual_degraded_total": degraded_total,
        "local_detail_accuracy": local_detail_correct / max(1, local_detail_total),
        "compositional_accuracy": compositional_correct / max(1, compositional_total),
        "multi_hop_accuracy": multi_hop_correct / max(1, multi_hop_total),
        "degradation_bin_accuracy": {
            key: deg_correct[key] / max(1, deg_total[key]) for key in deg_total
        },
        "routing": routing_rows,
    }
