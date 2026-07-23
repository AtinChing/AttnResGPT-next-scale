from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.vlm_attnres import TinyAttnResVLM
from src.vlm.ablation.routing import collect_routing_batch_stats


def _bucket_accuracy(correct_map: dict[Any, int], total_map: dict[Any, int]) -> dict[str, Any]:
    return {
        str(key): {
            "accuracy": correct_map[key] / max(1, total_map[key]),
            "correct": correct_map[key],
            "total": total_map[key],
        }
        for key in sorted(total_map, key=lambda item: str(item))
    }


def _depth_bin(depth: int) -> str:
    if depth <= 3:
        return "1-3"
    if depth <= 6:
        return "4-6"
    if depth <= 9:
        return "7-9"
    return "10+"


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

    category_correct: dict[str, int] = defaultdict(int)
    category_total: dict[str, int] = defaultdict(int)
    family_correct: dict[int, int] = defaultdict(int)
    family_total: dict[int, int] = defaultdict(int)
    length_correct: dict[str, int] = defaultdict(int)
    length_total: dict[str, int] = defaultdict(int)
    depth_correct: dict[str, int] = defaultdict(int)
    depth_total: dict[str, int] = defaultdict(int)
    shape_correct = {"cube": 0, "cylinder": 0, "sphere": 0}
    shape_total = {"cube": 0, "cylinder": 0, "sphere": 0}

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

        categories = batch.get("reasoning_categories") or batch.get("families")
        family_indices = batch.get("question_family_indices") or [0] * batch_size
        length_bins = batch.get("program_length_bins") or ["unknown"] * batch_size
        depths = batch.get("dependency_depths") or [0] * batch_size
        depth_groups = [f"depth_{_depth_bin(int(value))}" for value in depths]

        for row in range(batch_size):
            position = int(answer_positions[row].item())
            pred_id = int(preds[row, position].item())
            target_id = int(answer_ids[row].item())
            hit = pred_id == target_id
            answer_nll_sum += float(-log_probs[row, position, target_id].item())
            category = categories[row]
            family = int(family_indices[row])
            length_bin = length_bins[row]
            depth_bin = _depth_bin(int(depths[row]))

            category_total[category] += 1
            family_total[family] += 1
            length_total[length_bin] += 1
            depth_total[depth_bin] += 1
            if hit:
                correct += 1
                category_correct[category] += 1
                family_correct[family] += 1
                length_correct[length_bin] += 1
                depth_correct[depth_bin] += 1

            for shape, key in (("cube", "mentions_cube"), ("cylinder", "mentions_cylinder"), ("sphere", "mentions_sphere")):
                flags = batch.get(key) or [False] * batch_size
                if flags[row]:
                    shape_total[shape] += 1
                    shape_correct[shape] += int(hit)

        if capture_routing:
            routing_rows.append(
                collect_routing_batch_stats(
                    model,
                    families=list(categories),
                    difficulty_levels=None,
                    group_labels=depth_groups,
                    group_key="by_program_depth",
                    prefix_length=int(output["prefix_length"]),
                    text_length=input_ids.size(1),
                )
            )

    if capture_routing:
        model.set_weight_capture(False)

    return {
        "loss": total_loss / max(1, total_examples),
        "answer_token_nll": answer_nll_sum / max(1, total_examples),
        "accuracy": correct / max(1, total_examples),
        "correct": correct,
        "total": total_examples,
        "category_accuracy": _bucket_accuracy(category_correct, category_total),
        "question_family_accuracy": _bucket_accuracy(family_correct, family_total),
        "program_length_accuracy": _bucket_accuracy(length_correct, length_total),
        "dependency_depth_accuracy": _bucket_accuracy(depth_correct, depth_total),
        "shape_accuracy": {
            shape: {
                "accuracy": shape_correct[shape] / max(1, shape_total[shape]),
                "correct": shape_correct[shape],
                "total": shape_total[shape],
            }
            for shape in shape_total
        },
        # Back-compat aliases used by older plot code paths.
        "family_accuracy": {
            key: value["accuracy"] for key, value in _bucket_accuracy(category_correct, category_total).items()
        },
        "routing": routing_rows,
    }
