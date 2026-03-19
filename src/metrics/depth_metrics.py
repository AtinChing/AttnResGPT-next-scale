from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch


def contribution_breakdown(
    mean_weights: Iterable[torch.Tensor],
    source_indices: Iterable[list[int]],
) -> dict[str, float]:
    embedding_values: list[float] = []
    early_values: list[float] = []
    late_values: list[float] = []
    entropy_values: list[float] = []

    for weights, indices in zip(mean_weights, source_indices):
        weights_cpu = weights.detach().cpu()
        if not indices:
            continue
        if 0 in indices:
            embedding_values.append(float(weights_cpu[indices.index(0)].item()))
        non_embedding_positions = [position for position, index in enumerate(indices) if index != 0]
        if non_embedding_positions:
            split = max(1, len(non_embedding_positions) // 2)
            early_positions = non_embedding_positions[:split]
            late_positions = non_embedding_positions[split:]
            early_values.append(float(weights_cpu[early_positions].sum().item()))
            late_values.append(float(weights_cpu[late_positions].sum().item()) if late_positions else 0.0)
        entropy = -(weights_cpu * weights_cpu.clamp_min(1e-8).log()).sum().item()
        entropy_values.append(float(entropy))

    def _mean(values: list[float]) -> float:
        return float(sum(values) / max(1, len(values)))

    return {
        "embedding_contribution": _mean(embedding_values),
        "early_contribution": _mean(early_values),
        "late_contribution": _mean(late_values),
        "depth_attention_entropy": _mean(entropy_values),
    }


def average_scalars(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in keys
    }


def average_depth_artifacts(
    depth_rows_batches: list[list[torch.Tensor]],
    source_indices_batches: list[list[list[int]]],
) -> dict[str, Any]:
    if not depth_rows_batches:
        return {
            "depth_attention_rows": [],
            "depth_source_indices": [],
            "mean_embedding_contribution": None,
            "mean_early_contribution": None,
            "mean_late_contribution": None,
            "mean_depth_attention_entropy": None,
        }

    sums: list[torch.Tensor] = []
    source_indices = source_indices_batches[0]
    for batch_rows in depth_rows_batches:
        if not sums:
            sums = [row.detach().cpu().clone() for row in batch_rows]
        else:
            for index, row in enumerate(batch_rows):
                sums[index] += row.detach().cpu()

    averages = [row / len(depth_rows_batches) for row in sums]
    contributions = contribution_breakdown(averages, source_indices)
    return {
        "depth_attention_rows": [row.tolist() for row in averages],
        "depth_source_indices": source_indices,
        "mean_embedding_contribution": contributions["embedding_contribution"],
        "mean_early_contribution": contributions["early_contribution"],
        "mean_late_contribution": contributions["late_contribution"],
        "mean_depth_attention_entropy": contributions["depth_attention_entropy"],
    }
