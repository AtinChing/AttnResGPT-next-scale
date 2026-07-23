from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from src.models.vlm_attnres import TinyAttnResVLM


def _stats_from_weights(weights: torch.Tensor, source_indices: list[int]) -> dict[str, Any]:
    mean_weights = weights.float().mean(dim=(1, 2))
    entropy = -(weights.float() * weights.float().clamp_min(1e-8).log()).sum(dim=0).mean()
    max_prob = weights.float().amax(dim=0).mean()
    effective = torch.exp(entropy)
    embedding = 0.0
    if 0 in source_indices:
        embedding = float(mean_weights[source_indices.index(0)].item())
    return {
        "source_indices": list(source_indices),
        "mean_weights": mean_weights.cpu().tolist(),
        "entropy": float(entropy.item()),
        "max_source_prob": float(max_prob.item()),
        "effective_sources": float(effective.item()),
        "embedding_contribution": embedding,
    }


def _site_group_stats(
    weights: torch.Tensor,
    source_indices: list[int],
    groups: list[str],
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups)):
        mask = torch.tensor([item == group for item in groups], dtype=torch.bool)
        if not bool(mask.any()):
            continue
        group_weights = weights[:, mask, :]
        payload[group] = _stats_from_weights(group_weights, source_indices)
    return payload


def collect_routing_batch_stats(
    model: TinyAttnResVLM,
    *,
    families: list[str],
    prefix_length: int,
    text_length: int,
    difficulty_levels: list[int] | None = None,
    group_labels: list[str] | None = None,
    group_key: str = "by_program_depth",
) -> dict[str, Any]:
    encoder_sites: list[dict[str, Any]] = []
    decoder_sites: list[dict[str, Any]] = []
    level_groups = (
        [f"level_{level}" for level in difficulty_levels]
        if difficulty_levels is not None
        else []
    )
    extra_groups = group_labels or []

    for site_index, residual in enumerate(model.iter_encoder_depth_residuals()):
        if residual.last_weights is None:
            continue
        weights = residual.last_weights
        indices = residual.last_source_indices
        site = {
            "site_index": site_index,
            **_stats_from_weights(weights, indices),
            "by_family": _site_group_stats(weights, indices, families),
            "by_difficulty": _site_group_stats(weights, indices, level_groups) if level_groups else {},
            group_key: _site_group_stats(weights, indices, extra_groups) if extra_groups else {},
        }
        encoder_sites.append(site)

    for site_index, residual in enumerate(model.iter_decoder_depth_residuals()):
        if residual.last_weights is None:
            continue
        weights = residual.last_weights
        indices = residual.last_source_indices
        seq_len = int(weights.size(2))
        vision_slice = weights[:, :, : min(prefix_length, seq_len)]
        text_slice = weights[:, :, prefix_length : prefix_length + text_length]
        site = {
            "site_index": site_index,
            **_stats_from_weights(weights, indices),
            "by_family": _site_group_stats(weights, indices, families),
            "by_difficulty": _site_group_stats(weights, indices, level_groups) if level_groups else {},
            group_key: _site_group_stats(weights, indices, extra_groups) if extra_groups else {},
            "visual_prefix": _stats_from_weights(vision_slice, indices) if vision_slice.numel() else {},
            "text_positions": _stats_from_weights(text_slice, indices) if text_slice.numel() else {},
        }
        decoder_sites.append(site)

    return {
        "encoder_routing": encoder_sites,
        "decoder_routing": decoder_sites,
    }


def aggregate_routing_rows(rows: list[dict[str, Any]], *, group_key: str = "by_program_depth") -> dict[str, Any]:
    def _aggregate_sites(namespace: str) -> list[dict[str, Any]]:
        if not rows:
            return []
        n_sites = max((len(row.get(namespace, [])) for row in rows), default=0)
        aggregated: list[dict[str, Any]] = []
        for site_index in range(n_sites):
            entropies: list[float] = []
            max_probs: list[float] = []
            effective: list[float] = []
            embeddings: list[float] = []
            mean_weight_sums: list[list[float]] = []
            source_indices: list[int] = []
            family_buckets: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: {"entropy": [], "embedding": [], "max_source_prob": []}
            )
            group_buckets: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: {"entropy": [], "embedding": [], "max_source_prob": []}
            )
            for row in rows:
                sites = row.get(namespace, [])
                if site_index >= len(sites):
                    continue
                site = sites[site_index]
                source_indices = site.get("source_indices", source_indices)
                entropies.append(float(site["entropy"]))
                max_probs.append(float(site["max_source_prob"]))
                effective.append(float(site["effective_sources"]))
                embeddings.append(float(site["embedding_contribution"]))
                mean_weight_sums.append([float(value) for value in site["mean_weights"]])
                for family, family_stats in site.get("by_family", {}).items():
                    family_buckets[family]["entropy"].append(float(family_stats["entropy"]))
                    family_buckets[family]["embedding"].append(float(family_stats["embedding_contribution"]))
                    family_buckets[family]["max_source_prob"].append(float(family_stats["max_source_prob"]))
                for label, stats in site.get(group_key, {}).items():
                    group_buckets[label]["entropy"].append(float(stats["entropy"]))
                    group_buckets[label]["embedding"].append(float(stats["embedding_contribution"]))
                    group_buckets[label]["max_source_prob"].append(float(stats["max_source_prob"]))
            if not mean_weight_sums:
                continue
            width = len(mean_weight_sums[0])
            mean_weights = [
                sum(row[index] for row in mean_weight_sums) / len(mean_weight_sums)
                for index in range(width)
            ]
            aggregated.append(
                {
                    "site_index": site_index,
                    "source_indices": source_indices,
                    "mean_weights": mean_weights,
                    "entropy": sum(entropies) / max(1, len(entropies)),
                    "max_source_prob": sum(max_probs) / max(1, len(max_probs)),
                    "effective_sources": sum(effective) / max(1, len(effective)),
                    "embedding_contribution": sum(embeddings) / max(1, len(embeddings)),
                    "by_family": {
                        family: {key: sum(values) / max(1, len(values)) for key, values in bucket.items()}
                        for family, bucket in family_buckets.items()
                    },
                    group_key: {
                        label: {key: sum(values) / max(1, len(values)) for key, values in bucket.items()}
                        for label, bucket in group_buckets.items()
                    },
                }
            )
        return aggregated

    def _group_summary(sites: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        buckets: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"entropy": [], "embedding": [], "max_source_prob": []}
        )
        for site in sites:
            for label, stats in site.get(group_key, {}).items():
                buckets[label]["entropy"].append(float(stats.get("entropy", 0.0)))
                buckets[label]["embedding"].append(float(stats.get("embedding", 0.0)))
                buckets[label]["max_source_prob"].append(float(stats.get("max_source_prob", 0.0)))
        return {
            label: {key: sum(values) / max(1, len(values)) for key, values in bucket.items()}
            for label, bucket in buckets.items()
        }

    encoder_sites = _aggregate_sites("encoder_routing")
    decoder_sites = _aggregate_sites("decoder_routing")
    return {
        "encoder_routing": encoder_sites,
        "decoder_routing": decoder_sites,
        group_key: {
            "encoder": _group_summary(encoder_sites),
            "decoder": _group_summary(decoder_sites),
        },
    }
