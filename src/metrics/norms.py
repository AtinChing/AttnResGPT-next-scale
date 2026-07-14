from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict

import torch
import torch.nn as nn


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def second_half_language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    half_start = targets.size(1) // 2
    tail_logits = logits[:, half_start:, :]
    tail_targets = targets[:, half_start:]
    return language_model_loss(tail_logits, tail_targets)


def position_wise_language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='none',
    )
    per_token = per_token.view_as(targets)
    return per_token.mean(dim=0)


def perplexity_from_loss(loss: float) -> float:
    return float(torch.exp(torch.tensor(min(loss, 20.0))).item())


def _magnitude(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm(dim=-1).mean().item())


def _layer_input_site(module_name: str) -> str | None:
    """Return the canonical depth site for a transformer's pre-attention input."""
    suffix = ".attn_norm"
    if not module_name.endswith(suffix):
        return None
    site = module_name[: -len(suffix)]
    parts = site.split(".")
    if len(parts) < 2 or parts[-2] != "blocks" or not parts[-1].isdigit():
        return None
    return site


def _layer_number(site: str) -> int:
    parts = site.split(".")
    for index, part in enumerate(parts[:-1]):
        if part == "blocks":
            try:
                return int(parts[index + 1])
            except ValueError:
                break
    return -1


@dataclass
class LayerInputMagnitudeTracker:
    """Measure paper-style ``h_l`` and ``dL/dh_l`` at every transformer layer.

    The canonical site is the tensor entering each block's pre-attention RMSNorm:
    baseline ``x`` and Full/Block AttnRes ``attn_input``. Forward and gradient
    magnitudes therefore use one identical definition across all architectures.
    """

    input_sums: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    input_counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    gradient_sums: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    gradient_counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def register(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            site = _layer_input_site(name)
            if site is not None:
                self.handles.append(module.register_forward_pre_hook(self._forward_pre_hook(site)))

    def _forward_pre_hook(self, site: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            layer_input = inputs[0]
            self.input_sums[site] += _magnitude(layer_input)
            self.input_counts[site] += 1
            if layer_input.requires_grad:
                layer_input.register_hook(self._gradient_hook(site))

        return hook

    def _gradient_hook(self, site: str):
        def hook(gradient: torch.Tensor) -> None:
            self.gradient_sums[site] += _magnitude(gradient)
            self.gradient_counts[site] += 1

        return hook

    @staticmethod
    def _averages(sums: DefaultDict[str, float], counts: DefaultDict[str, int]) -> dict[str, float]:
        sites = sorted(sums, key=lambda site: (_layer_number(site), site))
        return {site: sums[site] / counts[site] for site in sites if counts[site] > 0}

    def reset_step(self) -> None:
        self.input_sums.clear()
        self.input_counts.clear()
        self.gradient_sums.clear()
        self.gradient_counts.clear()

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "layer_input_magnitudes": self._averages(self.input_sums, self.input_counts),
            "layer_input_gradient_magnitudes": self._averages(self.gradient_sums, self.gradient_counts),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def last_layer_input_magnitude(layer_input_magnitudes: dict[str, float]) -> float | None:
    if not layer_input_magnitudes:
        return None
    last_site = max(layer_input_magnitudes, key=lambda site: (_layer_number(site), site))
    return float(layer_input_magnitudes[last_site])


# Full AttnRes layer-1 pre-attn history: [emb, L0 attn out, L0 mlp out].
_FULL_LAYER1_SOURCE_NAMES = ("emb", "l0_attn", "l0_mlp")
# Block AttnRes layer-1 (second layer of first block): [emb, partial after L0].
_BLOCK_LAYER1_SOURCE_NAMES = ("emb", "partial_after_l0")


@dataclass
class Layer1DepthAttentionProbe:
    """Log layer-1 depth-attention weights and raw source magnitudes.

    Tracks the pre-attention ``DepthAttentionResidual`` at transformer layer 1
    (index 1). Used to monitor the Full AttnRes layer-1 magnitude spike seen at
    90M: softmax concentrating on a large L0-attn source. No-op for baseline.
    """

    architecture: str = "baseline"
    weight_sums: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    mag_sums: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    mixed_mag_sum: float = 0.0
    entropy_sum: float = 0.0
    n_sources: int = 0
    counts: int = 0
    _wrapped: Any = None
    _original_forward: Any = None

    def register(self, model: nn.Module) -> None:
        architecture = getattr(getattr(model, "config", None), "architecture", "baseline")
        mode = getattr(getattr(getattr(model, "config", None), "attnres", None), "mode", None)
        if architecture == "block_attnres" or (architecture == "attnres" and mode == "block"):
            self.architecture = "block_attnres"
        elif architecture == "attnres":
            self.architecture = "attnres"
        else:
            self.architecture = "baseline"
            return
        if not hasattr(model, "blocks") or len(model.blocks) < 2:
            return
        mixer = model.blocks[1].pre_attn_res
        self._wrapped = mixer
        self._original_forward = mixer.forward

        def wrapped_forward(history, *, return_stats: bool = False):
            mixed, stats = self._original_forward(history, return_stats=return_stats)
            self._record(history, mixed, stats if return_stats else None, mixer)
            return mixed, stats

        mixer.forward = wrapped_forward  # type: ignore[method-assign]

    def _source_names(self, n_sources: int) -> tuple[str, ...]:
        if self.architecture == "attnres" and n_sources == len(_FULL_LAYER1_SOURCE_NAMES):
            return _FULL_LAYER1_SOURCE_NAMES
        if self.architecture == "block_attnres" and n_sources == len(_BLOCK_LAYER1_SOURCE_NAMES):
            return _BLOCK_LAYER1_SOURCE_NAMES
        return tuple(f"source_{index}" for index in range(n_sources))

    def _record(self, history, mixed: torch.Tensor, stats: dict | None, mixer: nn.Module) -> None:
        selected_history, _indices = mixer._select_history(history)
        n_sources = len(selected_history)
        self.n_sources = n_sources
        names = self._source_names(n_sources)
        if stats is not None and "mean_weights" in stats:
            mean_weights = stats["mean_weights"]
            weights = [float(mean_weights[i].item()) for i in range(n_sources)]
            entropy = float(stats.get("entropy", 0.0))
        else:
            values = torch.stack(selected_history, dim=0)
            keys = mixer.key_norm(values)
            logits = torch.einsum("d,sbtd->sbt", mixer.query, keys) / max(mixer.temperature, 1e-6)
            weight_tensor = torch.softmax(logits, dim=0)
            weights = weight_tensor.detach().float().mean(dim=(1, 2)).tolist()
            entropy = float(
                (-(weight_tensor.detach().float() * weight_tensor.detach().float().clamp_min(1e-8).log())
                 .sum(dim=0)
                 .mean())
                .item()
            )
        for name, source, weight in zip(names, selected_history, weights):
            self.weight_sums[name] += float(weight)
            self.mag_sums[name] += _magnitude(source)
        self.mixed_mag_sum += _magnitude(mixed)
        self.entropy_sum += entropy
        self.counts += 1

    def reset_step(self) -> None:
        self.weight_sums.clear()
        self.mag_sums.clear()
        self.mixed_mag_sum = 0.0
        self.entropy_sum = 0.0
        self.counts = 0

    def snapshot(self) -> dict[str, Any]:
        if self.architecture == "baseline" or self.counts == 0:
            return {}
        scale = float(self.counts)
        payload: dict[str, Any] = {
            "layer1_depth_attn/n_sources": float(self.n_sources),
            "layer1_depth_attn/entropy": self.entropy_sum / scale,
            "layer1_depth_attn/mixed_mag": self.mixed_mag_sum / scale,
        }
        for name in self.weight_sums:
            payload[f"layer1_depth_attn/weight_{name}"] = self.weight_sums[name] / scale
            payload[f"layer1_depth_attn/mag_{name}"] = self.mag_sums[name] / scale
        return payload

    def close(self) -> None:
        if self._wrapped is not None and self._original_forward is not None:
            self._wrapped.forward = self._original_forward
        self._wrapped = None
        self._original_forward = None
