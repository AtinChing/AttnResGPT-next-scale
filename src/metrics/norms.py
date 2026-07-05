from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict

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
