from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict

import torch
import torch.nn as nn

from src.models.baseline import CausalSelfAttention, FeedForward


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


def _extract_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError("Expected tensor or tuple containing a tensor")


@dataclass
class StepNormTracker:
    activation_norms: dict[str, float] = field(default_factory=dict)
    gradient_norms: dict[str, float] = field(default_factory=dict)
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def register(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, (CausalSelfAttention, FeedForward)):
                self.handles.append(module.register_forward_hook(self._forward_hook(name)))
                self.handles.append(module.register_full_backward_hook(self._backward_hook(name)))

    def _forward_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: object) -> None:
            tensor = _extract_tensor(output)
            value = tensor.detach().float().norm(dim=-1).mean().item()
            self.activation_norms[name] = float(value)

        return hook

    def _backward_hook(self, name: str):
        def hook(
            _module: nn.Module,
            _grad_input: tuple[torch.Tensor | None, ...],
            grad_output: tuple[torch.Tensor | None, ...],
        ) -> None:
            grad = grad_output[0]
            if grad is None:
                return
            value = grad.detach().float().norm(dim=-1).mean().item()
            self.gradient_norms[name] = float(value)

        return hook

    def reset_step(self) -> None:
        self.activation_norms.clear()
        self.gradient_norms.clear()

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            "activation_norms": dict(self.activation_norms),
            "gradient_norms": dict(self.gradient_norms),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@dataclass
class EvalActivationAccumulator:
    sums: DefaultDict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def register(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if isinstance(module, (CausalSelfAttention, FeedForward)):
                self.handles.append(module.register_forward_hook(self._forward_hook(name)))

    def _forward_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: object) -> None:
            tensor = _extract_tensor(output)
            value = tensor.detach().float().norm(dim=-1).mean().item()
            self.sums[name] += float(value)
            self.counts[name] += 1

        return hook

    def finalize(self) -> dict[str, float]:
        return {
            name: self.sums[name] / max(1, self.counts[name])
            for name in sorted(self.sums.keys())
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def mean_last_layer_activation_norm(activation_norms: dict[str, float]) -> float | None:
    if not activation_norms:
        return None
    last_key = sorted(activation_norms.keys())[-1]
    return float(activation_norms[last_key])
