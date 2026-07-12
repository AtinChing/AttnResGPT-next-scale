"""Shared lm-eval task specs for benchmark sweeps and the blessed panel."""
from __future__ import annotations

from typing import Any

PANEL_TASKS = (
    "hellaswag",
    "lambada_openai",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "boolq",
    "sciq",
)

# Primary metric + random-guess baseline.
TASK_SPECS: dict[str, dict[str, Any]] = {
    "hellaswag": {
        "metric": "acc_norm,none",
        "chance": 0.25,
        "choices": 4,
        "label": "HellaSwag",
    },
    "lambada_openai": {
        "metric": "acc,none",
        "chance": 0.0,
        "choices": None,
        "label": "LAMBADA",
    },
    "piqa": {
        "metric": "acc_norm,none",
        "chance": 0.5,
        "choices": 2,
        "label": "PIQA",
    },
    "winogrande": {
        "metric": "acc,none",
        "chance": 0.5,
        "choices": 2,
        "label": "WinoGrande",
    },
    "arc_easy": {
        "metric": "acc_norm,none",
        "chance": 0.25,
        "choices": 4,
        "label": "ARC-Easy",
    },
    "arc_challenge": {
        "metric": "acc_norm,none",
        "chance": 0.25,
        "choices": 4,
        "label": "ARC-Challenge",
    },
    "openbookqa": {
        "metric": "acc_norm,none",
        "chance": 0.25,
        "choices": 4,
        "label": "OpenBookQA",
    },
    "boolq": {
        "metric": "acc,none",
        "chance": 0.5,
        "choices": 2,
        "label": "BoolQ",
    },
    "sciq": {
        "metric": "acc,none",
        "chance": 0.25,
        "choices": 4,
        "label": "SciQ",
    },
}


def metric_value(task_results: dict[str, Any], task_name: str) -> float | None:
    spec = TASK_SPECS[task_name]
    metric_key = spec["metric"]
    value = task_results.get(metric_key)
    if value is None:
        for key, candidate in task_results.items():
            if key.startswith(metric_key.split(",")[0]):
                value = candidate
                break
    if value is None:
        return None
    return float(value)
