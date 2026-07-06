#!/usr/bin/env python3
"""Plot benchmark sweep accuracy vs training tokens for HellaSwag, LAMBADA, ARC-Easy.

Example (from repo root):

    python scripts/plot_benchmark_sweep.py

    python scripts/plot_benchmark_sweep.py ^
      --input outputs/benchmark_sweeps/fineweb_edu_medium_baseline_ctx1024_steps15259_seed1337/benchmark_sweep_latest.json ^
      --show
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keep in sync with scripts/eval_benchmark_sweep.py TASK_SPECS.
TASK_SPECS: dict[str, dict[str, Any]] = {
    "hellaswag": {
        "label": "HellaSwag",
        "metric": "acc_norm,none",
        "stderr": "acc_norm_stderr,none",
        "chance": 0.25,
    },
    "lambada_openai": {
        "label": "LAMBADA",
        "metric": "acc,none",
        "stderr": "acc_stderr,none",
        "chance": 0.0,
    },
    "arc_easy": {
        "label": "ARC-Easy",
        "metric": "acc_norm,none",
        "stderr": "acc_norm_stderr,none",
        "chance": 0.25,
    },
}


def _resolve_input(path: Path | None) -> Path:
    if path is not None:
        candidate = path.expanduser().resolve()
        if candidate.is_dir():
            candidate = candidate / "benchmark_sweep_latest.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Sweep JSON not found: {candidate}")
        return candidate

    sweeps_root = REPO_ROOT / "outputs" / "benchmark_sweeps"
    if not sweeps_root.exists():
        raise FileNotFoundError(
            "No --input given and outputs/benchmark_sweeps does not exist."
        )
    candidates = sorted(
        sweeps_root.glob("*/benchmark_sweep_latest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No benchmark_sweep_latest.json found under outputs/benchmark_sweeps."
        )
    return candidates[0]


def _load_series(payload: dict[str, Any], task: str) -> tuple[list[float], list[float], list[float]]:
    spec = TASK_SPECS[task]
    metric = spec["metric"]
    stderr_key = spec["stderr"]

    checkpoints = sorted(payload.get("checkpoints", []), key=lambda row: row["step"])
    tokens_m: list[float] = []
    scores: list[float] = []
    stderrs: list[float] = []

    for row in checkpoints:
        task_results = row.get("results", {}).get(task, {})
        score = task_results.get(metric)
        if score is None:
            continue
        tokens_seen = row.get("tokens_seen")
        if tokens_seen is None:
            continue
        tokens_m.append(tokens_seen / 1_000_000.0)
        scores.append(float(score))
        stderrs.append(float(task_results.get(stderr_key, 0.0) or 0.0))

    return tokens_m, scores, stderrs


def plot_benchmark_sweep(
    payload: dict[str, Any],
    *,
    title: str | None = None,
) -> plt.Figure:
    tasks = [task for task in TASK_SPECS if task in payload.get("tasks", TASK_SPECS)]
    if not tasks:
        tasks = list(TASK_SPECS)

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 4.5), constrained_layout=True)
    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks, strict=True):
        spec = TASK_SPECS[task]
        tokens_m, scores, stderrs = _load_series(payload, task)

        ax.errorbar(
            tokens_m,
            scores,
            yerr=stderrs,
            marker="o",
            capsize=3,
            linewidth=1.8,
            markersize=6,
            label=spec["label"],
        )
        chance = float(spec["chance"])
        if chance > 0.0:
            ax.axhline(chance, color="0.55", linestyle="--", linewidth=1.2, label=f"chance ({chance:.0%})")

        ax.set_title(spec["label"])
        ax.set_xlabel("Tokens seen (M)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(bottom=0.0, top=min(1.0, max(scores + [chance, 0.05]) * 1.15))
        ax.grid(True, alpha=0.3)
        if chance > 0.0:
            ax.legend(loc="lower right", fontsize=9)

    run_name = payload.get("run_name") or Path(payload.get("checkpoint_dir", "sweep")).name
    fig.suptitle(title or f"Benchmark sweep — {run_name}", fontsize=12)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to benchmark_sweep_latest.json or its parent directory "
        "(default: newest sweep under outputs/benchmark_sweeps).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path (default: <sweep_dir>/benchmark_sweep_plot.png).",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive window.")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    input_path = _resolve_input(args.input)
    with input_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    fig = plot_benchmark_sweep(payload)
    output_path = args.output
    if output_path is None:
        output_path = input_path.parent / "benchmark_sweep_plot.png"
    else:
        output_path = output_path.expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
