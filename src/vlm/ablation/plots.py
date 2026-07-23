from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.vlm.ablation.aggregate import collect_run_rows
from src.vlm.ablation.io_utils import ensure_dir


def _save_figure(fig: plt.Figure, plots_dir: Path, name: str) -> None:
    png = plots_dir / f"{name}.png"
    pdf = plots_dir / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=160)
    fig.savefig(pdf)
    plt.close(fig)


def _mean_std_by_variant(rows: list[dict[str, Any]], key: str) -> tuple[list[str], list[float], list[float]]:
    by_variant: dict[str, list[float]] = {}
    for row in rows:
        if row.get(key) is None:
            continue
        by_variant.setdefault(str(row["variant"]), []).append(float(row[key]))
    variants = sorted(by_variant)
    means = [float(np.mean(by_variant[v])) for v in variants]
    stds = [float(np.std(by_variant[v], ddof=0)) for v in variants]
    return variants, means, stds


def _load_validation_curves(project_root: Path, config_hash: str) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {}
    runs_root = Path(project_root) / "runs"
    for path in runs_root.glob(f"*/*/{config_hash}/validation_metrics.jsonl"):
        variant = path.parts[-4]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        curves.setdefault(variant, []).append(rows)
    return curves


def _load_routing_summaries(project_root: Path, config_hash: str) -> dict[str, list[dict[str, Any]]]:
    summaries: dict[str, list[dict[str, Any]]] = {}
    runs_root = Path(project_root) / "runs"
    for path in runs_root.glob(f"*/*/{config_hash}/routing_summary.json"):
        variant = path.parts[-4]
        summaries.setdefault(variant, []).append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def _variant_mean_matrix(
    rows: list[dict[str, Any]],
    variants: list[str],
    keys: list[str],
) -> np.ndarray:
    matrix = np.full((len(variants), len(keys)), np.nan, dtype=float)
    for row_index, variant in enumerate(variants):
        for col_index, key in enumerate(keys):
            values = [
                float(row[key])
                for row in rows
                if row.get("variant") == variant and row.get(key) is not None
            ]
            if values:
                matrix[row_index, col_index] = float(np.mean(values))
    return matrix


def generate_plots(project_root: Path, config_hash: str) -> Path:
    plots_dir = ensure_dir(Path(project_root) / "plots" / config_hash)
    rows = collect_run_rows(project_root, config_hash)
    variants = sorted({str(row["variant"]) for row in rows})

    # 1. Accuracy versus difficulty level
    level_keys = [f"level_{level}_accuracy" for level in range(1, 6)]
    if variants:
        matrix = _variant_mean_matrix(rows, variants, level_keys)
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(1, 6)
        for row_index, variant in enumerate(variants):
            ax.plot(x, matrix[row_index], marker="o", label=variant)
        ax.set_xticks(x)
        ax.set_xlabel("Difficulty level")
        ax.set_ylabel("Test accuracy")
        ax.set_title("Accuracy versus difficulty level")
        ax.legend(fontsize=8)
        _save_figure(fig, plots_dir, "01_accuracy_vs_difficulty")

    # 2. Accuracy versus visual degradation strength
    deg_keys = ["degradation_low_accuracy", "degradation_mid_accuracy", "degradation_high_accuracy"]
    if variants:
        matrix = _variant_mean_matrix(rows, variants, deg_keys)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(deg_keys))
        width = 0.8 / max(1, len(variants))
        for index, variant in enumerate(variants):
            ax.bar(x + index * width, matrix[index], width=width, label=variant)
        ax.set_xticks(x + width * (len(variants) - 1) / 2)
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_ylabel("Test accuracy")
        ax.set_title("Accuracy versus visual degradation strength")
        ax.legend(fontsize=8)
        _save_figure(fig, plots_dir, "02_accuracy_vs_degradation")

    # 3. Accuracy versus reasoning-hop count
    hop_keys = [f"hops_{hop}_accuracy" for hop in (0, 1, 2)]
    if variants:
        matrix = _variant_mean_matrix(rows, variants, hop_keys)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(hop_keys))
        for row_index, variant in enumerate(variants):
            ax.plot(x, matrix[row_index], marker="o", label=variant)
        ax.set_xticks(x)
        ax.set_xticklabels(["0-hop", "1-hop", "2-hop"])
        ax.set_ylabel("Test accuracy")
        ax.set_title("Accuracy versus reasoning-hop count")
        ax.legend(fontsize=8)
        _save_figure(fig, plots_dir, "03_accuracy_vs_hops")

    # 4. Validation accuracy over training
    curves = _load_validation_curves(project_root, config_hash)
    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False
    for variant, variant_curves in curves.items():
        if not variant_curves:
            continue
        max_len = max(len(curve) for curve in variant_curves)
        matrix = np.full((len(variant_curves), max_len), np.nan)
        for row_index, curve in enumerate(variant_curves):
            for col_index, point in enumerate(curve):
                matrix[row_index, col_index] = point["accuracy"]
        mean = np.nanmean(matrix, axis=0)
        ax.plot(np.arange(len(mean)), mean, label=variant)
        plotted = True
    ax.set_title("Validation accuracy over training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    if plotted:
        ax.legend(fontsize=8)
    _save_figure(fig, plots_dir, "04_validation_accuracy_curves")

    # 5. Answer-token loss by variant
    variants, means, stds = _mean_std_by_variant(rows, "test_answer_token_nll")
    if variants:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(variants, means, yerr=stds, capsize=4)
        ax.set_title("Answer-token NLL by variant")
        ax.set_ylabel("NLL")
        ax.tick_params(axis="x", rotation=30)
        _save_figure(fig, plots_dir, "05_answer_token_loss_by_variant")

    # 6/7 Encoder/decoder routing by difficulty
    routing = _load_routing_summaries(project_root, config_hash)
    for namespace, plot_name, title in (
        ("encoder", "06_encoder_routing_by_difficulty", "Encoder routing by difficulty"),
        ("decoder", "07_decoder_routing_by_difficulty", "Decoder routing by difficulty"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for variant, summaries in routing.items():
            if not summaries:
                continue
            # Average entropy over seeds/sites via by_difficulty summary when present.
            level_to_values: dict[str, list[float]] = {}
            for summary in summaries:
                by_diff = (summary.get("by_difficulty_test") or summary.get("by_difficulty") or {}).get(
                    namespace, {}
                )
                if not by_diff:
                    # Fallback: mean over sites' by_difficulty.
                    sites = summary.get(f"{namespace}_routing") or summary.get(
                        f"test_{namespace}_routing", []
                    )
                    for site in sites:
                        for level, stats in site.get("by_difficulty", {}).items():
                            level_to_values.setdefault(level, []).append(float(stats.get("entropy", np.nan)))
                else:
                    for level, stats in by_diff.items():
                        level_to_values.setdefault(level, []).append(float(stats.get("entropy", np.nan)))
            if not level_to_values:
                continue
            levels = sorted(level_to_values, key=lambda item: int(item.split("_")[-1]) if "_" in item else item)
            ys = [float(np.nanmean(level_to_values[level])) for level in levels]
            ax.plot(range(len(levels)), ys, marker="o", label=variant)
            plotted = True
        if plotted:
            ax.set_title(title)
            ax.set_xlabel("Difficulty group")
            ax.set_ylabel("Routing entropy")
            ax.legend(fontsize=8)
            _save_figure(fig, plots_dir, plot_name)
        else:
            plt.close(fig)

    # Keep a few diagnostic family/resource plots under higher numbers.
    family_keys = [
        "local_detail_accuracy",
        "attribute_accuracy",
        "counting_accuracy",
        "location_accuracy",
        "relation_accuracy",
        "compositional_accuracy",
        "multi_hop_accuracy",
    ]
    if variants:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(variants))
        width = 0.11
        for index, family in enumerate(family_keys):
            family_means = []
            for variant in variants:
                values = [
                    float(row[family])
                    for row in rows
                    if row["variant"] == variant and row.get(family) is not None
                ]
                family_means.append(float(np.mean(values)) if values else 0.0)
            ax.bar(x + index * width, family_means, width=width, label=family.replace("_accuracy", ""))
        ax.set_xticks(x + width * 3)
        ax.set_xticklabels(variants, rotation=30)
        ax.set_title("Accuracy by question family and variant")
        ax.legend(fontsize=7, ncol=2)
        _save_figure(fig, plots_dir, "08_accuracy_by_family")

    for name, key, title in (
        ("09_held_out_accuracy", "held_out_accuracy", "Held-out combination accuracy"),
        ("10_parameter_count", "parameter_count", "Parameter count by variant"),
        ("11_peak_gpu_memory", "peak_gpu_memory_bytes", "Peak GPU memory by variant"),
    ):
        variants, means, stds = _mean_std_by_variant(rows, key)
        if not variants:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(variants, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        _save_figure(fig, plots_dir, name)

    return plots_dir
