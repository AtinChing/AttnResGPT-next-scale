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


def _load_routing_summary(project_root: Path, config_hash: str, variant: str) -> dict[str, Any] | None:
    runs_root = Path(project_root) / "runs" / variant
    for path in runs_root.glob(f"*/{config_hash}/routing_summary.json"):
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def generate_plots(project_root: Path, config_hash: str) -> Path:
    plots_dir = ensure_dir(Path(project_root) / "plots")
    rows = collect_run_rows(project_root, config_hash)

    # 1. Overall test accuracy
    variants, means, stds = _mean_std_by_variant(rows, "test_accuracy")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(variants, means, yerr=stds, capsize=4)
    ax.set_title("Overall test accuracy by variant (exploratory)")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=30)
    _save_figure(fig, plots_dir, "01_test_accuracy_by_variant")

    # 2. Accuracy by family
    families = [
        "local_detail_accuracy",
        "attribute_accuracy",
        "counting_accuracy",
        "location_accuracy",
        "relation_accuracy",
    ]
    if variants:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(variants))
        width = 0.15
        for index, family in enumerate(families):
            family_means = []
            for variant in variants:
                values = [float(row[family]) for row in rows if row["variant"] == variant and row.get(family) is not None]
                family_means.append(float(np.mean(values)) if values else 0.0)
            ax.bar(x + index * width, family_means, width=width, label=family.replace("_accuracy", ""))
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(variants, rotation=30)
        ax.set_title("Accuracy by question family and variant")
        ax.legend(fontsize=8)
        _save_figure(fig, plots_dir, "02_accuracy_by_family")

    # 3/4 validation curves
    curves = _load_validation_curves(project_root, config_hash)
    fig, ax = plt.subplots(figsize=(8, 4))
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
    ax.set_title("Validation accuracy over training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    _save_figure(fig, plots_dir, "03_validation_accuracy_curves")

    fig, ax = plt.subplots(figsize=(8, 4))
    for variant, variant_curves in curves.items():
        if not variant_curves:
            continue
        max_len = max(len(curve) for curve in variant_curves)
        matrix = np.full((len(variant_curves), max_len), np.nan)
        for row_index, curve in enumerate(variant_curves):
            for col_index, point in enumerate(curve):
                matrix[row_index, col_index] = point["loss"]
        mean = np.nanmean(matrix, axis=0)
        ax.plot(np.arange(len(mean)), mean, label=variant)
    ax.set_title("Validation loss over training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    _save_figure(fig, plots_dir, "04_validation_loss_curves")

    # 5/6/7 resource plots
    for index, (key, title, name) in enumerate(
        [
            ("parameter_count", "Parameter count by variant", "05_parameter_count"),
            ("peak_gpu_memory_bytes", "Peak GPU memory by variant", "06_peak_gpu_memory"),
            ("throughput_examples_per_sec", "Throughput by variant", "07_throughput"),
        ],
        start=5,
    ):
        variants, means, stds = _mean_std_by_variant(rows, key)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(variants, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        _save_figure(fig, plots_dir, name)

    # Routing heatmaps / entropy: use first available seed summary per interesting variant.
    for variant in ("encoder_full", "both_full", "decoder_full", "encoder_block", "both_block"):
        summary = _load_routing_summary(project_root, config_hash, variant)
        if not summary:
            continue
        for namespace, plot_prefix in (
            ("encoder_routing", "08_encoder"),
            ("decoder_routing", "09_decoder"),
        ):
            sites = summary.get(namespace, [])
            if not sites:
                continue
            families = sorted({family for site in sites for family in site.get("by_family", {})})
            if families:
                matrix = np.array(
                    [
                        [site.get("by_family", {}).get(family, {}).get("embedding", np.nan) for family in families]
                        for site in sites
                    ],
                    dtype=float,
                )
                fig, ax = plt.subplots(figsize=(8, 4))
                im = ax.imshow(matrix, aspect="auto", cmap="viridis")
                ax.set_yticks(range(len(sites)))
                ax.set_yticklabels([f"site_{site['site_index']}" for site in sites])
                ax.set_xticks(range(len(families)))
                ax.set_xticklabels(families, rotation=30)
                ax.set_title(f"{namespace} embedding contribution by family ({variant})")
                fig.colorbar(im, ax=ax, fraction=0.046)
                _save_figure(fig, plots_dir, f"{plot_prefix}_routing_heatmap_{variant}")

            entropies = [float(site.get("entropy", np.nan)) for site in sites]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(range(len(entropies)), entropies, marker="o")
            ax.set_title(f"{namespace} entropy by site ({variant})")
            ax.set_xlabel("Routing site")
            ax.set_ylabel("Entropy")
            prefix = "10_encoder" if namespace.startswith("encoder") else "11_decoder"
            _save_figure(fig, plots_dir, f"{prefix}_routing_entropy_{variant}")

    return plots_dir
