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


def _load_validation_curves(project_root: Path, config_hash: str, benchmark: str) -> dict[str, list[dict[str, Any]]]:
    curves: dict[str, list[dict[str, Any]]] = {}
    runs_root = Path(project_root) / "runs" / benchmark
    for path in runs_root.glob(f"*/*/{config_hash}/validation_metrics.jsonl"):
        variant = path.parts[-4]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        curves.setdefault(variant, []).append(rows)
    return curves


def _load_routing_summaries(project_root: Path, config_hash: str, benchmark: str) -> dict[str, list[dict[str, Any]]]:
    summaries: dict[str, list[dict[str, Any]]] = {}
    runs_root = Path(project_root) / "runs" / benchmark
    for path in runs_root.glob(f"*/*/{config_hash}/routing_summary.json"):
        variant = path.parts[-4]
        summaries.setdefault(variant, []).append(json.loads(path.read_text(encoding="utf-8")))
    return summaries


def _grouped_bar(rows: list[dict[str, Any]], variants: list[str], keys: list[str], labels: list[str], title: str, plots_dir: Path, name: str) -> None:
    if not variants:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(variants))
    width = 0.8 / max(1, len(keys))
    for index, (key, label) in enumerate(zip(keys, labels)):
        means = []
        for variant in variants:
            values = [float(row[key]) for row in rows if row["variant"] == variant and row.get(key) is not None]
            means.append(float(np.mean(values)) if values else np.nan)
        ax.bar(x + index * width, means, width=width, label=label)
    ax.set_xticks(x + width * (len(keys) - 1) / 2)
    ax.set_xticklabels(variants, rotation=30)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save_figure(fig, plots_dir, name)


def generate_plots(project_root: Path, config_hash: str, *, benchmark: str) -> Path:
    plots_dir = ensure_dir(Path(project_root) / "plots" / benchmark / config_hash)
    rows = collect_run_rows(project_root, config_hash, benchmark=benchmark)
    variants = sorted({str(row["variant"]) for row in rows})

    # 1. Overall accuracy
    v, means, stds = _mean_std_by_variant(rows, "test_accuracy")
    if v:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(v, means, yerr=stds, capsize=4)
        ax.set_title(f"{benchmark} overall test accuracy (official subset)")
        ax.tick_params(axis="x", rotation=30)
        _save_figure(fig, plots_dir, "01_overall_accuracy_by_variant")

    # 2. By reasoning category
    cat_keys = [
        "category_attribute_query_accuracy",
        "category_counting_accuracy",
        "category_existence_accuracy",
        "category_integer_comparison_accuracy",
        "category_attribute_comparison_accuracy",
    ]
    _grouped_bar(
        rows,
        variants,
        cat_keys,
        ["attr_query", "count", "exist", "int_cmp", "attr_cmp"],
        f"{benchmark} accuracy by official reasoning category",
        plots_dir,
        "02_accuracy_by_reasoning_category",
    )

    # 3. By program length
    len_keys = [
        "program_length_1-5_accuracy",
        "program_length_6-10_accuracy",
        "program_length_11-15_accuracy",
        "program_length_16+_accuracy",
    ]
    _grouped_bar(
        rows,
        variants,
        len_keys,
        ["1-5", "6-10", "11-15", "16+"],
        f"{benchmark} accuracy by functional-program length",
        plots_dir,
        "03_accuracy_by_program_length",
    )

    # 4. By dependency depth
    depth_keys = [
        "dependency_depth_1-3_accuracy",
        "dependency_depth_4-6_accuracy",
        "dependency_depth_7-9_accuracy",
        "dependency_depth_10+_accuracy",
    ]
    _grouped_bar(
        rows,
        variants,
        depth_keys,
        ["1-3", "4-6", "7-9", "10+"],
        f"{benchmark} accuracy by dependency-chain depth",
        plots_dir,
        "04_accuracy_by_dependency_depth",
    )

    # 5. Validation curves
    curves = _load_validation_curves(project_root, config_hash, benchmark)
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
    _save_figure(fig, plots_dir, "05_validation_accuracy_curves")

    # 6. Answer-token loss
    v, means, stds = _mean_std_by_variant(rows, "test_answer_token_nll")
    if v:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(v, means, yerr=stds, capsize=4)
        ax.set_title("Answer-token NLL by variant")
        ax.tick_params(axis="x", rotation=30)
        _save_figure(fig, plots_dir, "06_answer_token_loss_by_variant")

    # 7. Resources
    for name, key, title in (
        ("07a_parameter_count", "parameter_count", "Parameter count"),
        ("07b_peak_gpu_memory", "peak_gpu_memory_bytes", "Peak GPU memory"),
        ("07c_throughput", "throughput_examples_per_sec", "Throughput"),
    ):
        v, means, stds = _mean_std_by_variant(rows, key)
        if not v:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(v, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        _save_figure(fig, plots_dir, name)

    # 8/9 routing by program depth
    routing = _load_routing_summaries(project_root, config_hash, benchmark)
    for namespace, plot_name, title in (
        ("encoder", "08_encoder_routing_by_program_depth", "Encoder routing by program depth"),
        ("decoder", "09_decoder_routing_by_program_depth", "Decoder routing by program depth"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for variant, summaries in routing.items():
            level_to_values: dict[str, list[float]] = {}
            for summary in summaries:
                by_depth = (summary.get("by_program_depth_test") or summary.get("by_program_depth") or {}).get(namespace, {})
                for label, stats in by_depth.items():
                    level_to_values.setdefault(label, []).append(float(stats.get("entropy", np.nan)))
            if not level_to_values:
                continue
            labels = sorted(level_to_values)
            ys = [float(np.nanmean(level_to_values[label])) for label in labels]
            ax.plot(range(len(labels)), ys, marker="o", label=variant)
            plotted = True
        if plotted:
            ax.set_title(title)
            ax.set_xlabel("Program-depth group")
            ax.set_ylabel("Routing entropy")
            ax.legend(fontsize=8)
            _save_figure(fig, plots_dir, plot_name)
        else:
            plt.close(fig)

    # 10/11/12 CoGenT-specific
    if benchmark == "clevr_cogent_v1":
        v, means_a, stds_a = _mean_std_by_variant(rows, "condition_A_validation_accuracy")
        _, means_b, stds_b = _mean_std_by_variant(rows, "condition_B_test_accuracy")
        if v:
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(v))
            ax.bar(x - 0.2, means_a, width=0.4, yerr=stds_a, capsize=3, label="Condition A val")
            ax.bar(x + 0.2, means_b, width=0.4, yerr=stds_b, capsize=3, label="Condition B test")
            ax.set_xticks(x)
            ax.set_xticklabels(v, rotation=30)
            ax.set_title("CoGenT Condition A vs B accuracy")
            ax.legend()
            _save_figure(fig, plots_dir, "10_cogent_A_vs_B")

        v, means, stds = _mean_std_by_variant(rows, "a_to_b_accuracy_drop")
        if v:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(v, means, yerr=stds, capsize=4)
            ax.set_title("CoGenT A-to-B generalization drop")
            ax.tick_params(axis="x", rotation=30)
            _save_figure(fig, plots_dir, "11_cogent_generalization_drop")

        shape_keys = ["shape_cube_accuracy", "shape_cylinder_accuracy", "shape_sphere_accuracy"]
        _grouped_bar(
            rows,
            variants,
            shape_keys,
            ["cube", "cylinder", "sphere"],
            "CoGenT generalization by mentioned object shape",
            plots_dir,
            "12_cogent_by_shape",
        )

    return plots_dir
