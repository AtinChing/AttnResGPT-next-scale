from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.vlm.ablation.io_utils import atomic_write_json, ensure_dir


def _acc(payload: Any) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict) and "accuracy" in payload:
        return float(payload["accuracy"])
    return None


def _flatten_run_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    categories = metrics.get("category_accuracy_test") or {}
    lengths = metrics.get("program_length_accuracy_test") or {}
    depths = metrics.get("dependency_depth_accuracy_test") or {}
    shapes = metrics.get("shape_accuracy_test") or {}
    row = {
        "benchmark": metrics.get("benchmark"),
        "variant": metrics.get("variant"),
        "seed": metrics.get("seed"),
        "encoder_residual": metrics.get("encoder_residual"),
        "decoder_residual": metrics.get("decoder_residual"),
        "validation_accuracy": metrics.get("validation_accuracy"),
        "validation_loss": metrics.get("validation_loss"),
        "validation_answer_token_nll": metrics.get("validation_answer_token_nll"),
        "test_accuracy": metrics.get("test_accuracy"),
        "test_loss": metrics.get("test_loss"),
        "test_answer_token_nll": metrics.get("test_answer_token_nll"),
        "condition_A_validation_accuracy": metrics.get("condition_A_validation_accuracy"),
        "condition_B_test_accuracy": metrics.get("condition_B_test_accuracy"),
        "a_to_b_accuracy_drop": metrics.get("a_to_b_accuracy_drop"),
        "parameter_count": metrics.get("parameter_count"),
        "parameter_increase_pct": metrics.get("parameter_increase_pct"),
        "peak_gpu_memory_bytes": metrics.get("peak_allocated_bytes"),
        "throughput_examples_per_sec": metrics.get("examples_per_sec"),
        "training_duration_sec": metrics.get("training_duration_sec"),
        "best_epoch": metrics.get("best_epoch"),
        "config_hash": metrics.get("config_hash"),
        "subset_manifest_hash": metrics.get("subset_manifest_hash"),
    }
    for category in (
        "attribute_query",
        "counting",
        "existence",
        "integer_comparison",
        "attribute_comparison",
    ):
        row[f"category_{category}_accuracy"] = _acc(categories.get(category))
    for length_bin in ("1-5", "6-10", "11-15", "16+"):
        row[f"program_length_{length_bin}_accuracy"] = _acc(lengths.get(length_bin))
    for depth_bin in ("1-3", "4-6", "7-9", "10+"):
        row[f"dependency_depth_{depth_bin}_accuracy"] = _acc(depths.get(depth_bin))
    for shape in ("cube", "cylinder", "sphere"):
        row[f"shape_{shape}_accuracy"] = _acc(shapes.get(shape))
    return row


def collect_run_rows(project_root: Path, config_hash: str, *, benchmark: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs_root = Path(project_root) / "runs"
    if not runs_root.exists():
        return rows
    pattern = f"*/*/{config_hash}/final_test_metrics.json"
    if benchmark:
        pattern = f"{benchmark}/*/*/{config_hash}/final_test_metrics.json"
    for metrics_path in runs_root.glob(pattern):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(_flatten_run_metrics(metrics))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    numeric_keys = [key for key in (rows[0].keys() if rows else []) if key not in {"benchmark", "variant", "seed", "encoder_residual", "decoder_residual", "config_hash", "subset_manifest_hash"}]
    aggregated: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(by_variant.items()):
        payload: dict[str, Any] = {
            "variant": variant,
            "n_seeds": len(variant_rows),
            "encoder_residual": variant_rows[0].get("encoder_residual"),
            "decoder_residual": variant_rows[0].get("decoder_residual"),
            "benchmark": variant_rows[0].get("benchmark"),
            "label": "exploratory_mean_std_over_seeds_official_subset",
        }
        for key in numeric_keys:
            values = [float(row[key]) for row in variant_rows if row.get(key) is not None]
            if not values:
                payload[f"{key}_mean"] = None
                payload[f"{key}_std"] = None
                payload[f"{key}_values"] = []
                continue
            payload[f"{key}_mean"] = float(np.mean(values))
            payload[f"{key}_std"] = float(np.std(values, ddof=0))
            payload[f"{key}_values"] = values
        aggregated.append(payload)
    return aggregated


def write_tables(project_root: Path, config_hash: str, *, benchmark: str) -> dict[str, Path]:
    rows = collect_run_rows(project_root, config_hash, benchmark=benchmark)
    tables_dir = ensure_dir(Path(project_root) / "tables" / benchmark / config_hash)
    all_path = tables_dir / "all_runs.csv"
    agg_path = tables_dir / "aggregate_results.csv"
    write_csv(all_path, rows)
    aggregated = aggregate_rows(rows)
    flat_agg: list[dict[str, Any]] = []
    for row in aggregated:
        flat = dict(row)
        for key, value in list(flat.items()):
            if isinstance(value, list):
                flat[key] = ";".join(str(item) for item in value)
        flat_agg.append(flat)
    write_csv(agg_path, flat_agg)
    atomic_write_json(tables_dir / "aggregate_results.json", aggregated)
    return {"all_runs": all_path, "aggregate_results": agg_path}
