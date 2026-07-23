from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.vlm.ablation.io_utils import atomic_write_json, ensure_dir


FAMILY_KEYS = {
    "local_detail": "local_detail_accuracy",
    "attribute": "attribute_accuracy",
    "counting": "counting_accuracy",
    "location": "location_accuracy",
    "relation": "relation_accuracy",
}


def _flatten_run_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    family = metrics.get("family_accuracy_test", {})
    row = {
        "variant": metrics.get("variant"),
        "seed": metrics.get("seed"),
        "encoder_residual": metrics.get("encoder_residual"),
        "decoder_residual": metrics.get("decoder_residual"),
        "validation_accuracy": metrics.get("validation_accuracy"),
        "test_accuracy": metrics.get("test_accuracy"),
        "parameter_count": metrics.get("parameter_count"),
        "parameter_increase_pct": metrics.get("parameter_increase_pct"),
        "peak_gpu_memory_bytes": metrics.get("peak_allocated_bytes"),
        "throughput_examples_per_sec": metrics.get("examples_per_sec"),
        "training_duration_sec": metrics.get("training_duration_sec"),
        "best_epoch": metrics.get("best_epoch"),
        "config_hash": metrics.get("config_hash"),
    }
    for family_name, column in FAMILY_KEYS.items():
        row[column] = family.get(family_name)
    return row


def collect_run_rows(project_root: Path, config_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs_root = Path(project_root) / "runs"
    if not runs_root.exists():
        return rows
    for metrics_path in runs_root.glob(f"*/*/{config_hash}/final_test_metrics.json"):
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

    numeric_keys = [
        "validation_accuracy",
        "test_accuracy",
        "local_detail_accuracy",
        "attribute_accuracy",
        "counting_accuracy",
        "location_accuracy",
        "relation_accuracy",
        "parameter_count",
        "parameter_increase_pct",
        "peak_gpu_memory_bytes",
        "throughput_examples_per_sec",
        "training_duration_sec",
        "best_epoch",
    ]
    aggregated: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(by_variant.items()):
        payload: dict[str, Any] = {
            "variant": variant,
            "n_seeds": len(variant_rows),
            "encoder_residual": variant_rows[0].get("encoder_residual"),
            "decoder_residual": variant_rows[0].get("decoder_residual"),
            "label": "exploratory_mean_std_over_seeds",
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


def write_tables(project_root: Path, config_hash: str) -> dict[str, Path]:
    rows = collect_run_rows(project_root, config_hash)
    tables_dir = ensure_dir(Path(project_root) / "tables")
    all_path = tables_dir / "all_runs.csv"
    agg_path = tables_dir / "aggregate_results.csv"
    write_csv(all_path, rows)
    aggregated = aggregate_rows(rows)
    # Flatten list values for CSV readability.
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
