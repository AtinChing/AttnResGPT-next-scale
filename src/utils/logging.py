from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.config import Config, config_to_dict, save_config
from src.utils.runtime import ensure_dir, sanitize_name


@dataclass
class RunIdentity:
    run_name: str
    config_hash: str


@dataclass
class RunPaths:
    run_dir: Path
    probe_dir: Path
    checkpoint_dir: Path
    train_log_path: Path
    val_log_path: Path
    global_train_log_path: Path
    global_val_log_path: Path
    summary_json_path: Path
    summary_csv_path: Path
    metadata_path: Path
    config_snapshot_path: Path
    config_hash_path: Path
    tokenizer_dir: Path


def canonical_config_dict(config: Config) -> dict[str, Any]:
    payload = config_to_dict(config)
    payload["experiment"].pop("name", None)
    payload["experiment"].pop("stage", None)
    payload["experiment"].pop("notes", None)
    payload["training"].pop("resume_from", None)
    payload["training"].pop("allow_resume_mismatch", None)
    payload["logging"].pop("output_root", None)
    return payload


def config_hash(config: Config) -> str:
    serialized = json.dumps(canonical_config_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def build_run_name(config: Config) -> str:
    dataset = sanitize_name(config.data.dataset_name)
    size = sanitize_name(config.model.size_name)
    model = sanitize_name(config.model.architecture)
    context = config.data.block_size
    steps = config.training.max_steps
    seed = config.experiment.seed
    return f"{dataset}_{size}_{model}_ctx{context}_steps{steps}_seed{seed}"


def build_run_identity(config: Config) -> RunIdentity:
    return RunIdentity(run_name=build_run_name(config), config_hash=config_hash(config))


def create_run_paths(output_root: str | Path, identity: RunIdentity) -> RunPaths:
    output_root = Path(output_root)
    logs_dir = ensure_dir(output_root / "logs")
    run_dir = ensure_dir(output_root / "runs" / identity.run_name)
    probe_dir = ensure_dir(run_dir / "probes")
    checkpoint_dir = ensure_dir(output_root / "checkpoints" / identity.run_name)
    tokenizer_dir = ensure_dir(run_dir / "tokenizer")
    return RunPaths(
        run_dir=run_dir,
        probe_dir=probe_dir,
        checkpoint_dir=checkpoint_dir,
        train_log_path=run_dir / "train_metrics.jsonl",
        val_log_path=run_dir / "val_metrics.jsonl",
        global_train_log_path=logs_dir / f"{identity.run_name}_train.jsonl",
        global_val_log_path=logs_dir / f"{identity.run_name}_val.jsonl",
        summary_json_path=run_dir / "run_summary.json",
        summary_csv_path=run_dir / "run_summary.csv",
        metadata_path=run_dir / "run_metadata.json",
        config_snapshot_path=run_dir / "config.snapshot.yaml",
        config_hash_path=run_dir / "config.hash.txt",
        tokenizer_dir=tokenizer_dir,
    )


def write_run_snapshot(config: Config, identity: RunIdentity, paths: RunPaths, metadata: Mapping[str, Any]) -> None:
    save_config(config, paths.config_snapshot_path)
    paths.config_hash_path.write_text(identity.config_hash, encoding="utf-8")
    with paths.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metadata), handle, indent=2, sort_keys=True)


class ExperimentLogger:
    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def log_train(self, payload: Mapping[str, Any]) -> None:
        self._append_jsonl(self.paths.train_log_path, payload)
        self._append_jsonl(self.paths.global_train_log_path, payload)

    def log_val(self, payload: Mapping[str, Any]) -> None:
        self._append_jsonl(self.paths.val_log_path, payload)
        self._append_jsonl(self.paths.global_val_log_path, payload)

    def save_probe(self, step: int, payload: Mapping[str, Any]) -> Path:
        probe_path = self.paths.probe_dir / f"step_{step:07d}.json"
        with probe_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
        return probe_path

    def save_summary(self, payload: Mapping[str, Any]) -> None:
        row = _flatten_summary_row(dict(payload))
        with self.paths.summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
        write_csv_rows(self.paths.summary_csv_path, [row])

    def prune_old_checkpoints(self, keep_last_k: int) -> None:
        if keep_last_k <= 0:
            return
        checkpoints = sorted(self.paths.checkpoint_dir.glob("step_*.pt"))
        for checkpoint in checkpoints[:-keep_last_k]:
            checkpoint.unlink(missing_ok=True)


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _flatten_summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "run_name",
        "model",
        "size",
        "context",
        "dataset",
        "tokenizer_name",
        "seed",
        "val_loss",
        "perplexity",
        "second_half_loss",
        "mean_activation_norm_last_layer",
        "mean_early_contribution",
        "mean_late_contribution",
        "parameter_count_total",
        "config_hash",
        "checkpoint_path",
    )
    return {key: summary.get(key) for key in keys}


def _consolidated_summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "model",
        "size",
        "context",
        "val_loss",
        "perplexity",
        "second_half_loss",
        "mean_activation_norm_last_layer",
        "mean_early_contribution",
        "mean_late_contribution",
    )
    return {key: summary.get(key) for key in keys}


def _merge_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in existing_rows:
        key = tuple(row.get(field) for field in key_fields)
        merged[key] = dict(row)
    for row in new_rows:
        key = tuple(row.get(field) for field in key_fields)
        merged[key] = dict(row)
    return [
        row
        for _key, row in sorted(
            merged.items(),
            key=lambda item: tuple(str(part) for part in item[0]),
        )
    ]


def write_global_summary_artifacts(
    output_root: str | Path,
    summary_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_root = Path(output_root)
    logs_dir = ensure_dir(output_root / "logs")
    summary_path = logs_dir / "run_summaries.csv"
    consolidated_path = logs_dir / "consolidated_summary_table.csv"
    paired_path = logs_dir / "paired_comparisons.csv"

    merged_summaries = _merge_rows(read_csv_rows(summary_path), summary_rows, key_fields=("run_name",))
    merged_consolidated = _merge_rows(
        read_csv_rows(consolidated_path),
        [_consolidated_summary_row(row) for row in summary_rows],
        key_fields=("model", "size", "context"),
    )
    merged_paired = _merge_rows(read_csv_rows(paired_path), paired_rows, key_fields=("size", "context"))

    write_csv_rows(summary_path, merged_summaries)
    write_csv_rows(consolidated_path, merged_consolidated)
    write_csv_rows(paired_path, merged_paired)
