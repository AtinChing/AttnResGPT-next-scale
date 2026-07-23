from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.vlm.ablation.aggregate import write_tables
from src.vlm.ablation.config import (
    AblationExperimentConfig,
    config_hash,
    resolve_experiment_config,
)
from src.vlm.ablation.correctness import run_correctness_checks
from src.vlm.ablation.io_utils import atomic_write_json, create_project_layout, ensure_dir
from src.vlm.ablation.manifest import ExperimentManifest
from src.vlm.ablation.plots import generate_plots
from src.vlm.ablation.source_hash import combined_source_hash, hash_source_tree
from src.vlm.ablation.train import train_variant_seed
from src.vlm.ablation.wandb_logger import resolve_wandb_mode
from src.vlm.clevr.prepare import prepare_benchmark


def _log_experiment_artifacts_to_wandb(
    config: AblationExperimentConfig,
    *,
    project_root: Path,
    cfg_hash: str,
    summary: dict[str, Any],
    plots_dir: Path,
) -> dict[str, Any] | None:
    enabled, mode = resolve_wandb_mode(config)
    if not enabled:
        return None
    try:
        import wandb
    except Exception as error:  # noqa: BLE001
        return {"enabled": False, "error": f"wandb import failed: {error}"}

    init_kwargs = {
        "project": config.wandb_project,
        "entity": config.wandb_entity or None,
        "name": f"vlm_ablation_summary_{config.benchmark}_{config.benchmark_mode}_{cfg_hash}",
        "id": f"vlm_ablation_summary_{config.benchmark}_{cfg_hash}",
        "resume": "allow",
        "mode": mode,
        "dir": str(project_root / "logs"),
        "job_type": "summary",
        "group": f"vlm_ablation_{config.benchmark}_{config.benchmark_mode}_{cfg_hash}",
        "tags": ["vlm-ablation", "summary", config.benchmark, config.benchmark_mode],
        "config": config.to_dict(),
    }
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    try:
        run = wandb.init(**init_kwargs)
    except Exception as error:  # noqa: BLE001
        return {"enabled": False, "error": str(error)}

    for key, value in summary.items():
        if isinstance(value, (int, float, bool, str)):
            run.summary[key] = value
    for table_name, table_path in (summary.get("tables") or {}).items():
        path = Path(table_path)
        if path.exists():
            artifact = wandb.Artifact(f"{cfg_hash}_{table_name}", type="table")
            artifact.add_file(str(path))
            run.log_artifact(artifact)
    image_payload = {}
    for png in sorted(Path(plots_dir).glob("*.png")):
        image_payload[f"plots/{png.stem}"] = wandb.Image(str(png))
    if image_payload:
        run.log(image_payload)
    url = None
    try:
        url = run.get_url() if hasattr(run, "get_url") else None
    except Exception:  # noqa: BLE001
        url = None
    run.finish()
    return {"enabled": True, "mode": mode, "url": url, "project": config.wandb_project}


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is unavailable. Select a Colab GPU runtime and rerun. "
            "Training will not fall back to CPU or MPS."
        )
    return torch.device("cuda")


def print_cuda_environment(amp_dtype: torch.dtype | None) -> dict[str, Any]:
    device = require_cuda()
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    info = {
        "gpu_name": props.name,
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "total_memory_bytes": int(total_bytes),
        "available_memory_bytes": int(free_bytes),
        "amp_dtype": str(amp_dtype) if amp_dtype is not None else None,
    }
    print(json.dumps(info, indent=2))
    return info


def _run_one_benchmark(
    config: AblationExperimentConfig,
    *,
    project_root: Path,
    source_code_hash: str,
    device: torch.device,
    skip_correctness: bool,
) -> dict[str, Any]:
    config = resolve_experiment_config(config)
    prepared = prepare_benchmark(
        project_root=project_root,
        benchmark=config.benchmark,
        mode=config.benchmark_mode,
        subset_seed=config.subset_seed,
        preprocess=config.preprocess_config(),
    )
    config.subset_manifest_hash = prepared.subset_manifest_hash()
    config.vocab_hash = prepared.vocab_hash()
    config.preprocess_hash = prepared.preprocess.config_hash()
    config.dataset_version = prepared.dataset_version
    config = resolve_experiment_config(config)

    cfg_hash = config_hash(config)
    atomic_write_json(project_root / "configs" / f"experiment_{config.benchmark}_{cfg_hash}.json", config.to_dict())
    atomic_write_json(project_root / "summaries" / f"prepared_{config.benchmark}_{cfg_hash}.json", prepared.to_meta())
    print("Resolved experiment configuration:")
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))

    if not skip_correctness:
        report = run_correctness_checks(
            device=device,
            report_path=project_root / "summaries" / f"correctness_checks_{config.benchmark}.json",
        )
        if not report["ok"]:
            raise RuntimeError(f"Correctness checks failed: {report['failed']}")

    manifest = ExperimentManifest(project_root / "manifests" / f"experiment_manifest_{config.benchmark}.json")
    requested = config.requested_variants()
    completed: list[str] = []
    resumed: list[str] = []
    failed: list[str] = []
    results: list[dict[str, Any]] = []

    for variant in requested:
        for seed in config.seeds:
            key = f"{variant}/seed_{seed}"
            try:
                result = train_variant_seed(
                    config,
                    prepared=prepared,
                    variant=variant,
                    seed=seed,
                    project_root=project_root,
                    manifest=manifest,
                    source_code_hash=source_code_hash,
                    device=device,
                )
                results.append(result)
                if result["status"] == "skipped_completed":
                    completed.append(key)
                elif result.get("resumed"):
                    resumed.append(key)
                    completed.append(key)
                else:
                    completed.append(key)
            except Exception as exc:  # noqa: BLE001
                failed.append(key)
                print(f"FAILED {config.benchmark}/{key}: {exc}")
                if config.benchmark_mode == "smoke":
                    raise

    tables = write_tables(project_root, cfg_hash, benchmark=config.benchmark)
    plots_dir = generate_plots(project_root, cfg_hash, benchmark=config.benchmark)

    best_by_variant: dict[str, dict[str, Any]] = {}
    for result in results:
        metrics = result.get("metrics") or {}
        variant = metrics.get("variant")
        if not variant:
            continue
        current = best_by_variant.get(variant)
        if current is None or float(metrics.get("test_accuracy", -1)) > float(current.get("test_accuracy", -1)):
            best_by_variant[variant] = metrics

    summary = {
        "drive_project_root": str(project_root),
        "benchmark": config.benchmark,
        "dataset_version": config.dataset_version,
        "benchmark_mode": config.benchmark_mode,
        "subset_manifest_path": str(prepared.manifest_path),
        "subset_manifest_hash": config.subset_manifest_hash,
        "vocab_hash": config.vocab_hash,
        "preprocess_hash": config.preprocess_hash,
        "subset_sizes": prepared.to_meta()["split_sizes"],
        "config_hash": cfg_hash,
        "source_code_hash": source_code_hash,
        "requested_variants": requested,
        "completed": completed,
        "resumed": resumed,
        "failed": failed,
        "best_by_variant": best_by_variant,
        "tables": {key: str(path) for key, path in tables.items()},
        "plots_dir": str(plots_dir),
        "manifest": str(project_root / "manifests" / f"experiment_manifest_{config.benchmark}.json"),
        "prepared": prepared.to_meta(),
        "artifacts_under_drive": True,
        "label": "compute_constrained_official_subset",
        "wandb_project": config.wandb_project,
        "wandb_entity": config.wandb_entity,
        "test_label": (
            "held_out_official_validation_subset"
            if config.benchmark == "clevr_v1"
            else "condition_B_validation_subset"
        ),
    }
    wandb_summary = _log_experiment_artifacts_to_wandb(
        config,
        project_root=project_root,
        cfg_hash=cfg_hash,
        summary=summary,
        plots_dir=plots_dir,
    )
    if wandb_summary is not None:
        summary["wandb_summary"] = wandb_summary
    ensure_dir(project_root / "summaries")
    atomic_write_json(project_root / "summaries" / f"experiment_summary_{config.benchmark}_{cfg_hash}.json", summary)
    atomic_write_json(project_root / "summaries" / f"experiment_summary_{config.benchmark}.json", summary)
    return summary


def run_ablation_experiment(
    config: AblationExperimentConfig,
    *,
    project_root: Path,
    source_root: Path | None = None,
    skip_correctness: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root)
    create_project_layout(project_root)
    ensure_dir(project_root / "data")
    config.project_root = str(project_root)
    config = resolve_experiment_config(config)

    device = require_cuda()
    major, _ = torch.cuda.get_device_capability(device)
    amp_dtype = torch.bfloat16 if (config.amp_dtype == "auto" and major >= 8) else (
        torch.float16 if config.amp_dtype == "auto" else None
    )
    if config.amp_dtype != "auto":
        from src.utils.runtime import amp_dtype_from_string

        amp_dtype = amp_dtype_from_string(config.amp_dtype)
    cuda_info = print_cuda_environment(amp_dtype)

    source_root = Path(source_root) if source_root is not None else project_root / "source"
    source_hashes = hash_source_tree(source_root)
    source_code_hash = combined_source_hash(source_hashes)
    atomic_write_json(project_root / "summaries" / "source_hashes.json", source_hashes)

    benchmarks: list[str] = []
    if config.run_standard_clevr:
        benchmarks.append("clevr_v1")
    if config.run_cogent:
        benchmarks.append("clevr_cogent_v1")
    if not benchmarks:
        raise ValueError("Enable RUN_STANDARD_CLEVR and/or RUN_COGENT")

    per_benchmark: dict[str, Any] = {}
    failures: list[str] = []
    correctness_done = False
    for benchmark in benchmarks:
        bench_config = resolve_experiment_config(
            AblationExperimentConfig(**{**config.to_dict(), "benchmark": benchmark})
        )
        try:
            per_benchmark[benchmark] = _run_one_benchmark(
                bench_config,
                project_root=project_root,
                source_code_hash=source_code_hash,
                device=device,
                skip_correctness=skip_correctness or correctness_done,
            )
            correctness_done = True
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{benchmark}: {exc}")
            print(f"BENCHMARK FAILED {benchmark}: {exc}")
            if config.benchmark_mode == "smoke":
                raise

    summary = {
        "drive_project_root": str(project_root),
        "cuda": cuda_info,
        "benchmark_mode": config.benchmark_mode,
        "run_standard_clevr": config.run_standard_clevr,
        "run_cogent": config.run_cogent,
        "source_code_hash": source_code_hash,
        "benchmarks": per_benchmark,
        "failures": failures,
        "label": "compute_constrained_official_subset",
        "note": (
            "TinyShapes is no longer the main experiment; official CLEVR/CoGenT subsets are used. "
            "Prior TinyShapes run directories are preserved untouched."
        ),
        "wandb_project": config.wandb_project,
        "wandb_entity": config.wandb_entity,
    }
    atomic_write_json(project_root / "summaries" / "experiment_summary.json", summary)
    return summary
