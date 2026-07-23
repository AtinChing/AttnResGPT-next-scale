from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from src.vlm.ablation.aggregate import write_tables
from src.vlm.ablation.config import (
    AblationExperimentConfig,
    apply_difficulty_profile,
    config_hash,
    difficulty_profile_from_config,
    resolve_experiment_config,
)
from src.vlm.ablation.correctness import run_correctness_checks
from src.vlm.ablation.io_utils import atomic_write_json, create_project_layout, ensure_dir
from src.vlm.ablation.manifest import ExperimentManifest
from src.vlm.ablation.plots import generate_plots
from src.vlm.ablation.source_hash import combined_source_hash, hash_source_tree
from src.vlm.ablation.train import train_variant_seed
from src.vlm.ablation.wandb_logger import resolve_wandb_mode


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
        "name": f"vlm_ablation_summary_{config.run_mode}_{cfg_hash}",
        "id": f"vlm_ablation_summary_{cfg_hash}",
        "resume": "allow",
        "mode": mode,
        "dir": str(project_root / "logs"),
        "job_type": "summary",
        "group": f"vlm_ablation_{config.run_mode}_{cfg_hash}",
    }
    init_kwargs["tags"] = ["vlm-ablation", "summary", config.run_mode, config.dataset_version]
    init_kwargs["config"] = config.to_dict()
    init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
    try:
        run = wandb.init(**init_kwargs)
    except Exception as error:  # noqa: BLE001
        return {"enabled": False, "error": str(error)}

    for key, value in summary.items():
        if isinstance(value, (int, float, bool, str)):
            run.summary[key] = value
    tables = summary.get("tables") or {}
    for table_name, table_path in tables.items():
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


def _level45_accuracy(metrics: dict[str, Any]) -> float | None:
    levels = metrics.get("level_accuracy_test") or {}
    values = []
    for key in ("4", "5"):
        if key in levels and levels[key] is not None:
            values.append(float(levels[key]))
    if not values:
        return None
    return float(sum(values) / len(values))


def run_baseline_sanity_gate(
    config: AblationExperimentConfig,
    *,
    project_root: Path,
    source_code_hash: str,
    device: torch.device,
    manifest: ExperimentManifest,
) -> tuple[AblationExperimentConfig, dict[str, Any]]:
    """Run one baseline quick probe; bump difficulty if L4/L5 are still too easy."""
    report: dict[str, Any] = {
        "enabled": True,
        "attempts": [],
        "passed": False,
        "final_bump_level": config.difficulty_bump_level,
    }
    if not config.sanity_check_baseline:
        report["enabled"] = False
        report["passed"] = True
        return config, report

    working = copy.deepcopy(config)
    working = resolve_experiment_config(working)
    seed = int(working.seeds[0]) if working.seeds else 0

    for attempt in range(working.sanity_max_bumps + 1):
        profile = difficulty_profile_from_config(working)
        apply_difficulty_profile(working, profile)
        working = resolve_experiment_config(working)
        cfg_hash = config_hash(working)
        atomic_write_json(
            project_root / "configs" / f"experiment_{cfg_hash}.json",
            working.to_dict(),
        )
        print(
            f"[sanity] attempt={attempt} bump={working.difficulty_bump_level} "
            f"config_hash={cfg_hash}"
        )
        sanity_config = copy.deepcopy(working)
        sanity_config.run_primary_full_grid = True
        sanity_config.run_block_grid = False
        sanity_config.primary_variants = ["baseline"]
        sanity_config.block_variants = []
        sanity_config.seeds = [seed]
        # Force a quick-sized probe even when the outer mode is full.
        if sanity_config.run_mode == "full":
            sanity_config.run_mode = "quick"
            sanity_config = resolve_experiment_config(sanity_config)

        result = train_variant_seed(
            sanity_config,
            variant="baseline",
            seed=seed,
            project_root=project_root,
            manifest=manifest,
            source_code_hash=source_code_hash,
            device=device,
        )
        metrics = result.get("metrics") or {}
        l45 = _level45_accuracy(metrics)
        level_acc = metrics.get("level_accuracy_test") or {}
        attempt_row = {
            "attempt": attempt,
            "config_hash": cfg_hash,
            "bump_level": working.difficulty_bump_level,
            "status": result.get("status"),
            "test_accuracy": metrics.get("test_accuracy"),
            "level_accuracy_test": level_acc,
            "level45_mean_accuracy": l45,
            "too_easy": bool(l45 is not None and l45 > working.sanity_l45_accuracy_ceiling),
            "profile": profile.to_dict(),
        }
        report["attempts"].append(attempt_row)
        print(json.dumps(attempt_row, indent=2, sort_keys=True))

        if l45 is None:
            report["passed"] = False
            report["error"] = "sanity baseline missing level_accuracy_test for levels 4/5"
            break

        if l45 <= working.sanity_l45_accuracy_ceiling:
            report["passed"] = True
            report["final_bump_level"] = working.difficulty_bump_level
            report["final_config_hash"] = cfg_hash
            break

        if attempt >= working.sanity_max_bumps:
            report["passed"] = False
            report["error"] = (
                "baseline still above L4/L5 ceiling after max bumps; "
                "refusing to launch the full ablation grid"
            )
            break

        profile = profile.bumped()
        apply_difficulty_profile(working, profile)
        print(
            f"[sanity] baseline L4/L5 accuracy {l45:.3f} > "
            f"{working.sanity_l45_accuracy_ceiling:.3f}; bumping difficulty to "
            f"level {profile.bump_level}"
        )

    apply_difficulty_profile(config, difficulty_profile_from_config(working))
    config = resolve_experiment_config(config)
    report["final_bump_level"] = config.difficulty_bump_level
    report["final_config_hash"] = config_hash(config)
    atomic_write_json(project_root / "summaries" / "sanity_gate.json", report)
    return config, report


def run_ablation_experiment(
    config: AblationExperimentConfig,
    *,
    project_root: Path,
    source_root: Path | None = None,
    skip_correctness: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root)
    create_project_layout(project_root)
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

    if not skip_correctness:
        report = run_correctness_checks(
            device=device,
            report_path=project_root / "summaries" / "correctness_checks.json",
        )
        if not report["ok"]:
            raise RuntimeError(f"Correctness checks failed: {report['failed']}")

    manifest = ExperimentManifest(project_root / "manifests" / "experiment_manifest.json")
    config, sanity_report = run_baseline_sanity_gate(
        config,
        project_root=project_root,
        source_code_hash=source_code_hash,
        device=device,
        manifest=manifest,
    )
    if sanity_report.get("enabled") and not sanity_report.get("passed"):
        raise RuntimeError(
            f"Baseline sanity gate failed: {sanity_report.get('error') or sanity_report}"
        )

    cfg_hash = config_hash(config)
    atomic_write_json(project_root / "configs" / f"experiment_{cfg_hash}.json", config.to_dict())
    print("Resolved experiment configuration:")
    print(json.dumps(config.to_dict(), indent=2, sort_keys=True))

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
                print(f"FAILED {key}: {exc}")
                if config.run_mode == "smoke":
                    raise

    tables = write_tables(project_root, cfg_hash)
    plots_dir = generate_plots(project_root, cfg_hash)

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
        "cuda": cuda_info,
        "run_mode": config.run_mode,
        "dataset_version": config.dataset_version,
        "difficulty_bump_level": config.difficulty_bump_level,
        "config_hash": cfg_hash,
        "source_code_hash": source_code_hash,
        "requested_variants": requested,
        "completed": completed,
        "resumed": resumed,
        "failed": failed,
        "best_by_variant": best_by_variant,
        "tables": {key: str(path) for key, value in tables.items() for key, path in [(key, value)]},
        "plots_dir": str(plots_dir),
        "manifest": str(project_root / "manifests" / "experiment_manifest.json"),
        "artifacts_under_drive": True,
        "label": "exploratory",
        "wandb_project": config.wandb_project,
        "wandb_entity": config.wandb_entity,
        "sanity_gate": sanity_report,
    }
    summary["tables"] = {key: str(path) for key, path in tables.items()}
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
    atomic_write_json(project_root / "summaries" / f"experiment_summary_{cfg_hash}.json", summary)
    # Latest pointer only; hashed copy preserves prior easy-benchmark summary separately.
    atomic_write_json(project_root / "summaries" / "experiment_summary.json", summary)
    return summary
