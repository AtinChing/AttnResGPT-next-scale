from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import _cleanup_partial_outputs, train_from_config
from src.utils.config import Config, config_to_dict, load_config, load_config_from_dict
from src.utils.logging import build_run_identity, write_global_summary_artifacts


DEFAULT_BATCHING: dict[str, dict[int, dict[str, int]]] = {
    "small": {
        128: {"batch_size": 16, "grad_accum_steps": 1},
        256: {"batch_size": 8, "grad_accum_steps": 2},
        512: {"batch_size": 4, "grad_accum_steps": 4},
    },
    "medium": {
        128: {"batch_size": 8, "grad_accum_steps": 2},
        256: {"batch_size": 4, "grad_accum_steps": 4},
        512: {"batch_size": 2, "grad_accum_steps": 8},
    },
}


OOM_FALLBACK = {
    "medium": {
        512: {"batch_size": 2, "grad_accum_steps": 16},
    }
}


def _load_existing_summary(output_root: str | Path, run_name: str) -> dict[str, Any]:
    path = Path(output_root) / "runs" / run_name / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find existing summary at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _contexts_for_stage(config: Config) -> list[int]:
    if config.experiment.stage == "first_run":
        return [128, 512]
    return [128, 256, 512]


def _models_for_stage(_config: Config) -> list[str]:
    return ["baseline", "attnres"]


def _prepare_run_config(base_config: Config, *, model_type: str, context: int) -> Config:
    payload = deepcopy(config_to_dict(base_config))
    payload["model"]["architecture"] = model_type
    payload["model"]["max_seq_len"] = context
    payload["data"]["block_size"] = context
    payload["experiment"]["name"] = f"{base_config.experiment.stage}_{base_config.model.size_name}_{model_type}_ctx{context}"

    batching = DEFAULT_BATCHING[base_config.model.size_name][context]
    payload["data"]["batch_size"] = batching["batch_size"]
    payload["data"]["eval_batch_size"] = batching["batch_size"]
    payload["training"]["grad_accum_steps"] = batching["grad_accum_steps"]
    return load_config_from_dict(payload)
def _paired_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in summary_rows:
        key = (str(row["size"]), int(row["context"]))
        grouped.setdefault(key, {})[str(row["model"])] = row

    paired: list[dict[str, Any]] = []
    for (size, context), runs in sorted(grouped.items()):
        if "baseline" not in runs or "attnres" not in runs:
            continue
        baseline = runs["baseline"]
        attnres = runs["attnres"]
        paired.append(
            {
                "size": size,
                "context": context,
                "baseline_val_loss": baseline["val_loss"],
                "attnres_val_loss": attnres["val_loss"],
                "delta_val_loss": baseline["val_loss"] - attnres["val_loss"],
                "baseline_ppl": baseline["perplexity"],
                "attnres_ppl": attnres["perplexity"],
                "delta_ppl": baseline["perplexity"] - attnres["perplexity"],
                "baseline_params": baseline.get("parameter_count_total"),
                "attnres_params": attnres.get("parameter_count_total"),
                "parameter_delta_pct": (
                    100.0
                    * (float(attnres.get("parameter_count_total", 0)) - float(baseline.get("parameter_count_total", 0)))
                    / max(1.0, float(baseline.get("parameter_count_total", 0)))
                ),
            }
        )
    return paired


def _run_one(config: Config, *, skip_existing: bool) -> dict[str, Any]:
    identity = build_run_identity(config)
    summary_path = Path(config.logging.output_root) / "runs" / identity.run_name / "run_summary.json"
    if skip_existing and summary_path.exists():
        return _load_existing_summary(config.logging.output_root, identity.run_name)

    try:
        return train_from_config(config)
    except RuntimeError as error:
        message = str(error).lower()
        if "out of memory" not in message:
            raise
        size_name = config.model.size_name
        context = config.data.block_size
        fallback = OOM_FALLBACK.get(size_name, {}).get(context)
        if fallback is None:
            raise
        torch.cuda.empty_cache()
        retry_payload = deepcopy(config_to_dict(config))
        checkpoint_dir = Path(config.logging.output_root) / "checkpoints" / identity.run_name
        if checkpoint_dir.exists() and any(checkpoint_dir.glob("step_*.pt")):
            retry_payload["training"]["resume_from"] = str(checkpoint_dir)
            retry_payload["training"]["allow_resume_mismatch"] = True
        else:
            _cleanup_partial_outputs(config, identity)
        retry_payload["data"]["batch_size"] = fallback["batch_size"]
        retry_payload["data"]["eval_batch_size"] = fallback["batch_size"]
        retry_payload["training"]["grad_accum_steps"] = fallback["grad_accum_steps"]
        retry_config = load_config_from_dict(retry_payload)
        summary = train_from_config(retry_config)
        metadata_path = Path(config.logging.output_root) / "runs" / summary["run_name"] / "run_metadata.json"
        if metadata_path.exists():
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["oom_fallback_used"] = True
            payload["fallback_batch_size"] = fallback["batch_size"]
            payload["fallback_grad_accum_steps"] = fallback["grad_accum_steps"]
            metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a matched baseline vs AttnRes scale experiment.")
    parser.add_argument("--config", required=True, help="Path to a size config YAML.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.config)
    summary_rows: list[dict[str, Any]] = []

    for context in _contexts_for_stage(base_config):
        for model_type in _models_for_stage(base_config):
            run_config = _prepare_run_config(base_config, model_type=model_type, context=context)
            summary_rows.append(_run_one(run_config, skip_existing=args.skip_existing))

    paired_rows = _paired_rows(summary_rows)
    write_global_summary_artifacts(base_config.logging.output_root, summary_rows, paired_rows)
    print(json.dumps({"summaries": summary_rows, "paired": paired_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
