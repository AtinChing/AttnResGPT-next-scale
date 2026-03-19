from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train import train_from_config
from src.utils.config import Config, config_to_dict, load_config, load_config_from_dict


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


def _prepare_context_config(base_config: Config, *, model_type: str, context: int) -> Config:
    payload = deepcopy(config_to_dict(base_config))
    payload["model"]["architecture"] = model_type
    payload["model"]["max_seq_len"] = context
    payload["data"]["block_size"] = context
    payload["experiment"]["name"] = f"context_{base_config.model.size_name}_{model_type}_ctx{context}"
    batching = DEFAULT_BATCHING[base_config.model.size_name][context]
    payload["data"]["batch_size"] = batching["batch_size"]
    payload["data"]["eval_batch_size"] = batching["batch_size"]
    payload["training"]["grad_accum_steps"] = batching["grad_accum_steps"]
    return load_config_from_dict(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a context-length experiment for one architecture.")
    parser.add_argument("--config", required=True, help="Path to the YAML config.")
    parser.add_argument("--model", required=True, choices=["baseline", "attnres"])
    args = parser.parse_args()

    base_config = load_config(args.config)
    contexts = [128, 256, 512]
    summaries = []
    for context in contexts:
        summaries.append(train_from_config(_prepare_context_config(base_config, model_type=args.model, context=context)))
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
