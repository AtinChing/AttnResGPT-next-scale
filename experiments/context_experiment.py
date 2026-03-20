from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scale_experiment import _contexts_for_config, _resolve_batching
from src.training.train import train_from_config
from src.utils.config import Config, config_to_dict, load_config, load_config_from_dict


def _prepare_context_config(base_config: Config, *, model_type: str, context: int) -> Config:
    payload = deepcopy(config_to_dict(base_config))
    payload['model']['architecture'] = model_type
    payload['model']['max_seq_len'] = context
    payload['data']['block_size'] = context
    payload['experiment']['name'] = f"context_{base_config.model.size_name}_{model_type}_ctx{context}"

    batching = _resolve_batching(base_config, context)
    payload['data']['batch_size'] = batching['batch_size']
    payload['data']['eval_batch_size'] = batching['batch_size']
    payload['training']['grad_accum_steps'] = batching['grad_accum_steps']
    return load_config_from_dict(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a context-length experiment for one architecture.')
    parser.add_argument('--config', required=True, help='Path to the YAML config.')
    parser.add_argument('--model', required=True, choices=['baseline', 'attnres'])
    args = parser.parse_args()

    base_config = load_config(args.config)
    contexts = _contexts_for_config(base_config)
    summaries = []
    for context in contexts:
        summaries.append(train_from_config(_prepare_context_config(base_config, model_type=args.model, context=context)))
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
