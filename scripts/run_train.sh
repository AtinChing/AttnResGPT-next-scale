#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/first_run.yaml}"
MODEL_TYPE="${2:-baseline}"

python -m src.training.train --config "${CONFIG_PATH}" --model "${MODEL_TYPE}"
