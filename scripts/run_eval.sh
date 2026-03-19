#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/first_run.yaml}"
CHECKPOINT_PATH="${2:?checkpoint path required}"
MODEL_TYPE="${3:-baseline}"

python -m src.training.eval --config "${CONFIG_PATH}" --checkpoint "${CHECKPOINT_PATH}" --model "${MODEL_TYPE}"
