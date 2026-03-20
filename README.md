# AttnResGPT Next-Scale

Clean, Colab-friendly research code for comparing standard residual connections against Attention Residuals (AttnRes) in small-to-medium GPT-style causal language models.

## What This Repo Is For

This repository replaces the earlier notebook-mutated workflow with a modular experiment pipeline that:

- keeps the original `0_colab_full_pipeline.ipynb` strictly read-only
- ports the currently working AttnRes formulation as faithfully as possible for v1
- supports a required first-run experiment before the larger sweep
- saves deterministic run metadata, config snapshots, resume-safe checkpoints, and paired comparison artifacts

## Research Question

Primary question:

- Does Attention Residuals become more beneficial at larger scale and longer context?

Secondary questions:

- Does it stabilize activations?
- Does it improve depth utilization?
- Does it improve long-context prediction quality?

## Baseline vs AttnRes

Baseline block:

```text
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

AttnRes block in this repo:

- preserves the working small-scale formulation from the earlier project
- uses depth-wise softmax mixing over prior states
- keeps learned per-sublayer depth queries
- optionally uses a sliding depth window
- keeps a final depth-attention readout

This is intentionally minimal. v1 is meant to preserve the working behavior, not redesign AttnRes.

## First-Run Experiment

Run this before any full sweep:

- size: `SMALL`
- contexts: `128`, `512`
- models: `baseline`, `attnres`
- steps: `300`

This produces four runs and verifies:

- training works end to end
- checkpoints can be resumed safely
- the analysis notebook can read the outputs without manual edits

## Repo Layout

```text
src/
  models/
  training/
  data/
  metrics/
  utils/
experiments/
notebooks/
scripts/
configs/
tests/
outputs/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main Entry Points

Train one run:

```bash
python -m src.training.train --config configs/first_run.yaml --model baseline
python -m src.training.train --config configs/first_run.yaml --model attnres
```

Run the first-run matrix:

```bash
python experiments/scale_experiment.py --config configs/first_run.yaml
```

Run the broader sweep:

```bash
python experiments/scale_experiment.py --config configs/small.yaml
python experiments/scale_experiment.py --config configs/medium.yaml
python experiments/scale_experiment.py --config configs/large.yaml
```

Evaluate a checkpoint:

```bash
python -m src.training.eval --config configs/first_run.yaml --checkpoint outputs/checkpoints/<run_name>/step_0000300.pt
```

## Outputs

Each run is deterministic and writes to:

```text
outputs/runs/{dataset}_{size}_{model}_ctx{context}_steps{max_steps}_seed{seed}/
```

Every run directory contains:

- `config.snapshot.yaml`
- `config.hash.txt`
- `run_metadata.json`
- `train_metrics.jsonl`
- `val_metrics.jsonl`
- `run_summary.json`
- `run_summary.csv`
- `probes/`

Checkpoints are stored under:

```text
outputs/checkpoints/{run_name}/
```

Global artifacts:

- `outputs/logs/run_summaries.csv`
- `outputs/logs/consolidated_summary_table.csv`
- `outputs/logs/paired_comparisons.csv`
- `outputs/summary_large.csv`
- `outputs/summary_large_comparison.csv`

The paired artifact includes the required loss and perplexity deltas and also logs parameter-count deltas for fairness checks.

`delta_val_loss` and `delta_ppl` in `paired_comparisons.csv` are defined as:

```text
baseline - attnres
```

So positive values indicate AttnRes improvement.

## W&B Logging

Weights & Biases logging is enabled by default through `logging.wandb`.

- If `WANDB_API_KEY` is set, the repo uses online W&B logging.
- If no API key is set, the repo falls back to offline W&B logging automatically.
- Local JSONL, CSV, checkpoints, and plots are still written even when W&B is disabled or unavailable.

Useful controls:

```bash
export WANDB_API_KEY=...            # enable cloud-synced runs
export WANDB_MODE=disabled          # disable W&B for a session
python -m src.training.train --config configs/first_run.yaml --overrides logging.wandb.enabled=false
```

## Local Colab-Kernel Workflow

Use:

- `notebooks/0_full_pipeline.ipynb`
- `notebooks/1_train_scale.ipynb`
- `notebooks/2_analyze_results.ipynb`

The notebooks now use the local repo/filesystem directly. They look for the repo in:

1. the current working tree
2. the parent of the current working tree
3. `/content/AttnResGPT-next-scale` if it has already been synced locally

There is no Google Drive dependency and no agent tooling dependency in the notebook flow.

Recommended order:

1. Easiest single-notebook path: run `0_full_pipeline.ipynb`
2. Modular path: run the first-run preset in `1_train_scale.ipynb`
3. Confirm the four runs completed
4. Run the new large sweep with `configs/large.yaml` when you are ready
5. Open `2_analyze_results.ipynb` to review the consolidated summary tables and paired comparison artifacts

## Design Choices and Assumptions

- TinyStories is the default real-text dataset.
- GPT-2 tokenizer is the default tokenizer.
- The original notebook is a reference only and is never used at runtime.
- The current working AttnRes formulation is preserved as closely as possible in v1.
- The repo is single-GPU only and tuned for Colab T4 constraints.

## Faithfulness vs Simplification

- Faithful:
  - depth-wise softmax mixing over previous states
  - learned queries per depth-mixing site
  - optional sliding window
  - final depth readout
- Simplified:
  - no distributed training
  - no large-scale systems optimizations
  - no extra routing losses or architectural embellishments

## Potential Pitfalls

- `MEDIUM + ctx512` may require the built-in accumulation fallback on a T4.
- TinyStories streaming depends on Hugging Face availability and can be slower on first download.
- Deterministic run naming means rerunning the same config should use resume or clean the old outputs first.

## Next Research Extensions

- add longer-context held-out evaluation sets
- compare full vs sliding-window depth attention
- add layer-ablation studies on the new runs
- test whether depth usage shifts with training length beyond the short pilot budget
