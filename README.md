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

The paired artifact includes the required loss and perplexity deltas and also logs parameter-count deltas for fairness checks.

`delta_val_loss` and `delta_ppl` in `paired_comparisons.csv` are defined as:

```text
baseline - attnres
```

So positive values indicate AttnRes improvement.

## Colab Workflow

Use:

- `notebooks/0_full_pipeline.ipynb`
- `notebooks/1_train_scale.ipynb`
- `notebooks/2_analyze_results.ipynb`

The notebooks first try:

1. `/content/AttnResGPT-next-scale`
2. `/content/drive/MyDrive/AttnResGPT-next-scale`
3. `ATTNRES_REPO_URL` if you set it in Colab
4. the placeholder `REPO_URL` inside the notebook

Recommended order:

1. Easiest Colab path: run `0_full_pipeline.ipynb`
2. Modular path: run the first-run preset in `1_train_scale.ipynb`
3. Confirm the four runs completed
4. Open `2_analyze_results.ipynb`
5. Review the consolidated summary table and paired comparison artifact
6. Only then run the broader sweep

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
