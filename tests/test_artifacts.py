from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging import write_global_summary_artifacts


def test_global_summary_and_paired_csvs_are_written(tmp_path: Path) -> None:
    summary_rows = [
        {
            "run_name": "tinystories_small_baseline_ctx128_steps300_seed1337",
            "model": "baseline",
            "size": "small",
            "context": 128,
            "val_loss": 2.0,
            "perplexity": 7.0,
            "second_half_loss": 2.2,
            "mean_activation_norm_last_layer": 1.1,
            "mean_early_contribution": None,
            "mean_late_contribution": None,
        },
        {
            "run_name": "tinystories_small_attnres_ctx128_steps300_seed1337",
            "model": "attnres",
            "size": "small",
            "context": 128,
            "val_loss": 1.9,
            "perplexity": 6.7,
            "second_half_loss": 2.0,
            "mean_activation_norm_last_layer": 1.0,
            "mean_early_contribution": 0.35,
            "mean_late_contribution": 0.44,
        },
    ]
    paired_rows = [
        {
            "size": "small",
            "context": 128,
            "baseline_val_loss": 2.0,
            "attnres_val_loss": 1.9,
            "delta_val_loss": 0.1,
            "baseline_ppl": 7.0,
            "attnres_ppl": 6.7,
            "delta_ppl": 0.3,
            "baseline_params": 1000,
            "attnres_params": 1010,
            "parameter_delta_pct": 1.0,
        }
    ]

    write_global_summary_artifacts(tmp_path, summary_rows, paired_rows)

    summary_csv = pd.read_csv(tmp_path / "logs" / "run_summaries.csv")
    consolidated_csv = pd.read_csv(tmp_path / "logs" / "consolidated_summary_table.csv")
    paired_csv = pd.read_csv(tmp_path / "logs" / "paired_comparisons.csv")

    assert "mean_activation_norm_last_layer" in summary_csv.columns
    assert "mean_early_contribution" in summary_csv.columns
    assert list(consolidated_csv.columns) == [
        "model",
        "size",
        "context",
        "val_loss",
        "perplexity",
        "second_half_loss",
        "mean_activation_norm_last_layer",
        "mean_early_contribution",
        "mean_late_contribution",
    ]
    assert list(paired_csv.columns[:8]) == [
        "size",
        "context",
        "baseline_val_loss",
        "attnres_val_loss",
        "delta_val_loss",
        "baseline_ppl",
        "attnres_ppl",
        "delta_ppl",
    ]
    assert "parameter_delta_pct" in paired_csv.columns
