"""Pull legacy pre-fix sublayer-output norms; do not compare them as Fig. 5b/5c."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import wandb

ENTITY_PROJECT = "atin5551-uc-davis/attnres-next-scale"
MIN_STEP = 2500
HISTORY_SAMPLES = 500

# From outputs/multiseed_wandb_dump.json (best run per seed)
RUNS = [
    ("baseline", 123, "17f4k8rh"),
    ("baseline", 456, "tinystories_large_baseline_ctx512_steps3000_seed456"),
    ("baseline", 789, "tinystories_large_baseline_ctx512_steps3000_seed789"),
    ("attnres", 123, "tinystories_large_attnres_ctx512_steps3000_seed123"),
    ("attnres", 456, "e444xlsh"),
    ("attnres", 789, "tinystories_large_attnres_ctx512_steps3000_seed789"),
]

METRIC_KEYS = [
    "gradient_norms/blocks.0.attn",
    "gradient_norms/blocks.5.attn",
    "gradient_norms/blocks.11.attn",
    "gradient_norms/blocks.0.mlp",
    "activation_norms/blocks.0.attn",
    "activation_norms/blocks.5.attn",
    "activation_norms/blocks.11.attn",
    "activation_norms/blocks.0.mlp",
]


def fetch_run_series(api: wandb.Api, run_id: str) -> list[dict]:
    run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    df = run.history(samples=HISTORY_SAMPLES, keys=["_step", *METRIC_KEYS], pandas=True)
    if df.empty:
        return []
    df = df.drop_duplicates(subset=["_step"], keep="last").sort_values("_step")
    window = df[df["_step"] >= MIN_STEP]
    rows: list[dict] = []
    for _, row in window.iterrows():
        entry: dict = {"step": int(row["_step"])}
        for key in METRIC_KEYS:
            val = row.get(key)
            if val is not None and val == val:  # not NaN
                entry[key] = round(float(val), 6)
        rows.append(entry)
    return rows


def main() -> None:
    api = wandb.Api()
    payload: dict = {
        "project": ENTITY_PROJECT,
        "step_window": {"min_step_inclusive": MIN_STEP, "max_step_expected": 3000},
        "log_interval_note": "Train metrics logged every 100 steps → ~6 points in window",
        "metric_keys": METRIC_KEYS,
        "runs": [],
    }
    csv_rows: list[dict] = []

    for arch, seed, run_id in RUNS:
        series = fetch_run_series(api, run_id)
        payload["runs"].append(
            {
                "architecture": arch,
                "seed": seed,
                "wandb_run_id": run_id,
                "wandb_url": f"https://wandb.ai/{ENTITY_PROJECT}/runs/{run_id}",
                "n_steps": len(series),
                "history": series,
            }
        )
        for point in series:
            step = point["step"]
            for key in METRIC_KEYS:
                if key in point:
                    csv_rows.append(
                        {
                            "architecture": arch,
                            "seed": seed,
                            "step": step,
                            "metric": key,
                            "value": point[key],
                        }
                    )

    out_json = Path("outputs/multiseed_norm_history_last500.json")
    out_csv = Path("outputs/multiseed_norm_history_last500.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["architecture", "seed", "step", "metric", "value"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote {out_json} ({len(payload['runs'])} runs)")
    print(f"Wrote {out_csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
