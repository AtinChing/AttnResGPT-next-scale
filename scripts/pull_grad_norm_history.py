"""Audit legacy pre-fix W&B output-gradient history (not valid for Fig. 5c)."""
from __future__ import annotations

import json
import os
from pathlib import Path

for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import wandb

RUNS = {
    "baseline": {
        "path": "atin5551-uc-davis/attnres-next-scale/tinystories_large_baseline_ctx512_steps3000_seed456",
        "display": "tinystories_large_baseline_ctx512_steps3000_seed456",
    },
    "attnres": {
        "path": "atin5551-uc-davis/attnres-next-scale/e444xlsh",
        "display": "tinystories_large_attnres_ctx512_steps3000_seed456_rerun1",
    },
}

GRAD_KEYS = [
    "gradient_norms/blocks.0.attn",
    "gradient_norms/blocks.1.attn",
    "gradient_norms/blocks.5.attn",
    "gradient_norms/blocks.11.attn",
    "global_grad_norm",
]

MIN_STEP = 2500
HISTORY_SAMPLES = 500


def summarize(df, col: str) -> dict | None:
    if col not in df.columns:
        return None
    s = df[["_step", col]].dropna()
    if s.empty:
        return {"n_points": 0}
    vals = s[col]
    idx_max = vals.idxmax()
    return {
        "n_points": int(len(s)),
        "step_min": int(s["_step"].min()),
        "step_max": int(s["_step"].max()),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
        "mean": round(float(vals.mean()), 4),
        "max_at_step": int(s.loc[idx_max, "_step"]),
        "last_step": int(s["_step"].iloc[-1]),
        "last_value": round(float(vals.iloc[-1]), 4),
    }


def main() -> None:
    api = wandb.Api()
    out: dict = {
        "note": (
            "last_gradient_norms/* is written to run SUMMARY only (copy of final train step). "
            "History time series uses gradient_norms/* logged each train step."
        ),
        "min_step": MIN_STEP,
        "runs": {},
    }

    for label, meta in RUNS.items():
        run = api.run(meta["path"])
        summary = dict(run.summary)
        df = run.history(samples=HISTORY_SAMPLES, keys=["_step", *GRAD_KEYS], pandas=True)
        df = df.drop_duplicates(subset=["_step"], keep="last").sort_values("_step")
        window = df[df["_step"] >= MIN_STEP].copy()

        probe = run.history(samples=20, pandas=False)
        last_in_history = sorted({k for row in probe for k in row if "last_gradient" in k})

        col0 = "gradient_norms/blocks.0.attn"
        tail = []
        if col0 in window.columns:
            t = window[["_step", col0]].dropna().tail(12)
            tail = [{"step": int(s), "value": round(float(v), 4)} for s, v in zip(t["_step"], t[col0])]

        out["runs"][label] = {
            "wandb_path": meta["path"],
            "display_name": meta["display"],
            "summary_last_gradient_norms_blocks_0_attn": summary.get("last_gradient_norms/blocks.0.attn"),
            "summary_gradient_norms_blocks_0_attn": summary.get("gradient_norms/blocks.0.attn"),
            "last_gradient_norms_in_history": last_in_history,
            "rows_step_ge_min": int(len(window)),
            "gradient_norms_stats": {k: summarize(window, k) for k in GRAD_KEYS},
            "tail_blocks_0_attn": tail,
        }

    out_path = Path("outputs/grad_norm_history_check.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
