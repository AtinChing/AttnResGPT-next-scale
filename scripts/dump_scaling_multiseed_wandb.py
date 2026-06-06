"""Pull completed GPT scaling multi-seed results from W&B.

Writes outputs/scaling_multiseed_wandb_dump.json and prints grouped tables.
Loads WANDB_API_KEY from .env (never printed).
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ENTITY = "atin5551-uc-davis"
PROJECT = "attnres-next-scale"

NAME_RE = re.compile(
    r"^tinystories_(?P<size>small|medium|large)_(?P<arch>baseline|attnres)"
    r"_ctx(?P<ctx>\d+)_steps(?P<steps>\d+)_seed(?P<seed>\d+)"
    r"(?:_rerun\d+)?$"
)

SIZE_STEP_FLOOR = {"small": 300, "medium": 200, "large": 3000}
SMOKE_RE = re.compile(r"steps2_|seedtest|_smoke")


def load_env() -> None:
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if line.strip() and not line.strip().startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def expected_budget(size: str, steps_from_name: int) -> int:
    """Use run-name step budget when it exceeds the size default (e.g. extended multiseed)."""
    floor = SIZE_STEP_FLOOR[size]
    return max(floor, steps_from_name)


def is_smoke(name: str) -> bool:
    return bool(SMOKE_RE.search(name))


def run_priority(name: str, final_step: int, val_loss: float | None) -> tuple:
    """Prefer _rerun1, then higher final_step, then lower val_loss."""
    rerun_rank = 1 if "_rerun" in name else 0
    vl = val_loss if val_loss is not None else float("inf")
    return (rerun_rank, final_step, -vl if vl != float("inf") else 0)


def extract_row(run: Any) -> dict[str, Any] | None:
    m = NAME_RE.match(run.name)
    if not m:
        return None
    if is_smoke(run.name):
        return None
    if run.state != "finished":
        return None

    summary = dict(run.summary)
    final_step = summary.get("final_step") or summary.get("_step")
    if final_step is None:
        return None
    final_step = int(final_step)

    size = m.group("size")
    steps_from_name = int(m.group("steps"))
    budget = expected_budget(size, steps_from_name)
    if final_step < budget:
        return None

    val_loss = summary.get("val_loss")
    if val_loss is None:
        return None

    seed = int(m.group("seed"))
    arch = m.group("arch")
    ctx = int(m.group("ctx"))

    row: dict[str, Any] = {
        "run_name": run.name,
        "wandb_id": run.id,
        "wandb_url": run.url,
        "model_size": size,
        "context_length": ctx,
        "architecture": arch,
        "seed": seed,
        "step_budget": budget,
        "steps_in_name": steps_from_name,
        "core": {
            "seed": seed,
            "final_val_loss": float(val_loss),
            "final_perplexity": summary.get("perplexity"),
            "second_half_loss": summary.get("second_half_loss"),
            "final_step": final_step,
            "total_parameters": summary.get("parameter_count_total"),
            "trainable_parameters": summary.get("parameter_count_trainable"),
        },
    }
    if arch == "attnres":
        row["depth_utilization"] = {
            "mean_embedding_contribution": summary.get("mean_embedding_contribution"),
            "mean_early_contribution": summary.get("mean_early_contribution"),
            "mean_late_contribution": summary.get("mean_late_contribution"),
            "mean_depth_attention_entropy": summary.get("mean_depth_attention_entropy"),
        }
    return row


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = (
            row["model_size"],
            row["context_length"],
            row["architecture"],
            row["seed"],
            row["steps_in_name"],
        )
        cur = best.get(key)
        if cur is None:
            best[key] = row
            continue
        c = row["core"]
        o = cur["core"]
        if run_priority(row["run_name"], c["final_step"], c["final_val_loss"]) > run_priority(
            cur["run_name"], o["final_step"], o["final_val_loss"]
        ):
            best[key] = row
    return list(best.values())


def build_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, dict[str, Any]] = {}
    for row in rows:
        key = (row["model_size"], row["context_length"], row["architecture"], row["steps_in_name"])
        if key not in grouped:
            grouped[key] = {
                "model_size": row["model_size"],
                "context_length": row["context_length"],
                "architecture": row["architecture"],
                "step_budget": row["step_budget"],
                "steps_in_name": row["steps_in_name"],
                "seeds": [],
                "core": [],
                "depth_utilization": [],
            }
        g = grouped[key]
        g["seeds"].append(row["seed"])
        g["core"].append(row["core"])
        if "depth_utilization" in row:
            g["depth_utilization"].append(row["depth_utilization"])

    groups = []
    for g in sorted(grouped.values(), key=lambda x: (x["model_size"], x["context_length"], x["architecture"], x["steps_in_name"])):
        g["seeds"] = sorted(set(g["seeds"]))
        groups.append(g)
    return groups


def paired_seed_flags(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag when baseline and attnres seed sets differ for same (size, ctx, steps)."""
    by_setting: dict[tuple, dict[str, set[int]]] = defaultdict(lambda: {"baseline": set(), "attnres": set()})
    for g in groups:
        key = (g["model_size"], g["context_length"], g["steps_in_name"])
        by_setting[key][g["architecture"]] = set(g["seeds"])

    flags = []
    for key in sorted(by_setting):
        baseline_seeds = sorted(by_setting[key]["baseline"])
        attnres_seeds = sorted(by_setting[key]["attnres"])
        if not baseline_seeds and not attnres_seeds:
            continue
        paired = baseline_seeds == attnres_seeds and len(baseline_seeds) > 0
        flags.append(
            {
                "model_size": key[0],
                "context_length": key[1],
                "steps_in_name": key[2],
                "baseline_seeds": baseline_seeds,
                "attnres_seeds": attnres_seeds,
                "paired": paired,
            }
        )
    return flags


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def print_tables(groups: list[dict[str, Any]], flags: list[dict[str, Any]]) -> None:
    print("=" * 100)
    print("SCALING MULTI-SEED W&B DUMP")
    print(f"Project: {ENTITY}/{PROJECT}")
    print("Filter: finished, smoke/crash excluded, final_step >= size step budget")
    print("=" * 100)

    print("\n## Paired seed check\n")
    print(f"{'Size':<8} {'Ctx':>5} {'Steps':>6} {'Baseline seeds':<22} {'AttnRes seeds':<22} {'Paired?':>8}")
    print("-" * 80)
    for f in flags:
        b = ", ".join(map(str, f["baseline_seeds"])) or "—"
        a = ", ".join(map(str, f["attnres_seeds"])) or "—"
        mark = "Yes" if f["paired"] else "NO"
        print(f"{f['model_size']:<8} {f['context_length']:>5} {f['steps_in_name']:>6} {b:<22} {a:<22} {mark:>8}")

    for size in ("small", "medium", "large"):
        size_groups = [g for g in groups if g["model_size"] == size]
        if not size_groups:
            continue
        print(f"\n## {size.upper()}\n")
        for g in size_groups:
            arch = g["architecture"]
            ctx = g["context_length"]
            steps = g["steps_in_name"]
            print(f"### {arch} | ctx={ctx} | steps={steps} | seeds={g['seeds']}\n")
            hdr = (
                f"{'seed':>5} {'val_loss':>9} {'ppl':>8} {'2nd_half':>9} "
                f"{'step':>6} {'params':>10} {'trainable':>10}"
            )
            print(hdr)
            print("-" * len(hdr))
            for c in sorted(g["core"], key=lambda x: x["seed"]):
                print(
                    f"{c['seed']:>5} {fmt(c['final_val_loss']):>9} {fmt(c['final_perplexity'], 2):>8} "
                    f"{fmt(c['second_half_loss']):>9} {c['final_step']:>6} "
                    f"{c['total_parameters'] or '—':>10} {c['trainable_parameters'] or '—':>10}"
                )
            if arch == "attnres" and g["depth_utilization"]:
                print()
                dhdr = (
                    f"{'seed':>5} {'embed':>9} {'early':>9} {'late':>9} {'entropy':>9}"
                )
                print("DEPTH UTILIZATION")
                print(dhdr)
                print("-" * len(dhdr))
                depth_by_seed = {c["seed"]: d for c, d in zip(g["core"], g["depth_utilization"])}
                for seed in sorted(depth_by_seed):
                    d = depth_by_seed[seed]
                    print(
                        f"{seed:>5} {fmt(d['mean_embedding_contribution']):>9} "
                        f"{fmt(d['mean_early_contribution']):>9} {fmt(d['mean_late_contribution']):>9} "
                        f"{fmt(d['mean_depth_attention_entropy']):>9}"
                    )
            print()


def main() -> None:
    load_env()
    import wandb

    api = wandb.Api(timeout=120)
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=300))

    raw_rows = []
    excluded = []
    for run in all_runs:
        if not run.name.startswith("tinystories_"):
            continue
        row = extract_row(run)
        if row is None:
            reason = []
            if is_smoke(run.name):
                reason.append("smoke")
            elif run.state != "finished":
                reason.append(run.state)
            else:
                m = NAME_RE.match(run.name)
                if not m:
                    reason.append("name_mismatch")
                else:
                    summary = dict(run.summary)
                    final_step = summary.get("final_step") or summary.get("_step")
                    size = m.group("size")
                    steps = int(m.group("steps"))
                    budget = expected_budget(size, steps)
                    if final_step is None or int(final_step) < budget:
                        reason.append(f"under_budget({final_step}<{budget})")
                    elif summary.get("val_loss") is None:
                        reason.append("no_val_loss")
            excluded.append({"name": run.name, "state": run.state, "reason": ",".join(reason) or "filtered"})
            continue
        raw_rows.append(row)

    rows = dedupe_rows(raw_rows)
    groups = build_groups(rows)
    flags = paired_seed_flags(groups)

    payload = {
        "project": f"{ENTITY}/{PROJECT}",
        "filters": {
            "state": "finished",
            "exclude_smoke": True,
            "size_step_floor": SIZE_STEP_FLOOR,
            "budget_rule": "max(size_floor, steps_in_run_name)",
        },
        "paired_seed_check": flags,
        "groups": groups,
        "runs": rows,
        "excluded_tinystories": excluded,
    }

    out = REPO / "outputs" / "scaling_multiseed_wandb_dump.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {out} ({len(rows)} runs, {len(groups)} groups, {len(excluded)} excluded)\n")
    print_tables(groups, flags)


if __name__ == "__main__":
    main()
