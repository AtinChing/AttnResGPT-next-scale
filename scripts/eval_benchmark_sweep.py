#!/usr/bin/env python3
"""Sweep benchmark eval across training checkpoints (Fig. crossover probe).

Runs EleutherAI lm-evaluation-harness tasks on every ``step_*.pt`` checkpoint
and writes a table of accuracy vs depth (tokens seen), with above-chance flags.

Example (PowerShell, from repo root):

    python scripts/eval_benchmark_sweep.py ^
      --config configs/fineweb_200m_budget.yaml ^
      --checkpoint-dir "D:\\path\\to\\checkpoints\\fineweb_edu_medium_baseline_ctx1024_steps15259_seed1337" ^
      --device cuda ^
      --batch-size 4

Quick smoke test on the latest checkpoint only:

    python scripts/eval_benchmark_sweep.py --config configs/fineweb_200m_budget.yaml ^
      --checkpoint-dir outputs/checkpoints/<run_name> --limit 50 --latest-only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from lm_eval import evaluator

# Register adapter import path when invoked as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.benchmark_tasks import TASK_SPECS, metric_value
from src.eval.lm_eval_gpt import AttnResGPTLM
from src.utils.config import load_config
from src.utils.runtime import get_device


DEFAULT_TASKS = ("hellaswag", "lambada_openai", "arc_easy")


@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    step: int
    tokens_seen: int | None


def _parse_step(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("step_"):
        raise ValueError(f"Unexpected checkpoint name: {path.name}")
    # Local saves: step_0002290.pt. WandB artifact pulls: step_0002290-002.pt.
    step_token = stem.removeprefix("step_").split("-", 1)[0]
    return int(step_token)


def discover_checkpoints(checkpoint_dir: Path, *, latest_only: bool) -> list[CheckpointRef]:
    paths = sorted(checkpoint_dir.glob("step_*.pt"), key=_parse_step)
    if not paths:
        raise FileNotFoundError(f"No step_*.pt checkpoints found under {checkpoint_dir}")
    if latest_only:
        paths = [paths[-1]]

    refs: list[CheckpointRef] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        refs.append(
            CheckpointRef(
                path=path,
                step=int(payload.get("global_step", _parse_step(path))),
                tokens_seen=payload.get("cumulative_tokens_seen"),
            )
        )
    return refs


def _metric_value(task_results: dict[str, Any], task_name: str) -> float | None:
    return metric_value(task_results, task_name)


def _extract_task_rows(
    eval_results: dict[str, Any],
    *,
    checkpoint: CheckpointRef,
    run_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_results = eval_results.get("results", {})
    for task_name in task_results:
        if task_name not in TASK_SPECS:
            continue
        spec = TASK_SPECS[task_name]
        score = _metric_value(task_results[task_name], task_name)
        chance = float(spec["chance"])
        above_chance = score is not None and score > chance
        rows.append(
            {
                "run_name": run_name,
                "checkpoint": str(checkpoint.path),
                "step": checkpoint.step,
                "tokens_seen": checkpoint.tokens_seen,
                "tokens_millions": (
                    None if checkpoint.tokens_seen is None else checkpoint.tokens_seen / 1_000_000.0
                ),
                "task": task_name,
                "metric": spec["metric"],
                "score": score,
                "chance": chance,
                "above_chance": above_chance,
                "margin_over_chance": None if score is None else score - chance,
            }
        )
    return rows


def _summarize_crossover(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"tasks": {}, "first_all_tasks_above_chance": None}
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)

    for task_name, task_rows in by_task.items():
        ordered = sorted(task_rows, key=lambda row: row["step"])
        first_cross = next((row for row in ordered if row["above_chance"]), None)
        best = max(
            (row for row in ordered if row["score"] is not None),
            key=lambda row: row["score"],
            default=None,
        )
        summary["tasks"][task_name] = {
            "first_above_chance_step": None if first_cross is None else first_cross["step"],
            "first_above_chance_tokens_m": None
            if first_cross is None
            else first_cross["tokens_millions"],
            "first_above_chance_score": None if first_cross is None else first_cross["score"],
            "best_score": None if best is None else best["score"],
            "best_step": None if best is None else best["step"],
        }

    tracked_tasks = set(by_task)
    ordered_steps = sorted({row["step"] for row in rows})
    for step in ordered_steps:
        step_rows = {
            row["task"]: row
            for row in rows
            if row["step"] == step and row["task"] in tracked_tasks
        }
        if tracked_tasks and tracked_tasks.issubset(step_rows) and all(
            step_rows[task]["above_chance"] for task in tracked_tasks
        ):
            summary["first_all_tasks_above_chance"] = step
            break

    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, overrides=list(args.overrides or []))
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = discover_checkpoints(checkpoint_dir, latest_only=args.latest_only)
    device = get_device(args.device)
    tasks = tuple(args.tasks)
    unknown = [task for task in tasks if task not in TASK_SPECS]
    if unknown:
        raise ValueError(f"Unsupported tasks {unknown}. Supported: {sorted(TASK_SPECS)}")

    run_name = checkpoint_dir.name
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir.parent.parent / "benchmark_sweeps" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    per_checkpoint: list[dict[str, Any]] = []

    for index, checkpoint in enumerate(checkpoints, start=1):
        print(
            f"\n[{index}/{len(checkpoints)}] step={checkpoint.step} "
            f"tokens={checkpoint.tokens_seen}  {checkpoint.path.name}",
            flush=True,
        )
        lm = AttnResGPTLM(
            config=config,
            checkpoint_path=checkpoint.path,
            device=device,
            batch_size=args.batch_size,
            mixed_precision=args.mixed_precision,
            amp_dtype=args.amp_dtype,
        )
        try:
            eval_results = evaluator.simple_evaluate(
                model=lm,
                tasks=list(tasks),
                batch_size=args.batch_size,
                device=str(device),
                limit=args.limit,
                log_samples=False,
            )
        finally:
            del lm
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()

        rows = _extract_task_rows(eval_results, checkpoint=checkpoint, run_name=run_name)
        all_rows.extend(rows)
        per_checkpoint.append(
            {
                "step": checkpoint.step,
                "tokens_seen": checkpoint.tokens_seen,
                "checkpoint": str(checkpoint.path),
                "results": eval_results.get("results", {}),
            }
        )

        for row in rows:
            score = "n/a" if row["score"] is None else f"{row['score']:.4f}"
            flag = "YES" if row["above_chance"] else "no"
            print(
                f"  {row['task']:16s} {row['metric']:14s} {score:>8s}  "
                f"(chance={row['chance']:.2f})  above_chance={flag}",
                flush=True,
            )

    summary = _summarize_crossover(all_rows)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "checkpoint_dir": str(checkpoint_dir),
        "tasks": list(tasks),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "device": str(device),
        "summary": summary,
        "rows": all_rows,
        "checkpoints": per_checkpoint,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"benchmark_sweep_{stamp}.json"
    csv_path = output_dir / f"benchmark_sweep_{stamp}.csv"
    latest_json = output_dir / "benchmark_sweep_latest.json"
    latest_csv = output_dir / "benchmark_sweep_latest.csv"

    _write_json(json_path, payload)
    _write_csv(csv_path, all_rows)
    _write_json(latest_json, payload)
    _write_csv(latest_csv, all_rows)

    print("\n=== crossover summary ===")
    for task_name, task_summary in summary["tasks"].items():
        print(
            f"{task_name}: first above chance at step={task_summary['first_above_chance_step']} "
            f"(tokens_m={task_summary['first_above_chance_tokens_m']}, "
            f"score={task_summary['first_above_chance_score']})",
        )
    print(f"first checkpoint above chance on ALL tasks: step={summary['first_all_tasks_above_chance']}")
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark crossover sweep over saved training checkpoints.",
    )
    parser.add_argument("--config", default="configs/fineweb_200m_budget.yaml")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory containing step_*.pt files.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help=f"lm-eval tasks (default: {' '.join(DEFAULT_TASKS)})",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--batch-size", type=int, default=4, help="Harness batch size for loglikelihood.")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task (smoke tests).")
    parser.add_argument("--latest-only", action="store_true", help="Evaluate only the highest step checkpoint.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write CSV/JSON (default: outputs/benchmark_sweeps/<run_name>).",
    )
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override training mixed_precision (default: use config).",
    )
    parser.add_argument("--amp-dtype", default=None, help="Override amp dtype (e.g. bfloat16).")
    parser.add_argument("--overrides", nargs="*", default=[], help="Optional config key=value overrides.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
