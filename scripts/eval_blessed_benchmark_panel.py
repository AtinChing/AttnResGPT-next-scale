#!/usr/bin/env python3
"""Evaluate the full downstream panel on blessed 30M / 90M final checkpoints.

Streams one final checkpoint at a time into a local scratch directory (important
when Drive/cloud copies are large and disk is tight), runs lm-evaluation-harness
tasks, then deletes the scratch copy before the next run.

Benchmarks (all loglikelihood / multiple-choice via AttnResGPTLM):
  hellaswag, lambada_openai, piqa, winogrande, arc_easy, arc_challenge,
  openbookqa, boolq, sciq

Example (Apple Silicon, from repo root):

    python scripts/eval_blessed_benchmark_panel.py --device auto

Sanity check that multi-task == single-task (request-reorder fix):

    python scripts/eval_blessed_benchmark_panel.py --sanity-check-only --limit 32
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from lm_eval import evaluator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.benchmark_tasks import PANEL_TASKS, TASK_SPECS, metric_value
from src.eval.lm_eval_gpt import AttnResGPTLM
from src.utils.config import load_config
from src.utils.runtime import get_device

DEFAULT_DRIVE_ROOT = Path(
    "/Users/atin5551/Library/CloudStorage/GoogleDrive-atin5551@gmail.com/"
    "My Drive/AttnResGPT-next-scale-artifacts"
)

VARIANT_OVERRIDES: dict[str, list[str]] = {
    "baseline": [
        "model.architecture=baseline",
        "model.attnres.enabled=false",
    ],
    "full_attnres": [
        "model.architecture=attnres",
        "model.attnres.enabled=true",
        "model.attnres.mode=full",
    ],
    "block_attnres": [
        "model.architecture=block_attnres",
    ],
}

# Folder stem uses architecture short name (attnres), not the suite label (full_attnres).
VARIANT_RUN_TOKEN: dict[str, str] = {
    "baseline": "baseline",
    "full_attnres": "attnres",
    "block_attnres": "block_attnres",
}


@dataclass(frozen=True)
class BlessedRun:
    scale: str  # "30m" | "90m"
    variant: str
    seed: int
    config_path: str
    run_name: str
    final_step: int

    @property
    def label(self) -> str:
        return f"{self.scale}/{self.variant}/seed{self.seed}"


def blessed_inventory() -> list[BlessedRun]:
    runs: list[BlessedRun] = []
    for seed in (123, 1337, 456):
        for variant, token in VARIANT_RUN_TOKEN.items():
            runs.append(
                BlessedRun(
                    scale="30m",
                    variant=variant,
                    seed=seed,
                    config_path="configs/fineweb_30m_blessed.yaml",
                    run_name=f"fineweb_edu_small_{token}_ctx1024_steps2289_seed{seed}",
                    final_step=2289,
                )
            )
    for seed in (123, 1337):
        for variant, token in VARIANT_RUN_TOKEN.items():
            runs.append(
                BlessedRun(
                    scale="90m",
                    variant=variant,
                    seed=seed,
                    config_path="configs/fineweb_90m_offcurve.yaml",
                    run_name=f"fineweb_edu_medium_{token}_ctx1024_steps7010_seed{seed}",
                    final_step=7010,
                )
            )
    return runs


def _applescript_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def materialize_checkpoint(src: Path, scratch_dir: Path) -> Path:
    """Copy ``src`` into ``scratch_dir``, using Finder when direct I/O is blocked (DriveFS TCC)."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    src_resolved = src
    try:
        src_resolved = src.resolve()
    except OSError:
        src_resolved = src
    dest = scratch_dir / src_resolved.name
    try:
        if dest.exists() and dest.resolve() == src_resolved:
            return dest
    except OSError:
        pass
    if dest.exists():
        dest.unlink()

    # Prefer direct copy when the process can read Drive / local files.
    try:
        with src.open("rb") as handle:
            handle.read(1)
        shutil.copy2(src, dest)
        return dest
    except (PermissionError, OSError):
        pass

    # macOS Google Drive File Stream often allows Finder reads but not agent/TCC reads.
    script = (
        f"set srcPath to POSIX file {_applescript_string(str(src))}\n"
        f"set dstFolder to POSIX file {_applescript_string(str(scratch_dir))}\n"
        "tell application \"Finder\"\n"
        "  duplicate srcPath to dstFolder with replacing\n"
        "end tell\n"
    )
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not dest.exists():
        raise RuntimeError(
            f"Failed to materialize checkpoint {src} → {dest}\n"
            f"osascript stderr: {result.stderr.strip()}\n"
            "Grant Finder/Drive access or copy finals into --checkpoint-root manually."
        )
    return dest


def release_checkpoint(path: Path | None, device: torch.device) -> None:
    if path is not None and path.exists():
        path.unlink()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def resolve_final_checkpoint(checkpoint_root: Path, run: BlessedRun) -> Path:
    # Use the known final step path. Avoid directory listing — DriveFS TCC often
    # allows stat/exists on concrete paths but blocks listdir/glob.
    preferred = checkpoint_root / run.run_name / f"step_{run.final_step:07d}.pt"
    if preferred.exists():
        return preferred
    run_dir = checkpoint_root / run.run_name
    try:
        candidates = sorted(run_dir.glob("step_*.pt"))
    except OSError:
        candidates = []
    if not candidates:
        raise FileNotFoundError(
            f"Missing final checkpoint {preferred} (and no step_*.pt listing under {run_dir})"
        )
    return candidates[-1]


def extract_scores(eval_results: dict[str, Any], tasks: tuple[str, ...]) -> dict[str, float | None]:
    task_results = eval_results.get("results", {})
    scores: dict[str, float | None] = {}
    for task in tasks:
        if task not in task_results:
            scores[task] = None
            continue
        scores[task] = metric_value(task_results[task], task)
    return scores


def evaluate_checkpoint(
    *,
    run: BlessedRun,
    checkpoint_path: Path,
    device: torch.device,
    tasks: tuple[str, ...],
    batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    config = load_config(
        run.config_path,
        overrides=[f"experiment.seed={run.seed}", *VARIANT_OVERRIDES[run.variant]],
    )
    lm = AttnResGPTLM(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
        mixed_precision=False,
    )
    started = time.perf_counter()
    try:
        eval_results = evaluator.simple_evaluate(
            model=lm,
            tasks=list(tasks),
            batch_size=batch_size,
            device=str(device),
            limit=limit,
            log_samples=False,
        )
    finally:
        tokens_seen = lm.tokens_seen
        global_step = lm.global_step
        del lm
        gc.collect()
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    elapsed = time.perf_counter() - started
    scores = extract_scores(eval_results, tasks)
    return {
        "scale": run.scale,
        "variant": run.variant,
        "seed": run.seed,
        "run_name": run.run_name,
        "checkpoint": str(checkpoint_path),
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "elapsed_sec": elapsed,
        "scores": scores,
        "raw_results": eval_results.get("results", {}),
    }


def run_reorder_sanity_check(
    *,
    run: BlessedRun,
    checkpoint_path: Path,
    device: torch.device,
    tasks: tuple[str, ...],
    batch_size: int,
    limit: int,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Confirm joint multi-task eval matches per-task eval (Collator reorder fix)."""
    print(
        f"\n=== reorder sanity check ===\n"
        f"run={run.label}  tasks={list(tasks)}  limit={limit}  device={device}",
        flush=True,
    )
    joint = evaluate_checkpoint(
        run=run,
        checkpoint_path=checkpoint_path,
        device=device,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
    )
    singles: dict[str, float | None] = {}
    for task in tasks:
        one = evaluate_checkpoint(
            run=run,
            checkpoint_path=checkpoint_path,
            device=device,
            tasks=(task,),
            batch_size=batch_size,
            limit=limit,
        )
        singles[task] = one["scores"][task]

    diffs: dict[str, float | None] = {}
    ok = True
    for task in tasks:
        a = joint["scores"][task]
        b = singles[task]
        if a is None or b is None:
            diffs[task] = None
            ok = False
            continue
        diff = abs(float(a) - float(b))
        diffs[task] = diff
        status = "OK" if diff <= atol else "MISMATCH"
        if diff > atol:
            ok = False
        print(f"  {task:16s} joint={a:.6f}  single={b:.6f}  |Δ|={diff:.2e}  {status}", flush=True)

    return {
        "passed": ok,
        "atol": atol,
        "limit": limit,
        "tasks": list(tasks),
        "joint_scores": joint["scores"],
        "single_scores": singles,
        "abs_diffs": diffs,
        "run": run.label,
        "device": str(device),
    }


def aggregate_panel(rows: list[dict[str, Any]], tasks: tuple[str, ...]) -> dict[str, Any]:
    """Per scale / variant / task: per-seed scores + mean ± std."""
    summary: dict[str, Any] = {}
    scales = sorted({row["scale"] for row in rows})
    for scale in scales:
        scale_rows = [row for row in rows if row["scale"] == scale]
        variants = sorted({row["variant"] for row in scale_rows}, key=lambda v: list(VARIANT_OVERRIDES).index(v))
        scale_block: dict[str, Any] = {"variants": {}, "tasks": {}}
        for variant in variants:
            v_rows = sorted(
                (row for row in scale_rows if row["variant"] == variant),
                key=lambda row: row["seed"],
            )
            scale_block["variants"][variant] = {
                "seeds": [row["seed"] for row in v_rows],
                "run_names": [row["run_name"] for row in v_rows],
            }
        for task in tasks:
            chance = float(TASK_SPECS[task]["chance"])
            task_block: dict[str, Any] = {
                "label": TASK_SPECS[task].get("label", task),
                "chance": chance,
                "metric": TASK_SPECS[task]["metric"],
                "variants": {},
            }
            for variant in variants:
                v_rows = sorted(
                    (row for row in scale_rows if row["variant"] == variant),
                    key=lambda row: row["seed"],
                )
                per_seed = {
                    str(row["seed"]): row["scores"].get(task) for row in v_rows
                }
                values = [float(v) for v in per_seed.values() if v is not None]
                mean = statistics.fmean(values) if values else None
                std = statistics.stdev(values) if len(values) >= 2 else (0.0 if values else None)
                task_block["variants"][variant] = {
                    "per_seed": per_seed,
                    "mean": mean,
                    "std": std,
                    "n": len(values),
                    "above_chance_mean": None if mean is None else mean > chance,
                }
            scale_block["tasks"][task] = task_block
        summary[scale] = scale_block
    return summary


def format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "n/a"
    if std is None:
        return f"{100 * mean:.1f}"
    return f"{100 * mean:.1f}±{100 * std:.1f}"


def render_markdown_tables(summary: dict[str, Any], tasks: tuple[str, ...]) -> str:
    lines: list[str] = ["# Blessed downstream benchmark panel", ""]
    for scale, scale_block in summary.items():
        variants = list(scale_block["variants"])
        lines.append(f"## {scale.upper()}")
        lines.append("")
        header = (
            "| Benchmark | Chance |"
            + "".join(f" {v} mean±std |" for v in variants)
            + "".join(
                f" {v} s{seed} |"
                for v in variants
                for seed in scale_block["variants"][v]["seeds"]
            )
        )
        sep = (
            "|---|---:|"
            + "".join("---:|" for _ in variants)
            + "".join(
                "---:|"
                for v in variants
                for _ in scale_block["variants"][v]["seeds"]
            )
        )
        lines.append(header)
        lines.append(sep)
        for task in tasks:
            task_block = scale_block["tasks"][task]
            cells = [
                task_block["label"],
                f"{100 * task_block['chance']:.0f}%" if task_block["chance"] > 0 else "—",
            ]
            for variant in variants:
                stats = task_block["variants"][variant]
                cells.append(format_mean_std(stats["mean"], stats["std"]))
            for variant in variants:
                stats = task_block["variants"][variant]
                for seed in scale_block["variants"][variant]["seeds"]:
                    val = stats["per_seed"].get(str(seed))
                    cells.append("n/a" if val is None else f"{100 * float(val):.1f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_csv_long(path: Path, rows: list[dict[str, Any]], tasks: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scale",
        "variant",
        "seed",
        "run_name",
        "global_step",
        "tokens_seen",
        "elapsed_sec",
        "task",
        "metric",
        "score",
        "chance",
        "above_chance",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for task in tasks:
                score = row["scores"].get(task)
                chance = float(TASK_SPECS[task]["chance"])
                writer.writerow(
                    {
                        "scale": row["scale"],
                        "variant": row["variant"],
                        "seed": row["seed"],
                        "run_name": row["run_name"],
                        "global_step": row["global_step"],
                        "tokens_seen": row["tokens_seen"],
                        "elapsed_sec": row["elapsed_sec"],
                        "task": task,
                        "metric": TASK_SPECS[task]["metric"],
                        "score": score,
                        "chance": chance,
                        "above_chance": None if score is None else float(score) > chance,
                    }
                )


def select_runs(args: argparse.Namespace) -> list[BlessedRun]:
    runs = blessed_inventory()
    if args.scales:
        normalized: set[str] = set()
        for item in args.scales:
            key = item.lower()
            if key in {"30", "30m"}:
                normalized.add("30m")
            elif key in {"90", "90m"}:
                normalized.add("90m")
            else:
                normalized.add(key)
        runs = [run for run in runs if run.scale in normalized]
    if args.variants:
        runs = [run for run in runs if run.variant in set(args.variants)]
    if args.seeds:
        seeds = {int(s) for s in args.seeds}
        runs = [run for run in runs if run.seed in seeds]
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path(os.environ.get("BLESSED_CHECKPOINT_ROOT", DEFAULT_DRIVE_ROOT / "checkpoints")),
        help="Directory containing per-run checkpoint folders (Drive or local mirror).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "benchmark_scratch",
        help="Local dir for one-at-a-time checkpoint materialization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "benchmark_panel",
        help="Where to write JSON / CSV / markdown results.",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per task (smoke).")
    parser.add_argument("--tasks", nargs="+", default=list(PANEL_TASKS))
    parser.add_argument("--scales", nargs="+", default=None, help="Subset: 30m 90m")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=list(VARIANT_OVERRIDES),
    )
    parser.add_argument("--seeds", nargs="+", default=None)
    parser.add_argument(
        "--sanity-check-only",
        action="store_true",
        help="Only run multi-task vs single-task reorder sanity check, then exit.",
    )
    parser.add_argument(
        "--sanity-tasks",
        nargs="+",
        default=["hellaswag", "arc_easy", "boolq"],
        help="Tasks used for the reorder sanity check.",
    )
    parser.add_argument(
        "--keep-scratch",
        action="store_true",
        help="Do not delete materialized checkpoints after each eval.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs already present in the latest partial results JSON.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tasks = tuple(args.tasks)
    unknown = [task for task in tasks if task not in TASK_SPECS]
    if unknown:
        raise ValueError(f"Unsupported tasks {unknown}. Known: {sorted(TASK_SPECS)}")

    device = get_device(args.device)
    print(f"device={device}  torch={torch.__version__}  mps={torch.backends.mps.is_available()}")
    print(f"checkpoint_root={args.checkpoint_root}")
    print(f"scratch_dir={args.scratch_dir}")

    runs = select_runs(args)
    if not runs:
        raise RuntimeError("No blessed runs selected.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "blessed_panel_partial.json"

    if args.sanity_check_only:
        run = runs[0]
        src = resolve_final_checkpoint(args.checkpoint_root, run)
        print(f"materializing {src} ...", flush=True)
        local = materialize_checkpoint(src, args.scratch_dir)
        try:
            report = run_reorder_sanity_check(
                run=run,
                checkpoint_path=local,
                device=device,
                tasks=tuple(args.sanity_tasks),
                batch_size=args.batch_size,
                limit=args.limit or 32,
            )
        finally:
            if not args.keep_scratch:
                release_checkpoint(local, device)
        out = output_dir / "reorder_sanity.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nsanity passed={report['passed']}  wrote {out}")
        if not report["passed"]:
            raise SystemExit(1)
        return

    completed: list[dict[str, Any]] = []
    if args.skip_existing and partial_path.exists():
        payload = json.loads(partial_path.read_text(encoding="utf-8"))
        completed = list(payload.get("rows", []))
        print(f"resuming with {len(completed)} completed rows from {partial_path}")

    done_keys = {(row["scale"], row["variant"], row["seed"]) for row in completed}
    suite_start = time.perf_counter()

    for index, run in enumerate(runs, start=1):
        key = (run.scale, run.variant, run.seed)
        if key in done_keys:
            print(f"[{index}/{len(runs)}] skip {run.label} (already done)", flush=True)
            continue

        print(f"\n[{index}/{len(runs)}] {run.label}  run_name={run.run_name}", flush=True)
        src = resolve_final_checkpoint(args.checkpoint_root, run)
        print(f"  source={src}", flush=True)
        local = materialize_checkpoint(src, args.scratch_dir)
        print(f"  local={local}  size_mb={local.stat().st_size / 1e6:.1f}", flush=True)
        try:
            row = evaluate_checkpoint(
                run=run,
                checkpoint_path=local,
                device=device,
                tasks=tasks,
                batch_size=args.batch_size,
                limit=args.limit,
            )
        finally:
            if not args.keep_scratch:
                release_checkpoint(local, device)
                print("  scratch released", flush=True)

        completed.append(row)
        for task in tasks:
            score = row["scores"].get(task)
            chance = float(TASK_SPECS[task]["chance"])
            score_s = "n/a" if score is None else f"{score:.4f}"
            flag = "" if score is None else ("YES" if score > chance else "no")
            print(f"  {task:16s} {score_s:>8s}  chance={chance:.2f}  above={flag}", flush=True)
        print(f"  elapsed={row['elapsed_sec'] / 60:.1f} min", flush=True)

        partial = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "tasks": list(tasks),
            "rows": completed,
        }
        partial_path.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")

    suite_hours = (time.perf_counter() - suite_start) / 3600
    summary = aggregate_panel(completed, tasks)
    markdown = render_markdown_tables(summary, tasks)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "checkpoint_root": str(args.checkpoint_root),
        "tasks": list(tasks),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "suite_wall_clock_hours": suite_hours,
        "rows": completed,
        "summary": summary,
        "markdown": markdown,
    }

    json_path = output_dir / f"blessed_panel_{stamp}.json"
    latest_json = output_dir / "blessed_panel_latest.json"
    csv_path = output_dir / f"blessed_panel_{stamp}.csv"
    latest_csv = output_dir / "blessed_panel_latest.csv"
    md_path = output_dir / f"blessed_panel_{stamp}.md"
    latest_md = output_dir / "blessed_panel_latest.md"

    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_csv_long(csv_path, completed, tasks)
    write_csv_long(latest_csv, completed, tasks)
    md_path.write_text(markdown + "\n", encoding="utf-8")
    latest_md.write_text(markdown + "\n", encoding="utf-8")

    print("\n" + markdown)
    print(f"\ndevice={device}  suite_wall_clock_hours={suite_hours:.2f}")
    print(f"Wrote {latest_json}")
    print(f"Wrote {latest_csv}")
    print(f"Wrote {latest_md}")


if __name__ == "__main__":
    main()
