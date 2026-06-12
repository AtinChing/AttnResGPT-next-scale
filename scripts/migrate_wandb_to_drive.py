"""Download W&B runs, histories, and artifacts to local disk or Google Drive.

Designed to recover checkpoints for offline eval when W&B storage/upload is blocked.
Reads WANDB_API_KEY from repo-root .env (never printed).

Typical Colab usage (after mounting Drive):
  python scripts/migrate_wandb_to_drive.py \\
    --dest /content/drive/MyDrive/AttnResGPT-next-scale-artifacts

Checkpoints-only (fast path for notebook 6 / 8 eval):
  python scripts/migrate_wandb_to_drive.py \\
    --dest /content/drive/MyDrive/AttnResGPT-next-scale-artifacts \\
    --checkpoints-only

Single run (e.g. missing VLM baseline seed 123):
  python scripts/migrate_wandb_to_drive.py \\
    --dest /content/drive/MyDrive/AttnResGPT-next-scale-artifacts \\
    --run-name vlm_baseline_flickr30k_b8_seed123 \\
    --checkpoints-only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_RE = re.compile(r"(?:^|/)step_(\d+)\.pt$")

DEFAULT_ENTITY = "atin5551-uc-davis"
DEFAULT_PROJECT = "attnres-next-scale"
def _artifact_entry_names(artifact: Any) -> list[str]:
    return sorted(artifact.manifest.entries.keys())


def _checkpoint_entries(artifact: Any) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for name in _artifact_entry_names(artifact):
        match = CHECKPOINT_RE.search(name)
        if match:
            entries.append((int(match.group(1)), name))
    return sorted(entries)


def find_logged_checkpoint_artifact(api: Any, *, run_path: str) -> Any:
    run = api.run(run_path)
    candidates: list[tuple[int, Any]] = []
    for artifact in run.logged_artifacts():
        checkpoints = _checkpoint_entries(artifact)
        if checkpoints:
            candidates.append((len(checkpoints), artifact))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-bearing logged artifacts found for {run_path}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def download_checkpoint_from_artifact(
    artifact: Any,
    *,
    step: int | None,
    target_dir: str | Path,
) -> tuple[Path, int]:
    checkpoints = _checkpoint_entries(artifact)
    if not checkpoints:
        raise FileNotFoundError(f"No step_XXXX.pt checkpoints found in artifact {artifact.name}")

    if step is None:
        resolved_step, entry_name = checkpoints[-1]
    else:
        matches = [item for item in checkpoints if item[0] == step]
        if not matches:
            available = [item[0] for item in checkpoints]
            raise FileNotFoundError(f"Checkpoint step {step} not found. Available steps: {available}")
        resolved_step, entry_name = matches[0]

    local_path = Path(artifact.get_path(entry_name).download(root=str(target_dir)))
    return local_path, resolved_step


def load_env() -> None:
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if line.strip() and not line.strip().startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


@dataclass
class RunReport:
    run_name: str
    run_id: str
    state: str
    checkpoint_path: str | None = None
    checkpoint_step: int | None = None
    artifacts_downloaded: list[str] = field(default_factory=list)
    history_rows: int | None = None
    errors: list[str] = field(default_factory=list)


def _preferred_checkpoint_step(run: Any) -> int | None:
    summary = dict(run.summary)
    for key in ("final_step", "_step"):
        value = summary.get(key)
        if value is not None:
            return int(value)
    return None


def _install_checkpoint(
    run: Any,
    *,
    checkpoints_dir: Path,
    prefer_step: int | None,
    force: bool,
) -> tuple[Path | None, int | None]:
    import wandb

    api = wandb.Api()
    run_path = f"{run.entity}/{run.project}/{run.id}"
    artifact = find_logged_checkpoint_artifact(api, run_path=run_path)
    run_ckpt_dir = checkpoints_dir / run.name
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if prefer_step is not None:
        target = run_ckpt_dir / f"step_{prefer_step:07d}.pt"
        if target.exists() and not force:
            return target, prefer_step
        try:
            path, resolved = download_checkpoint_from_artifact(
                artifact,
                step=prefer_step,
                target_dir=run_ckpt_dir,
            )
            final = run_ckpt_dir / f"step_{resolved:07d}.pt"
            if path.resolve() != final.resolve():
                shutil.copy2(path, final)
            return final, resolved
        except FileNotFoundError:
            pass

    path, resolved = download_checkpoint_from_artifact(
        artifact,
        step=None,
        target_dir=run_ckpt_dir,
    )
    final = run_ckpt_dir / f"step_{resolved:07d}.pt"
    if path.resolve() != final.resolve():
        shutil.copy2(path, final)
    return final, resolved


def _export_run_metadata(run: Any, export_dir: Path) -> int | None:
    run_dir = export_dir / "runs" / run.name
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {k: v for k, v in dict(run.summary).items() if not str(k).startswith("_")}
    config = {k: v for k, v in dict(run.config).items() if not str(k).startswith("_")}
    (run_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(_jsonable(config), indent=2), encoding="utf-8")
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "url": run.url,
                "created_at": str(run.created_at),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        history = run.history(samples=10_000_000)
        if history is not None and len(history) > 0:
            try:
                history.to_parquet(run_dir / "history.parquet", index=False)
            except Exception:
                history.to_csv(run_dir / "history.csv", index=False)
            return len(history)
    except Exception as exc:  # noqa: BLE001
        (run_dir / "history.error.txt").write_text(str(exc), encoding="utf-8")
    return None


def _download_all_artifacts(run: Any, artifacts_dir: Path, *, force: bool) -> list[str]:
    downloaded: list[str] = []
    for artifact in run.logged_artifacts():
        safe_name = artifact.name.replace(":", "_")
        target = artifacts_dir / run.name / safe_name
        if target.exists() and any(target.iterdir()) and not force:
            downloaded.append(f"{artifact.name} (skipped, exists)")
            continue
        target.mkdir(parents=True, exist_ok=True)
        artifact.download(root=str(target))
        downloaded.append(artifact.name)
    return downloaded


def migrate_run(
    run: Any,
    *,
    dest: Path,
    checkpoints_only: bool,
    force: bool,
) -> RunReport:
    report = RunReport(run_name=run.name, run_id=run.id, state=run.state)
    checkpoints_dir = dest / "checkpoints"
    export_dir = dest / "wandb_export"

    try:
        ckpt_path, ckpt_step = _install_checkpoint(
            run,
            checkpoints_dir=checkpoints_dir,
            prefer_step=_preferred_checkpoint_step(run),
            force=force,
        )
        report.checkpoint_path = str(ckpt_path) if ckpt_path else None
        report.checkpoint_step = ckpt_step
    except FileNotFoundError as exc:
        report.errors.append(f"checkpoint: {exc}")
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"checkpoint: {type(exc).__name__}: {exc}")

    if checkpoints_only:
        return report

    try:
        report.history_rows = _export_run_metadata(run, export_dir)
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"metadata: {type(exc).__name__}: {exc}")

    try:
        report.artifacts_downloaded = _download_all_artifacts(
            run,
            export_dir / "artifacts",
            force=force,
        )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"artifacts: {type(exc).__name__}: {exc}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate W&B project data to Google Drive or local disk.")
    parser.add_argument(
        "--dest",
        required=True,
        help="Artifact root (e.g. /content/drive/MyDrive/AttnResGPT-next-scale-artifacts)",
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--run-name", default=None, help="Only migrate runs whose name matches exactly.")
    parser.add_argument(
        "--state",
        default="finished",
        help="Comma-separated run states to include (default: finished). Use 'any' for all.",
    )
    parser.add_argument(
        "--checkpoints-only",
        action="store_true",
        help="Only pull step_*.pt files into checkpoints/{run_name}/ (fast eval recovery).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist.")
    parser.add_argument("--dry-run", action="store_true", help="List runs only; do not download.")
    return parser.parse_args()


def main() -> None:
    load_env()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit(
            "WANDB_API_KEY not set. Add it to .env or export it before running this script."
        )

    args = parse_args()
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    import wandb

    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    api = wandb.Api(timeout=120)
    project_path = f"{args.entity}/{args.project}"

    runs = list(api.runs(project_path, per_page=200))
    if args.run_name:
        runs = [run for run in runs if run.name == args.run_name]
    if args.state != "any":
        allowed = {state.strip() for state in args.state.split(",")}
        runs = [run for run in runs if run.state in allowed]

    print(f"Project: {project_path}")
    print(f"Destination: {dest}")
    print(f"Runs matched: {len(runs)}")
    if args.dry_run:
        for run in runs:
            print(f"  {run.state:9} {run.name}")
        return

    reports: list[RunReport] = []
    for index, run in enumerate(runs, start=1):
        print(f"\n[{index}/{len(runs)}] {run.name} ({run.state})")
        report = migrate_run(
            run,
            dest=dest,
            checkpoints_only=args.checkpoints_only,
            force=args.force,
        )
        reports.append(report)
        if report.checkpoint_path:
            print(f"  checkpoint: {report.checkpoint_path}")
        if report.history_rows is not None:
            print(f"  history rows: {report.history_rows}")
        if report.artifacts_downloaded:
            print(f"  artifacts: {len(report.artifacts_downloaded)}")
        for error in report.errors:
            print(f"  ERROR: {error}")

    manifest = {
        "project": project_path,
        "destination": str(dest),
        "checkpoints_only": args.checkpoints_only,
        "runs": [asdict(report) for report in reports],
    }
    manifest_path = dest / "wandb_export" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path}")

    ok_ckpt = sum(1 for r in reports if r.checkpoint_path)
    failed = [r.run_name for r in reports if r.errors]
    print(f"Checkpoints recovered: {ok_ckpt}/{len(reports)}")
    if failed:
        print("Runs with errors:", ", ".join(failed))


if __name__ == "__main__":
    main()
