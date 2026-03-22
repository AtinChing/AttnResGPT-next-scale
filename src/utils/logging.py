from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.utils.config import Config, WandbConfig, config_to_dict, save_config
from src.utils.runtime import ensure_dir, sanitize_name


@dataclass
class RunIdentity:
    run_name: str
    config_hash: str


@dataclass
class RunPaths:
    run_dir: Path
    probe_dir: Path
    positionwise_dir: Path
    checkpoint_dir: Path
    train_log_path: Path
    val_log_path: Path
    global_train_log_path: Path
    global_val_log_path: Path
    summary_json_path: Path
    summary_csv_path: Path
    metadata_path: Path
    config_snapshot_path: Path
    config_hash_path: Path
    tokenizer_dir: Path


@dataclass
class WandbState:
    enabled: bool = False
    mode: str = 'disabled'
    project: str | None = None
    run_id: str | None = None
    run_name: str | None = None
    url: str | None = None
    error: str | None = None


def canonical_config_dict(config: Config) -> dict[str, Any]:
    payload = config_to_dict(config)
    payload['experiment'].pop('name', None)
    payload['experiment'].pop('stage', None)
    payload['experiment'].pop('notes', None)
    payload['training'].pop('resume_from', None)
    payload['training'].pop('allow_resume_mismatch', None)
    payload['logging'].pop('output_root', None)
    payload['logging'].pop('wandb', None)
    return payload


def config_hash(config: Config) -> str:
    serialized = json.dumps(canonical_config_dict(config), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]


def build_run_name(config: Config) -> str:
    dataset = sanitize_name(config.data.dataset_name)
    size = sanitize_name(config.model.size_name)
    model = sanitize_name(config.model.architecture)
    context = config.data.block_size
    steps = config.training.max_steps
    seed = config.experiment.seed
    return f'{dataset}_{size}_{model}_ctx{context}_steps{steps}_seed{seed}'


def build_run_identity(config: Config) -> RunIdentity:
    return RunIdentity(run_name=build_run_name(config), config_hash=config_hash(config))


def create_run_paths(output_root: str | Path, identity: RunIdentity) -> RunPaths:
    output_root = Path(output_root)
    logs_dir = ensure_dir(output_root / 'logs')
    run_dir = ensure_dir(output_root / 'runs' / identity.run_name)
    probe_dir = ensure_dir(run_dir / 'probes')
    positionwise_dir = ensure_dir(run_dir / 'positionwise')
    checkpoint_dir = ensure_dir(output_root / 'checkpoints' / identity.run_name)
    tokenizer_dir = ensure_dir(run_dir / 'tokenizer')
    return RunPaths(
        run_dir=run_dir,
        probe_dir=probe_dir,
        positionwise_dir=positionwise_dir,
        checkpoint_dir=checkpoint_dir,
        train_log_path=run_dir / 'train_metrics.jsonl',
        val_log_path=run_dir / 'val_metrics.jsonl',
        global_train_log_path=logs_dir / f'{identity.run_name}_train.jsonl',
        global_val_log_path=logs_dir / f'{identity.run_name}_val.jsonl',
        summary_json_path=run_dir / 'run_summary.json',
        summary_csv_path=run_dir / 'run_summary.csv',
        metadata_path=run_dir / 'run_metadata.json',
        config_snapshot_path=run_dir / 'config.snapshot.yaml',
        config_hash_path=run_dir / 'config.hash.txt',
        tokenizer_dir=tokenizer_dir,
    )


def write_run_snapshot(config: Config, identity: RunIdentity, paths: RunPaths, metadata: Mapping[str, Any]) -> None:
    save_config(config, paths.config_snapshot_path)
    paths.config_hash_path.write_text(identity.config_hash, encoding='utf-8')
    with paths.metadata_path.open('w', encoding='utf-8') as handle:
        json.dump(dict(metadata), handle, indent=2, sort_keys=True)


def _env_truthy(name: str) -> bool:
    return os.getenv(name, '').strip().lower() in {'1', 'true', 'yes', 'on'}


def _resolve_wandb_mode(wandb_config: WandbConfig) -> tuple[bool, str]:
    if not wandb_config.enabled:
        return False, 'disabled'

    env_mode = os.getenv('WANDB_MODE', '').strip().lower()
    if env_mode == 'disabled':
        return False, 'disabled'
    if env_mode in {'online', 'offline'}:
        return True, env_mode
    if _env_truthy('WANDB_DISABLED'):
        return False, 'disabled'

    if wandb_config.mode == 'disabled':
        return False, 'disabled'
    if wandb_config.mode in {'online', 'offline'}:
        return True, wandb_config.mode

    if os.getenv('WANDB_API_KEY'):
        return True, 'online'
    return True, 'offline'


def _flatten_wandb_payload(
    payload: Mapping[str, Any],
    *,
    prefix: str = '',
    allow_strings: bool = False,
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        name = f'{prefix}/{key}' if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_wandb_payload(value, prefix=name, allow_strings=allow_strings))
            continue
        if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, (int, float, bool)):
            flattened[name] = value
        elif allow_strings and isinstance(value, str):
            flattened[name] = value
    return flattened


def _build_wandb_tags(config: Config) -> list[str]:
    tags = list(config.logging.wandb.tags)
    tags.extend(
        [
            config.model.architecture,
            config.model.size_name,
            f'ctx{config.data.block_size}',
            config.experiment.stage,
        ]
    )
    seen: set[str] = set()
    unique: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


class OptionalWandbLogger:
    def __init__(
        self,
        *,
        config: Config | None = None,
        identity: RunIdentity | None = None,
        paths: RunPaths | None = None,
    ) -> None:
        self.state = WandbState()
        self._run = None
        if config is None or identity is None or paths is None:
            return
        self._initialize(config=config, identity=identity, paths=paths)

    def _start_run(self, wandb_module: Any, init_kwargs: dict[str, Any]) -> Any:
        cleaned = {key: value for key, value in init_kwargs.items() if value is not None}
        return wandb_module.init(**cleaned)

    def _initialize(self, *, config: Config, identity: RunIdentity, paths: RunPaths) -> None:
        enabled, mode = _resolve_wandb_mode(config.logging.wandb)
        if not enabled:
            self.state = WandbState(enabled=False, mode='disabled', project=config.logging.wandb.project)
            return

        try:
            import wandb  # type: ignore
        except Exception as error:
            self.state = WandbState(
                enabled=False,
                mode='disabled',
                project=config.logging.wandb.project,
                error=f'wandb import failed: {error}',
            )
            return

        run_name = config.logging.wandb.run_name or identity.run_name
        init_kwargs = {
            'project': config.logging.wandb.project,
            'entity': config.logging.wandb.entity,
            'name': run_name,
            'id': identity.run_name,
            'resume': 'allow',
            'mode': mode,
            'dir': str(paths.run_dir),
            'job_type': config.logging.wandb.job_type,
            'group': config.logging.wandb.group,
            'tags': _build_wandb_tags(config),
            'config': config_to_dict(config),
        }

        init_error: str | None = None
        try:
            run = self._start_run(wandb, init_kwargs)
        except Exception as error:
            init_error = str(error)
            if mode != 'online':
                self.state = WandbState(
                    enabled=False,
                    mode='disabled',
                    project=config.logging.wandb.project,
                    error=f'wandb init failed: {error}',
                )
                return
            init_kwargs['mode'] = 'offline'
            try:
                run = self._start_run(wandb, init_kwargs)
                mode = 'offline'
            except Exception as offline_error:
                self.state = WandbState(
                    enabled=False,
                    mode='disabled',
                    project=config.logging.wandb.project,
                    error=f'wandb init failed: {error}; offline fallback failed: {offline_error}',
                )
                return

        url = None
        try:
            if hasattr(run, 'get_url'):
                url = run.get_url()
        except Exception:
            url = None

        self._run = run
        self.state = WandbState(
            enabled=True,
            mode=mode,
            project=config.logging.wandb.project,
            run_id=getattr(run, 'id', identity.run_name),
            run_name=getattr(run, 'name', run_name),
            url=url,
            error=init_error,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            'wandb_enabled': self.state.enabled,
            'wandb_mode': self.state.mode,
            'wandb_project': self.state.project,
            'wandb_run_id': self.state.run_id,
            'wandb_run_name': self.state.run_name,
            'wandb_url': self.state.url,
            'wandb_error': self.state.error,
        }

    def log(self, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        if self._run is None:
            return
        metrics = _flatten_wandb_payload(payload)
        metrics.pop('step', None)
        if metrics:
            self._run.log(metrics, step=step)

    def update_summary(self, payload: Mapping[str, Any]) -> None:
        if self._run is None:
            return
        summary_payload = _flatten_wandb_payload(payload, allow_strings=True)
        for key, value in summary_payload.items():
            self._run.summary[key] = value

    def finish(self, *, status: str = 'completed') -> None:
        if self._run is None:
            return
        exit_code = 0 if status == 'completed' else 1
        try:
            self._run.finish(exit_code=exit_code)
        except TypeError:
            self._run.finish()
        except Exception:
            return


class ExperimentLogger:
    def __init__(
        self,
        paths: RunPaths,
        *,
        config: Config | None = None,
        identity: RunIdentity | None = None,
    ) -> None:
        self.paths = paths
        self._wandb = OptionalWandbLogger(config=config, identity=identity, paths=paths)

    def wandb_metadata(self) -> dict[str, Any]:
        return self._wandb.metadata()

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, sort_keys=True) + '\n')

    def log_train(self, payload: Mapping[str, Any]) -> None:
        self._append_jsonl(self.paths.train_log_path, payload)
        self._append_jsonl(self.paths.global_train_log_path, payload)
        self._wandb.log(payload, step=int(payload['step']) if 'step' in payload else None)

    def log_val(self, payload: Mapping[str, Any]) -> None:
        self._append_jsonl(self.paths.val_log_path, payload)
        self._append_jsonl(self.paths.global_val_log_path, payload)
        self._wandb.log(payload, step=int(payload['step']) if 'step' in payload else None)

    def save_probe(self, step: int, payload: Mapping[str, Any]) -> Path:
        probe_path = self.paths.probe_dir / f'step_{step:07d}.json'
        with probe_path.open('w', encoding='utf-8') as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
        return probe_path

    def save_positionwise(self, step: int, payload: Mapping[str, Any]) -> tuple[Path, Path]:
        json_path = self.paths.positionwise_dir / f'step_{step:07d}.json'
        csv_path = self.paths.positionwise_dir / f'step_{step:07d}.csv'

        with json_path.open('w', encoding='utf-8') as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)

        position_losses = payload.get('position_losses', [])
        rows = [
            {'position': position, 'loss': loss}
            for position, loss in enumerate(position_losses)
        ]
        if rows:
            write_csv_rows(csv_path, rows)
        return json_path, csv_path

    def log_positionwise(self, step: int, payload: Mapping[str, Any]) -> tuple[Path, Path]:
        json_path, csv_path = self.save_positionwise(step, payload)
        position_losses = payload.get('position_losses', [])
        wandb_payload: dict[str, Any] = {
            'step': step,
            'positionwise_eval': {
                key: value
                for key, value in payload.items()
                if key != 'position_losses'
            },
            'positionwise_loss': {
                f'pos_{position:04d}': float(loss)
                for position, loss in enumerate(position_losses)
            },
        }
        self._wandb.log(wandb_payload, step=step)
        return json_path, csv_path

    def save_summary(self, payload: Mapping[str, Any]) -> None:
        row = _flatten_summary_row(dict(payload))
        with self.paths.summary_json_path.open('w', encoding='utf-8') as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
        write_csv_rows(self.paths.summary_csv_path, [row])
        self._wandb.update_summary(payload)

    def prune_old_checkpoints(self, keep_last_k: int) -> None:
        if keep_last_k <= 0:
            return
        checkpoints = sorted(self.paths.checkpoint_dir.glob('step_*.pt'))
        for checkpoint in checkpoints[:-keep_last_k]:
            checkpoint.unlink(missing_ok=True)

    def close(self, *, status: str = 'completed') -> None:
        self._wandb.finish(status=status)


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        return
    ensure_dir(path.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def read_csv_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _flatten_summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        'run_name',
        'model',
        'size',
        'context',
        'dataset',
        'tokenizer_name',
        'seed',
        'val_loss',
        'perplexity',
        'second_half_loss',
        'mean_activation_norm_last_layer',
        'mean_early_contribution',
        'mean_late_contribution',
        'parameter_count_total',
        'config_hash',
        'checkpoint_path',
    )
    return {key: summary.get(key) for key in keys}


def _consolidated_summary_row(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        'model',
        'size',
        'context',
        'val_loss',
        'perplexity',
        'second_half_loss',
        'mean_activation_norm_last_layer',
        'mean_early_contribution',
        'mean_late_contribution',
    )
    return {key: summary.get(key) for key in keys}


def _merge_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in existing_rows:
        key = tuple(row.get(field) for field in key_fields)
        merged[key] = dict(row)
    for row in new_rows:
        key = tuple(row.get(field) for field in key_fields)
        merged[key] = dict(row)
    return [
        row
        for _key, row in sorted(
            merged.items(),
            key=lambda item: tuple(str(part) for part in item[0]),
        )
    ]


def write_global_summary_artifacts(
    output_root: str | Path,
    summary_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_root = Path(output_root)
    logs_dir = ensure_dir(output_root / 'logs')
    summary_path = logs_dir / 'run_summaries.csv'
    consolidated_path = logs_dir / 'consolidated_summary_table.csv'
    paired_path = logs_dir / 'paired_comparisons.csv'

    merged_summaries = _merge_rows(read_csv_rows(summary_path), summary_rows, key_fields=('run_name',))
    merged_consolidated = _merge_rows(
        read_csv_rows(consolidated_path),
        [_consolidated_summary_row(row) for row in summary_rows],
        key_fields=('model', 'size', 'context'),
    )
    merged_paired = _merge_rows(read_csv_rows(paired_path), paired_rows, key_fields=('size', 'context'))

    write_csv_rows(summary_path, merged_summaries)
    write_csv_rows(consolidated_path, merged_consolidated)
    write_csv_rows(paired_path, merged_paired)
