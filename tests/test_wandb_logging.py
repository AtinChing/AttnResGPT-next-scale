from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.utils.config import Config, load_config_from_dict
from src.utils.logging import (
    ExperimentLogger,
    _resolve_wandb_mode,
    build_run_identity,
    canonical_config_dict,
    create_run_paths,
)


class FakeRun:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logged: list[tuple[int | None, dict[str, object]]] = []
        self.summary: dict[str, object] = {}
        self.id = kwargs.get('id')
        self.name = kwargs.get('name')
        self.finished = False

    def log(self, payload, step=None):
        self.logged.append((step, dict(payload)))

    def get_url(self):
        return f"https://wandb.local/{self.id}"

    def finish(self, *args, **kwargs):
        self.finished = True


def _fake_wandb_module() -> ModuleType:
    module = ModuleType('wandb')
    module.runs = []

    def init(**kwargs):
        run = FakeRun(**kwargs)
        module.runs.append(run)
        return run

    module.init = init
    return module


def test_wandb_defaults_are_enabled_and_excluded_from_config_hash() -> None:
    config = Config()
    enabled, mode = _resolve_wandb_mode(config.logging.wandb)
    canonical = canonical_config_dict(config)

    assert enabled is True
    assert mode in {'offline', 'online'}
    assert 'wandb' not in canonical['logging']


def test_invalid_wandb_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        load_config_from_dict({'logging': {'wandb': {'mode': 'banana'}}})


def test_experiment_logger_logs_to_wandb_when_available(monkeypatch, tmp_path: Path) -> None:
    fake_wandb = _fake_wandb_module()
    monkeypatch.setitem(sys.modules, 'wandb', fake_wandb)
    monkeypatch.delenv('WANDB_API_KEY', raising=False)
    monkeypatch.delenv('WANDB_MODE', raising=False)
    monkeypatch.delenv('WANDB_DISABLED', raising=False)

    config = Config()
    config.logging.output_root = str(tmp_path)
    identity = build_run_identity(config)
    paths = create_run_paths(config.logging.output_root, identity)
    logger = ExperimentLogger(paths, config=config, identity=identity)

    metadata = logger.wandb_metadata()
    assert metadata['wandb_enabled'] is True
    assert metadata['wandb_mode'] == 'offline'
    assert fake_wandb.runs[0].kwargs['project'] == 'attnres-next-scale'

    logger.log_train({'step': 1, 'train_loss': 1.23, 'activation_norms': {'layer0': 0.4}, 'ignored': [1, 2]})
    logger.log_val({'step': 1, 'val_loss': 1.11, 'perplexity': 3.0})
    logger.save_summary(
        {
            'run_name': identity.run_name,
            'val_loss': 1.11,
            'checkpoint_path': '/tmp/checkpoint.pt',
            'mean_early_contribution': 0.33,
            'depth_attention_rows': [[0.1, 0.9]],
            'last_gradient_norms': {'layer0': 0.2},
        }
    )
    logger.close(status='completed')

    run = fake_wandb.runs[0]
    assert run.logged[0][0] == 1
    assert run.logged[0][1]['train_loss'] == 1.23
    assert run.logged[0][1]['activation_norms/layer0'] == 0.4
    assert 'ignored' not in run.logged[0][1]
    assert run.summary['checkpoint_path'] == '/tmp/checkpoint.pt'
    assert run.summary['last_gradient_norms/layer0'] == 0.2
    assert 'depth_attention_rows' not in run.summary
    assert run.finished is True
