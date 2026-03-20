from __future__ import annotations

import json
from pathlib import Path

from experiments.context_experiment import _prepare_context_config
from experiments.scale_experiment import (
    _contexts_for_config,
    _fallback_batching,
    _large_comparison_rows,
    _large_summary_rows,
    _prepare_run_config,
    _resolve_batching,
)
from src.utils.config import load_config
from src.utils.logging import ExperimentLogger, build_run_identity, create_run_paths


def test_large_config_loads_and_validates() -> None:
    config = load_config('configs/large.yaml')
    assert config.model.size_name == 'large'
    assert config.model.n_layers == 12
    assert config.model.d_model == 640
    assert config.model.n_heads == 10
    assert config.model.d_ff == 2560
    assert config.data.context_lengths == [256, 512]
    assert config.batching['ctx256']['batch_size'] == 2
    assert config.batching['ctx256']['grad_accum_steps'] == 8
    assert config.batching['ctx512']['batch_size'] == 1
    assert config.batching['ctx512']['grad_accum_steps'] == 16


def test_large_runner_uses_only_256_and_512_contexts() -> None:
    config = load_config('configs/large.yaml')
    contexts = _contexts_for_config(config)
    assert contexts == [256, 512]

    baseline_256 = _prepare_run_config(config, model_type='baseline', context=256)
    attnres_512 = _prepare_run_config(config, model_type='attnres', context=512)

    assert baseline_256.model.architecture == 'baseline'
    assert baseline_256.data.block_size == 256
    assert baseline_256.training.grad_accum_steps == 8
    assert baseline_256.data.batch_size == 2
    assert attnres_512.model.architecture == 'attnres'
    assert attnres_512.data.block_size == 512
    assert attnres_512.training.grad_accum_steps == 16
    assert attnres_512.data.batch_size == 1


def test_large_context_batching_and_oom_fallback() -> None:
    config = load_config('configs/large.yaml')

    batching_256 = _resolve_batching(config, 256)
    batching_512 = _resolve_batching(config, 512)
    assert batching_256 == {'batch_size': 2, 'grad_accum_steps': 8}
    assert batching_512 == {'batch_size': 1, 'grad_accum_steps': 16}

    prepared_512 = _prepare_context_config(config, model_type='baseline', context=512)
    fallback = _fallback_batching(prepared_512)
    assert prepared_512.data.batch_size == 1
    assert prepared_512.training.grad_accum_steps == 16
    assert fallback == {'batch_size': 1, 'grad_accum_steps': 32}


def test_large_summary_row_helpers_match_required_columns() -> None:
    summary_rows = [
        {
            'model': 'baseline',
            'context': 256,
            'val_loss': 3.1,
            'perplexity': 22.0,
            'second_half_loss': 3.3,
            'mean_activation_norm_last_layer': 5.5,
            'mean_early_contribution': None,
            'mean_late_contribution': None,
        },
        {
            'model': 'attnres',
            'context': 256,
            'val_loss': 3.0,
            'perplexity': 20.0,
            'second_half_loss': 3.1,
            'mean_activation_norm_last_layer': 5.2,
            'mean_early_contribution': 0.36,
            'mean_late_contribution': 0.48,
        },
    ]
    paired_rows = [
        {
            'context': 256,
            'baseline_val_loss': 3.1,
            'attnres_val_loss': 3.0,
            'delta_val_loss': 0.1,
            'baseline_ppl': 22.0,
            'attnres_ppl': 20.0,
            'delta_ppl': 2.0,
        }
    ]

    summary_payload = _large_summary_rows(summary_rows)
    comparison_payload = _large_comparison_rows(paired_rows)

    assert list(summary_payload[0].keys()) == [
        'model',
        'context',
        'val_loss',
        'perplexity',
        'second_half_loss',
        'mean_activation_norm_last_layer',
        'mean_early_contribution',
        'mean_late_contribution',
    ]
    assert list(comparison_payload[0].keys()) == [
        'context',
        'baseline_val_loss',
        'attnres_val_loss',
        'delta_val_loss',
        'baseline_ppl',
        'attnres_ppl',
        'delta_ppl',
    ]


def test_logger_mirrors_train_and_val_logs_to_outputs_logs(tmp_path: Path) -> None:
    config = load_config('configs/first_run.yaml')
    config.logging.output_root = str(tmp_path)
    identity = build_run_identity(config)
    paths = create_run_paths(config.logging.output_root, identity)
    logger = ExperimentLogger(paths)

    logger.log_train({'step': 1, 'train_loss': 1.23})
    logger.log_val({'step': 1, 'val_loss': 1.11})

    assert paths.train_log_path.exists()
    assert paths.val_log_path.exists()
    assert paths.global_train_log_path.exists()
    assert paths.global_val_log_path.exists()

    train_rows = paths.global_train_log_path.read_text(encoding='utf-8').strip().splitlines()
    val_rows = paths.global_val_log_path.read_text(encoding='utf-8').strip().splitlines()
    assert json.loads(train_rows[-1])['train_loss'] == 1.23
    assert json.loads(val_rows[-1])['val_loss'] == 1.11


def test_notebooks_are_local_only_and_expose_large_config() -> None:
    train_notebook = Path('notebooks/1_train_scale.ipynb').read_text(encoding='utf-8')
    full_notebook = Path('notebooks/0_full_pipeline.ipynb').read_text(encoding='utf-8')
    analysis_notebook = Path('notebooks/2_analyze_results.ipynb').read_text(encoding='utf-8')

    assert 'google.colab' not in train_notebook
    assert 'drive.mount' not in train_notebook
    assert 'run_scale(' in train_notebook
    assert 'configs/large.yaml' in train_notebook
    assert 'google.colab' not in full_notebook
    assert 'drive.mount' not in full_notebook
    assert 'run_scale(' in full_notebook
    assert 'configs/large.yaml' in full_notebook
    assert 'google.colab' not in analysis_notebook
    assert 'drive.mount' not in analysis_notebook
