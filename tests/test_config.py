from __future__ import annotations

import pytest

from src.utils.config import Config, DataConfig, validate_config


def test_dataset_caps_are_mutually_exclusive() -> None:
    config = Config(
        data=DataConfig(
            max_train_examples=100,
            max_train_tokens=1000,
        )
    )
    with pytest.raises(ValueError):
        validate_config(config)


def test_attnres_architecture_enables_attnres_flag() -> None:
    config = Config()
    config.model.architecture = "attnres"
    validated = validate_config(config)
    assert validated.model.attnres.enabled is True


def test_positionwise_steps_must_fit_within_training_budget() -> None:
    config = Config()
    config.training.max_steps = 100
    config.evaluation.positionwise_steps = [50, 150]
    with pytest.raises(ValueError):
        validate_config(config)


def test_console_step_tracking_defaults_off() -> None:
    config = validate_config(Config())
    assert config.training.console_step_tracking is False
    assert config.training.console_step_interval == 1


def test_console_step_interval_must_be_positive() -> None:
    config = Config()
    config.training.console_step_interval = 0
    with pytest.raises(ValueError):
        validate_config(config)
