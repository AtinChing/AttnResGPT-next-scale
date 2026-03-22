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
