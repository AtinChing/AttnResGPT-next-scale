from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class AttnResConfig:
    enabled: bool = False
    window_size: Optional[int] = None
    temperature: float = 1.0
    rmsnorm_keys: bool = True
    zero_init_queries: bool = True
    include_embedding: bool = True
    keep_embedding_in_window: bool = True
    final_readout: bool = True


@dataclass
class ModelConfig:
    architecture: str = "baseline"
    size_name: str = "small"
    vocab_size: int = 0
    max_seq_len: int = 128
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True
    norm_eps: float = 1e-5
    init_std: float = 0.02
    attnres: AttnResConfig = field(default_factory=AttnResConfig)


@dataclass
class DataConfig:
    dataset_type: str = "tinystories"
    dataset_name: str = "tinystories"
    tokenizer_name: str = "gpt2"
    text_path: Optional[str] = None
    train_text_path: Optional[str] = None
    val_text_path: Optional[str] = None
    train_split: str = "train"
    val_split: str = "validation"
    block_size: int = 128
    batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 0
    pin_memory: bool = True
    max_train_examples: Optional[int] = None
    max_train_tokens: Optional[int] = 250_000
    max_val_examples: Optional[int] = None
    max_val_tokens: Optional[int] = 50_000


@dataclass
class TrainingConfig:
    max_steps: int = 300
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 30
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    mixed_precision: bool = True
    amp_dtype: str = "bfloat16"
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 100
    probe_interval: int = 100
    eval_max_batches: Optional[int] = 10
    device: str = "auto"
    resume_from: Optional[str] = None
    allow_resume_mismatch: bool = False


@dataclass
class LoggingConfig:
    output_root: str = "outputs"
    save_probes: bool = True
    save_checkpoints: bool = True
    keep_last_k_checkpoints: int = 2


@dataclass
class EvaluationConfig:
    max_batches: Optional[int] = 10


@dataclass
class ExperimentConfig:
    name: str = "first_run"
    stage: str = "first_run"
    seed: int = 1337
    deterministic: bool = False
    notes: str = ""


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _construct_config(values: dict[str, Any]) -> Config:
    return Config(
        experiment=ExperimentConfig(**values.get("experiment", {})),
        data=DataConfig(**values.get("data", {})),
        model=ModelConfig(
            **{
                **values.get("model", {}),
                "attnres": AttnResConfig(**values.get("model", {}).get("attnres", {})),
            }
        ),
        training=TrainingConfig(**values.get("training", {})),
        logging=LoggingConfig(**values.get("logging", {})),
        evaluation=EvaluationConfig(**values.get("evaluation", {})),
    )


def config_to_dict(config: Config) -> dict[str, Any]:
    return asdict(config)


def apply_overrides(config_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    for override in overrides:
        key, raw_value = override.split("=", maxsplit=1)
        cursor = config_dict
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    return config_dict


def _validate_cap_pair(name_a: str, value_a: Optional[int], name_b: str, value_b: Optional[int]) -> None:
    if value_a is not None and value_a <= 0:
        raise ValueError(f"{name_a} must be positive when set")
    if value_b is not None and value_b <= 0:
        raise ValueError(f"{name_b} must be positive when set")
    if value_a is not None and value_b is not None:
        raise ValueError(f"Set at most one of {name_a} and {name_b}")


def validate_config(config: Config) -> Config:
    if config.model.architecture not in {"baseline", "attnres"}:
        raise ValueError(f"Unsupported architecture: {config.model.architecture}")
    if config.model.size_name not in {"small", "medium"}:
        raise ValueError("model.size_name must be one of: small, medium")
    if config.model.d_model % config.model.n_heads != 0:
        raise ValueError("model.d_model must be divisible by model.n_heads")
    if config.data.block_size > config.model.max_seq_len:
        raise ValueError("data.block_size must be <= model.max_seq_len")
    if config.training.min_lr > config.training.learning_rate:
        raise ValueError("training.min_lr must be <= training.learning_rate")
    if config.data.dataset_type not in {"tinystories", "local_text"}:
        raise ValueError("data.dataset_type must be one of: tinystories, local_text")
    _validate_cap_pair(
        "data.max_train_examples",
        config.data.max_train_examples,
        "data.max_train_tokens",
        config.data.max_train_tokens,
    )
    _validate_cap_pair(
        "data.max_val_examples",
        config.data.max_val_examples,
        "data.max_val_tokens",
        config.data.max_val_tokens,
    )
    if config.model.architecture == "attnres":
        config.model.attnres.enabled = True
    return config


def load_config(path: str | Path, overrides: Optional[list[str]] = None) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if overrides:
        payload = apply_overrides(payload, overrides)
    return validate_config(_construct_config(payload))


def load_config_from_dict(payload: dict[str, Any]) -> Config:
    return validate_config(_construct_config(payload))


def save_config(config: Config, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)
