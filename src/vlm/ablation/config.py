from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.utils.config import AttnResConfig, ModelConfig
from src.models.vision_attnres import VisionConfig

VARIANTS: dict[str, dict[str, str]] = {
    "baseline": {"encoder": "baseline", "decoder": "baseline"},
    "encoder_full": {"encoder": "attnres", "decoder": "baseline"},
    "decoder_full": {"encoder": "baseline", "decoder": "attnres"},
    "both_full": {"encoder": "attnres", "decoder": "attnres"},
    "encoder_block": {"encoder": "block_attnres", "decoder": "baseline"},
    "decoder_block": {"encoder": "baseline", "decoder": "block_attnres"},
    "both_block": {"encoder": "block_attnres", "decoder": "block_attnres"},
}

PRIMARY_VARIANTS = ["baseline", "encoder_full", "decoder_full", "both_full"]
BLOCK_VARIANTS = ["encoder_block", "decoder_block", "both_block"]

SMOKE_SIZES = {"train": 128, "validation": 64, "test": 64}
QUICK_SIZES = {"train": 4_000, "validation": 800, "test": 800}
FULL_SIZES = {"train": 12_000, "validation": 2_000, "test": 2_000}


@dataclass
class AblationExperimentConfig:
    run_mode: Literal["smoke", "quick", "full"] = "quick"
    resume: bool = True
    force_restart: bool = False
    run_primary_full_grid: bool = True
    run_block_grid: bool = True
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    primary_variants: list[str] = field(default_factory=lambda: list(PRIMARY_VARIANTS))
    block_variants: list[str] = field(default_factory=lambda: list(BLOCK_VARIANTS))

    batch_size: int = 64
    num_workers: int = 2
    checkpoint_interval: int = 100
    evaluation_interval: int = 1
    max_epochs: int = 15
    early_stopping_patience: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    mixed_precision: bool = True
    amp_dtype: str = "auto"
    supervise_eos: bool = True

    image_size: int = 64
    patch_size: int = 8
    vision_d_model: int = 128
    vision_n_layers: int = 6
    vision_n_heads: int = 4
    vision_d_ff: int = 512
    decoder_d_model: int = 128
    decoder_n_layers: int = 6
    decoder_n_heads: int = 4
    decoder_d_ff: int = 512
    dropout: float = 0.0
    num_blocks: int = 3
    max_seq_len: int = 64
    dataset_seed_offset: int = 17

    project_root: str = ""
    train_size: int = 0
    validation_size: int = 0
    test_size: int = 0

    # W&B (mirrors other notebooks; project is VLM-specific)
    wandb_enabled: bool = True
    wandb_project: str = "attnres-next-scale-vlm"
    wandb_entity: str = "atin5551-uc-davis"
    wandb_mode: str = "auto"  # auto | online | offline | disabled
    wandb_resume: str = "allow"  # allow | must | never
    wandb_tags: list[str] = field(default_factory=list)
    wandb_log_interval: int = 1  # log train loss every N optimizer steps (1 = every step)

    def requested_variants(self) -> list[str]:
        variants: list[str] = []
        if self.run_primary_full_grid:
            variants.extend(self.primary_variants)
        if self.run_block_grid:
            variants.extend(self.block_variants)
        return variants

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dataset_sizes_for_mode(run_mode: str) -> dict[str, int]:
    if run_mode == "smoke":
        return dict(SMOKE_SIZES)
    if run_mode == "quick":
        return dict(QUICK_SIZES)
    if run_mode == "full":
        return dict(FULL_SIZES)
    raise ValueError(f"Unknown run_mode: {run_mode}")


def resolve_experiment_config(config: AblationExperimentConfig) -> AblationExperimentConfig:
    sizes = dataset_sizes_for_mode(config.run_mode)
    config.train_size = sizes["train"]
    config.validation_size = sizes["validation"]
    config.test_size = sizes["test"]
    if config.run_mode == "smoke":
        config.batch_size = min(config.batch_size, 16)
        config.max_epochs = min(config.max_epochs, 3)
        config.num_workers = 0
    return config


def canonical_config_payload(config: AblationExperimentConfig) -> dict[str, Any]:
    payload = config.to_dict()
    for key in (
        "project_root",
        "resume",
        "force_restart",
        "wandb_enabled",
        "wandb_project",
        "wandb_entity",
        "wandb_mode",
        "wandb_resume",
        "wandb_tags",
        "wandb_log_interval",
    ):
        payload.pop(key, None)
    return payload


def config_hash(config: AblationExperimentConfig) -> str:
    serialized = json.dumps(canonical_config_payload(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def build_vision_config(config: AblationExperimentConfig, residual: str) -> VisionConfig:
    attnres = AttnResConfig(
        enabled=residual != "baseline",
        mode="block" if residual == "block_attnres" else "full",
        num_blocks=config.num_blocks if residual == "block_attnres" else None,
        final_readout=True,
        zero_init_queries=True,
        rmsnorm_keys=True,
        include_embedding=True,
    )
    return VisionConfig(
        image_size=config.image_size,
        patch_size=config.patch_size,
        d_model=config.vision_d_model,
        n_layers=config.vision_n_layers,
        n_heads=config.vision_n_heads,
        d_ff=config.vision_d_ff,
        dropout=config.dropout,
        residual=residual,  # type: ignore[arg-type]
        attnres=attnres,
    )


def build_decoder_config(config: AblationExperimentConfig, residual: str, vocab_size: int) -> ModelConfig:
    attnres = AttnResConfig(
        enabled=residual != "baseline",
        mode="block" if residual == "block_attnres" else "full",
        num_blocks=config.num_blocks if residual == "block_attnres" else None,
        final_readout=True,
        zero_init_queries=True,
        rmsnorm_keys=True,
        include_embedding=True,
    )
    architecture = "baseline" if residual == "baseline" else residual
    return ModelConfig(
        architecture=architecture,
        size_name="small",
        vocab_size=vocab_size,
        max_seq_len=config.max_seq_len,
        d_model=config.decoder_d_model,
        n_layers=config.decoder_n_layers,
        n_heads=config.decoder_n_heads,
        d_ff=config.decoder_d_ff,
        dropout=config.dropout,
        tie_weights=False,
        attnres=attnres,
    )


def run_dir_for(project_root: Path, variant: str, seed: int, cfg_hash: str) -> Path:
    return project_root / "runs" / variant / f"seed_{seed}" / cfg_hash
