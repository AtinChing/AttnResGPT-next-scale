from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from src.models.vision_attnres import VisionConfig
from src.utils.config import AttnResConfig, ModelConfig
from src.vlm.clevr.official import CLEVR_SUBSETS
from src.vlm.clevr.preprocess import PreprocessConfig

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
STANDARD_VARIANTS = PRIMARY_VARIANTS + BLOCK_VARIANTS

BenchmarkName = Literal["clevr_v1", "clevr_cogent_v1"]
BenchmarkMode = Literal["smoke", "quick", "full"]


@dataclass
class AblationExperimentConfig:
    # Benchmark selection
    benchmark: BenchmarkName = "clevr_v1"
    benchmark_mode: BenchmarkMode = "quick"
    run_mode: BenchmarkMode = "quick"  # alias kept for notebook compatibility
    run_standard_clevr: bool = True
    run_cogent: bool = True
    subset_seed: int = 17

    resume: bool = True
    force_restart: bool = False
    run_primary_full_grid: bool = True
    run_block_grid: bool = True
    run_primary_full_only: bool = False
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    primary_variants: list[str] = field(default_factory=lambda: list(PRIMARY_VARIANTS))
    block_variants: list[str] = field(default_factory=lambda: list(BLOCK_VARIANTS))

    batch_size: int = 16
    grad_accum_steps: int = 4
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
    # Predict answer at the <answer> marker (next-token), never at the answer token itself.
    answer_supervision: str = "at_answer_marker"
    run_controls: bool = True

    # Tiny controlled VLM defaults for CLEVR (128x128 / 16x16 -> 64 patches)
    image_size: int = 128
    patch_size: int = 16
    vision_d_model: int = 128
    vision_n_layers: int = 10
    vision_n_heads: int = 4
    vision_d_ff: int = 512
    decoder_d_model: int = 128
    decoder_n_layers: int = 10
    decoder_n_heads: int = 4
    decoder_d_ff: int = 512
    dropout: float = 0.0
    num_blocks: int = 5
    max_seq_len: int = 160
    text_context_budget: int = 96

    # Filled after dataset preparation / identity hashing
    project_root: str = ""
    subset_manifest_hash: str = ""
    vocab_hash: str = ""
    preprocess_hash: str = ""
    dataset_version: str = "CLEVR_v1.0"
    train_size: int = 0
    validation_size: int = 0
    test_size: int = 0

    wandb_enabled: bool = True
    wandb_project: str = "attnres-next-scale-vlm"
    wandb_entity: str = "atin5551-uc-davis"
    wandb_mode: str = "auto"
    wandb_resume: str = "allow"
    wandb_tags: list[str] = field(default_factory=list)
    wandb_log_interval: int = 1

    def requested_variants(self) -> list[str]:
        if self.run_primary_full_only:
            return list(self.primary_variants)
        variants: list[str] = []
        if self.run_primary_full_grid:
            variants.extend(self.primary_variants)
        if self.run_block_grid:
            variants.extend(self.block_variants)
        return variants

    def preprocess_config(self) -> PreprocessConfig:
        return PreprocessConfig(image_size=self.image_size)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_experiment_config(config: AblationExperimentConfig) -> AblationExperimentConfig:
    # Keep run_mode / benchmark_mode synchronized.
    mode = config.benchmark_mode or config.run_mode
    config.benchmark_mode = mode
    config.run_mode = mode
    if config.run_primary_full_only:
        config.run_block_grid = False
        config.run_primary_full_grid = True
    sizes = CLEVR_SUBSETS[mode]
    config.train_size = sizes["train_images"]
    config.validation_size = sizes["validation_images"]
    config.test_size = sizes["test_images"]
    if mode == "smoke":
        config.batch_size = min(config.batch_size, 8)
        config.max_epochs = min(config.max_epochs, 3)
        config.num_workers = 0
        config.grad_accum_steps = max(1, min(config.grad_accum_steps, 2))
    patches_per_side = config.image_size // config.patch_size
    num_patches = patches_per_side * patches_per_side
    config.text_context_budget = max(config.text_context_budget, 96)
    minimum_seq_len = num_patches + config.text_context_budget
    if config.max_seq_len < minimum_seq_len:
        config.max_seq_len = minimum_seq_len
    if config.benchmark == "clevr_v1":
        config.dataset_version = "CLEVR_v1.0"
    else:
        config.dataset_version = "CLEVR_CoGenT_v1.0"
    preprocess = config.preprocess_config()
    config.preprocess_hash = preprocess.config_hash()
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
        "run_controls",
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


def run_dir_for(project_root: Path, benchmark: str, variant: str, seed: int, cfg_hash: str) -> Path:
    return Path(project_root) / "runs" / benchmark / variant / f"seed_{seed}" / cfg_hash
