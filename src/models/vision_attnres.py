from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

from src.metrics.depth_metrics import contribution_breakdown
from src.models.attnres import DepthAttentionResidual, _block_sizes
from src.models.baseline import BidirectionalSelfAttention, FeedForward, RMSNorm
from src.utils.config import AttnResConfig


@dataclass
class VisionConfig:
    image_size: int = 64
    patch_size: int = 8
    in_channels: int = 3
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    bias: bool = True
    norm_eps: float = 1e-5
    init_std: float = 0.02
    residual: Literal["baseline", "attnres", "block_attnres"] = "baseline"
    attnres: AttnResConfig = field(default_factory=AttnResConfig)

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.residual == "block_attnres":
            self.attnres.enabled = True
            self.attnres.mode = "block"
            if self.attnres.num_blocks is None:
                self.attnres.num_blocks = max(1, self.n_layers // 2)
        elif self.residual == "attnres":
            self.attnres.enabled = True
            self.attnres.mode = "full"

    @property
    def num_patches(self) -> int:
        side = self.image_size // self.patch_size
        return side * side


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.num_patches = config.num_patches
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, config.d_model))
        self.dropout = nn.Dropout(config.dropout)
        nn.init.normal_(self.position_embedding, mean=0.0, std=config.init_std)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, D, H', W'] -> [B, N, D]
        x = self.proj(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        return self.dropout(x)


class BaselineVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = BidirectionalSelfAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            bias=config.bias,
        )
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = FeedForward(config.d_model, config.d_ff, config.dropout, bias=config.bias)

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        attn_delta, _ = self.attn(self.attn_norm(x))
        x = x + attn_delta
        mlp_delta = self.mlp(self.mlp_norm(x))
        x = x + mlp_delta
        return x, {}


class AttnResVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        attnres = config.attnres
        self.pre_attn_res = DepthAttentionResidual(
            config.d_model,
            temperature=attnres.temperature,
            window_size=attnres.window_size,
            rmsnorm_keys=attnres.rmsnorm_keys,
            zero_init_query=attnres.zero_init_queries,
            include_embedding=attnres.include_embedding,
            keep_embedding_in_window=attnres.keep_embedding_in_window,
            eps=config.norm_eps,
        )
        self.pre_mlp_res = DepthAttentionResidual(
            config.d_model,
            temperature=attnres.temperature,
            window_size=attnres.window_size,
            rmsnorm_keys=attnres.rmsnorm_keys,
            zero_init_query=attnres.zero_init_queries,
            include_embedding=attnres.include_embedding,
            keep_embedding_in_window=attnres.keep_embedding_in_window,
            eps=config.norm_eps,
        )
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = BidirectionalSelfAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            bias=config.bias,
        )
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = FeedForward(config.d_model, config.d_ff, config.dropout, bias=config.bias)

    def forward(
        self,
        history: list[torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        attn_input, attn_stats = self.pre_attn_res(history, return_stats=return_aux)
        attn_delta, _ = self.attn(self.attn_norm(attn_input))
        history.append(attn_delta)

        mlp_input, mlp_stats = self.pre_mlp_res(history, return_stats=return_aux)
        mlp_delta = self.mlp(self.mlp_norm(mlp_input))
        history.append(mlp_delta)

        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "depth_attention": [
                    {"name": f"block_{self.layer_index:02d}_pre_attn", **attn_stats},
                    {"name": f"block_{self.layer_index:02d}_pre_mlp", **mlp_stats},
                ],
            }
        return history, aux


class BlockAttnResVisionBlock(nn.Module):
    def __init__(self, config: VisionConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        attnres = config.attnres
        self.pre_attn_res = DepthAttentionResidual(
            config.d_model,
            temperature=attnres.temperature,
            window_size=None,
            rmsnorm_keys=attnres.rmsnorm_keys,
            zero_init_query=attnres.zero_init_queries,
            include_embedding=attnres.include_embedding,
            keep_embedding_in_window=attnres.keep_embedding_in_window,
            eps=config.norm_eps,
        )
        self.pre_mlp_res = DepthAttentionResidual(
            config.d_model,
            temperature=attnres.temperature,
            window_size=None,
            rmsnorm_keys=attnres.rmsnorm_keys,
            zero_init_query=attnres.zero_init_queries,
            include_embedding=attnres.include_embedding,
            keep_embedding_in_window=attnres.keep_embedding_in_window,
            eps=config.norm_eps,
        )
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = BidirectionalSelfAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            bias=config.bias,
        )
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = FeedForward(config.d_model, config.d_ff, config.dropout, bias=config.bias)

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial: torch.Tensor | None,
        *,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        sources = blocks if partial is None else [*blocks, partial]
        attn_input, attn_stats = self.pre_attn_res(sources, return_stats=return_aux)
        attn_delta, _ = self.attn(self.attn_norm(attn_input))
        partial = attn_delta if partial is None else partial + attn_delta

        mlp_input, mlp_stats = self.pre_mlp_res([*blocks, partial], return_stats=return_aux)
        mlp_delta = self.mlp(self.mlp_norm(mlp_input))
        partial = partial + mlp_delta

        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "depth_attention": [
                    {"name": f"block_{self.layer_index:02d}_pre_attn", **attn_stats},
                    {"name": f"block_{self.layer_index:02d}_pre_mlp", **mlp_stats},
                ],
            }
        return partial, aux


class BaselineVisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = VisionPatchEmbed(config)
        self.blocks = nn.ModuleList(
            [BaselineVisionBlock(config, layer_index=index) for index in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_weight_capture(self, enabled: bool) -> None:
        return None

    def iter_depth_residuals(self) -> list[DepthAttentionResidual]:
        return []

    def forward(self, pixel_values: torch.Tensor, *, return_aux: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        x = self.patch_embed(pixel_values)
        for block in self.blocks:
            x, _ = block(x, return_aux=return_aux)
        x = self.final_norm(x)
        aux: dict[str, Any] = {}
        if return_aux:
            aux = {"depth_attention_rows": [], "depth_source_indices": []}
        return x, aux


class AttnResVisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = VisionPatchEmbed(config)
        self.blocks = nn.ModuleList(
            [AttnResVisionBlock(config, layer_index=index) for index in range(config.n_layers)]
        )
        self.final_residual = DepthAttentionResidual(
            config.d_model,
            temperature=config.attnres.temperature,
            window_size=config.attnres.window_size,
            rmsnorm_keys=config.attnres.rmsnorm_keys,
            zero_init_query=config.attnres.zero_init_queries,
            include_embedding=config.attnres.include_embedding,
            keep_embedding_in_window=config.attnres.keep_embedding_in_window,
            eps=config.norm_eps,
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_weight_capture(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, DepthAttentionResidual):
                module.capture_weights = enabled
                if not enabled:
                    module.last_weights = None
                    module.last_source_indices = []

    def iter_depth_residuals(self) -> list[DepthAttentionResidual]:
        return [module for module in self.modules() if isinstance(module, DepthAttentionResidual)]

    def forward(self, pixel_values: torch.Tensor, *, return_aux: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        history: list[torch.Tensor] = [self.patch_embed(pixel_values)]
        depth_rows: list[torch.Tensor] = []
        depth_source_indices: list[list[int]] = []

        for block in self.blocks:
            history, block_aux = block(history, return_aux=return_aux)
            if return_aux:
                for row in block_aux["depth_attention"]:
                    depth_rows.append(row["mean_weights"])
                    depth_source_indices.append(row["source_indices"])

        if self.config.attnres.final_readout:
            x, final_stats = self.final_residual(history, return_stats=return_aux)
            if return_aux:
                depth_rows.append(final_stats["mean_weights"])
                depth_source_indices.append(final_stats["source_indices"])
        else:
            x = history[-1]

        x = self.final_norm(x)
        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "depth_attention_rows": depth_rows,
                "depth_source_indices": depth_source_indices,
                **contribution_breakdown(depth_rows, depth_source_indices),
            }
        return x, aux


class BlockAttnResVisionTransformer(nn.Module):
    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        if config.attnres.num_blocks is None:
            raise ValueError("Block AttnRes vision encoder requires attnres.num_blocks")
        self.block_sizes = _block_sizes(config.n_layers, config.attnres.num_blocks)
        boundaries: set[int] = set()
        accumulated = 0
        for size in self.block_sizes:
            accumulated += size
            boundaries.add(accumulated - 1)
        self.block_boundaries = boundaries

        self.patch_embed = VisionPatchEmbed(config)
        self.blocks = nn.ModuleList(
            [BlockAttnResVisionBlock(config, layer_index=index) for index in range(config.n_layers)]
        )
        self.final_residual = DepthAttentionResidual(
            config.d_model,
            temperature=config.attnres.temperature,
            window_size=None,
            rmsnorm_keys=config.attnres.rmsnorm_keys,
            zero_init_query=config.attnres.zero_init_queries,
            include_embedding=config.attnres.include_embedding,
            keep_embedding_in_window=config.attnres.keep_embedding_in_window,
            eps=config.norm_eps,
        )
        self.final_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_weight_capture(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, DepthAttentionResidual):
                module.capture_weights = enabled
                if not enabled:
                    module.last_weights = None
                    module.last_source_indices = []

    def iter_depth_residuals(self) -> list[DepthAttentionResidual]:
        return [module for module in self.modules() if isinstance(module, DepthAttentionResidual)]

    def forward(self, pixel_values: torch.Tensor, *, return_aux: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        x0 = self.patch_embed(pixel_values)
        blocks: list[torch.Tensor] = [x0]
        partial: torch.Tensor | None = None
        depth_rows: list[torch.Tensor] = []
        depth_source_indices: list[list[int]] = []

        for layer_index, block in enumerate(self.blocks):
            partial, block_aux = block(blocks, partial, return_aux=return_aux)
            if return_aux:
                for row in block_aux["depth_attention"]:
                    depth_rows.append(row["mean_weights"])
                    depth_source_indices.append(row["source_indices"])
            if layer_index in self.block_boundaries:
                blocks.append(partial)
                partial = None

        if self.config.attnres.final_readout:
            x, final_stats = self.final_residual(blocks, return_stats=return_aux)
            if return_aux:
                depth_rows.append(final_stats["mean_weights"])
                depth_source_indices.append(final_stats["source_indices"])
        else:
            x = blocks[-1]

        x = self.final_norm(x)
        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "depth_attention_rows": depth_rows,
                "depth_source_indices": depth_source_indices,
                **contribution_breakdown(depth_rows, depth_source_indices),
            }
        return x, aux


def build_vision_encoder(config: VisionConfig) -> nn.Module:
    if config.residual == "baseline":
        return BaselineVisionTransformer(config)
    if config.residual == "attnres":
        return AttnResVisionTransformer(config)
    if config.residual == "block_attnres":
        return BlockAttnResVisionTransformer(config)
    raise ValueError(f"Unsupported vision residual: {config.residual}")
