from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn

from src.metrics.depth_metrics import contribution_breakdown
from src.models.baseline import CausalSelfAttention, FeedForward, GPTBaseline, RMSNorm
from src.utils.config import ModelConfig


class DepthAttentionResidual(nn.Module):
    """Depth-wise softmax aggregation over previous states."""

    def __init__(
        self,
        d_model: int,
        *,
        temperature: float = 1.0,
        window_size: int | None = None,
        rmsnorm_keys: bool = True,
        zero_init_query: bool = True,
        include_embedding: bool = True,
        keep_embedding_in_window: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.window_size = window_size
        self.include_embedding = include_embedding
        self.keep_embedding_in_window = keep_embedding_in_window
        self.query = nn.Parameter(torch.empty(d_model))
        self.key_norm = RMSNorm(d_model, eps=eps) if rmsnorm_keys else nn.Identity()
        self.capture_weights = False
        self.last_weights: torch.Tensor | None = None
        self.last_source_indices: list[int] = []
        if zero_init_query:
            nn.init.zeros_(self.query)
        else:
            nn.init.normal_(self.query, mean=0.0, std=0.02)

    def _select_history(self, history: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[int]]:
        if not history:
            raise ValueError("history must contain at least one tensor")
        indices = list(range(len(history)))
        selected_history = history
        if not self.include_embedding and indices:
            selected_history = selected_history[1:]
            indices = indices[1:]
        if self.window_size is None or len(selected_history) <= self.window_size + int(self.keep_embedding_in_window):
            return selected_history, indices
        if self.keep_embedding_in_window and indices:
            return [selected_history[0], *selected_history[-self.window_size :]], [indices[0], *indices[-self.window_size :]]
        return selected_history[-self.window_size :], indices[-self.window_size :]

    def forward(self, history: list[torch.Tensor], *, return_stats: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        selected_history, selected_indices = self._select_history(history)
        values = torch.stack(selected_history, dim=0)
        keys = self.key_norm(values)
        logits = torch.einsum("d,sbtd->sbt", self.query, keys)
        logits = logits / max(self.temperature, 1e-6)
        weights = torch.softmax(logits, dim=0)
        mixed = torch.einsum("sbt,sbtd->btd", weights, values)

        if self.capture_weights:
            self.last_weights = weights.detach().cpu()
            self.last_source_indices = list(selected_indices)

        stats: dict[str, Any] = {}
        if return_stats:
            mean_weights = weights.detach().mean(dim=(1, 2)).cpu()
            entropy = -(weights.detach() * weights.detach().clamp_min(1e-8).log()).sum(dim=0).mean().cpu()
            stats = {
                "source_indices": selected_indices,
                "mean_weights": mean_weights,
                "entropy": float(entropy.item()),
            }
        return mixed, stats


class AttnResTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
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
        self.attn = CausalSelfAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            bias=config.bias,
            max_seq_len=config.max_seq_len,
        )
        self.mlp_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.mlp = FeedForward(config.d_model, config.d_ff, config.dropout, bias=config.bias)

    def forward(self, history: list[torch.Tensor], *, return_aux: bool = False) -> tuple[list[torch.Tensor], dict[str, Any]]:
        attn_input, attn_stats = self.pre_attn_res(history, return_stats=return_aux)
        attn_out, _ = self.attn(self.attn_norm(attn_input))
        history.append(attn_out)

        mlp_input, mlp_stats = self.pre_mlp_res(history, return_stats=return_aux)
        mlp_out = self.mlp(self.mlp_norm(mlp_input))
        history.append(mlp_out)

        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "depth_attention": [
                    {"name": f"block_{self.layer_index:02d}_pre_attn", **attn_stats},
                    {"name": f"block_{self.layer_index:02d}_pre_mlp", **mlp_stats},
                ],
            }
        return history, aux


class GPTAttnRes(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [AttnResTransformerBlock(config, layer_index=index) for index in range(config.n_layers)]
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
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    @property
    def num_sublayers(self) -> int:
        return 2 * self.config.n_layers

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def set_weight_capture(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, DepthAttentionResidual):
                module.capture_weights = enabled
                if not enabled:
                    module.last_weights = None
                    module.last_source_indices = []

    def iter_depth_residuals(self) -> list[DepthAttentionResidual]:
        return [module for module in self.modules() if isinstance(module, DepthAttentionResidual)]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        *,
        return_aux: bool = False,
        prefix_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if input_ids is None and prefix_embeddings is None:
            raise ValueError("Provide input_ids, prefix_embeddings, or both")
        batch_size = prefix_embeddings.size(0) if prefix_embeddings is not None else input_ids.size(0)
        text_len = 0 if input_ids is None else input_ids.size(1)
        prefix_len = 0 if prefix_embeddings is None else prefix_embeddings.size(1)
        seq_len = prefix_len + text_len
        if seq_len > self.config.max_seq_len:
            raise ValueError("Input sequence is longer than model.max_seq_len")
        device = prefix_embeddings.device if prefix_embeddings is not None else input_ids.device
        positions = torch.arange(seq_len, device=device)
        position_embeddings = self.position_embedding(positions)[None, :, :]
        token_embeddings = self.token_embedding(input_ids) if input_ids is not None else None
        if prefix_embeddings is None:
            x0 = token_embeddings
        elif token_embeddings is None:
            x0 = prefix_embeddings
        else:
            x0 = torch.cat([prefix_embeddings, token_embeddings], dim=1)
        x0 = x0 + position_embeddings
        x0 = self.dropout(x0)
        history: list[torch.Tensor] = [x0]

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
        logits = self.lm_head(x)

        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "depth_attention_rows": depth_rows,
                "depth_source_indices": depth_source_indices,
                **contribution_breakdown(depth_rows, depth_source_indices),
            }
        return logits, aux


def _block_sizes(n_layers: int, num_blocks: int) -> list[int]:
    """Partition transformer layers into ``num_blocks`` contiguous blocks.

    Each block holds ``n_layers // num_blocks`` transformer layers and the final
    block absorbs any remainder, matching the paper ("the last block contains the
    remaining L mod N layers"). Counting note: ``num_blocks`` partitions
    transformer layers, and each transformer layer is two sublayers (attention +
    MLP), so the paper's sublayer-level ``block_size`` equals ``2 * (n_layers //
    num_blocks)``.
    """
    layers_per_block = n_layers // num_blocks
    remainder = n_layers % num_blocks
    sizes = [layers_per_block] * (num_blocks - 1) + [layers_per_block + remainder]
    if remainder != 0:
        warnings.warn(
            f"n_layers={n_layers} is not divisible by num_blocks={num_blocks}; "
            f"using block sizes {sizes} (final block absorbs the remainder)."
        )
    return sizes


class BlockAttnResTransformerBlock(nn.Module):
    """One transformer layer for Block AttnRes.

    Inside a block the residual stream (``partial``) accumulates additively, while
    the depth-attention mixers attend only over block-level sources (completed
    block summaries plus the current partial). Block resets are handled by the
    parent model at block boundaries.
    """

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
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
        self.attn = CausalSelfAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            bias=config.bias,
            max_seq_len=config.max_seq_len,
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
        attn_out, _ = self.attn(self.attn_norm(attn_input))
        partial = attn_out if partial is None else partial + attn_out

        mlp_input, mlp_stats = self.pre_mlp_res([*blocks, partial], return_stats=return_aux)
        mlp_out = self.mlp(self.mlp_norm(mlp_input))
        partial = partial + mlp_out

        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "depth_attention": [
                    {"name": f"block_{self.layer_index:02d}_pre_attn", **attn_stats},
                    {"name": f"block_{self.layer_index:02d}_pre_mlp", **mlp_stats},
                ],
            }
        return partial, aux


class GPTBlockAttnRes(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.attnres.num_blocks is None:
            raise ValueError("Block AttnRes requires model.attnres.num_blocks to be set")
        self.block_sizes = _block_sizes(config.n_layers, config.attnres.num_blocks)
        boundaries: set[int] = set()
        accumulated = 0
        for size in self.block_sizes:
            accumulated += size
            boundaries.add(accumulated - 1)
        self.block_boundaries = boundaries

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [BlockAttnResTransformerBlock(config, layer_index=index) for index in range(config.n_layers)]
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
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    @property
    def num_sublayers(self) -> int:
        return 2 * self.config.n_layers

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def set_weight_capture(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, DepthAttentionResidual):
                module.capture_weights = enabled
                if not enabled:
                    module.last_weights = None
                    module.last_source_indices = []

    def iter_depth_residuals(self) -> list[DepthAttentionResidual]:
        return [module for module in self.modules() if isinstance(module, DepthAttentionResidual)]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        *,
        return_aux: bool = False,
        prefix_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if input_ids is None and prefix_embeddings is None:
            raise ValueError("Provide input_ids, prefix_embeddings, or both")
        text_len = 0 if input_ids is None else input_ids.size(1)
        prefix_len = 0 if prefix_embeddings is None else prefix_embeddings.size(1)
        seq_len = prefix_len + text_len
        if seq_len > self.config.max_seq_len:
            raise ValueError("Input sequence is longer than model.max_seq_len")
        device = prefix_embeddings.device if prefix_embeddings is not None else input_ids.device
        positions = torch.arange(seq_len, device=device)
        position_embeddings = self.position_embedding(positions)[None, :, :]
        token_embeddings = self.token_embedding(input_ids) if input_ids is not None else None
        if prefix_embeddings is None:
            x0 = token_embeddings
        elif token_embeddings is None:
            x0 = prefix_embeddings
        else:
            x0 = torch.cat([prefix_embeddings, token_embeddings], dim=1)
        x0 = x0 + position_embeddings
        x0 = self.dropout(x0)

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
        logits = self.lm_head(x)

        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "depth_attention_rows": depth_rows,
                "depth_source_indices": depth_source_indices,
                **contribution_breakdown(depth_rows, depth_source_indices),
            }
        return logits, aux


def build_model(config: ModelConfig) -> nn.Module:
    if config.architecture == "baseline":
        return GPTBaseline(config)
    if config.architecture == "block_attnres" or (
        config.architecture == "attnres" and config.attnres.mode == "block"
    ):
        return GPTBlockAttnRes(config)
    if config.architecture == "attnres":
        return GPTAttnRes(config)
    raise ValueError(f"Unsupported architecture: {config.architecture}")
