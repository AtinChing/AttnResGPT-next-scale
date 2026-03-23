from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        bias: bool = True,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if return_attention or not hasattr(F, "scaled_dot_product_attention"):
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            causal = self.causal_mask[:seq_len, :seq_len]
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            output = torch.matmul(attn, v)
            attn_summary = attn.mean(dim=1)
        else:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            attn_summary = None

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.resid_dropout(self.out_proj(output))
        return output, attn_summary


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, *, bias: bool = True) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=bias)
        self.activation = nn.GELU(approximate="tanh")
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return self.dropout(x)


class BaselineTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
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

    def forward(self, x: torch.Tensor, *, return_aux: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        attn_out, _ = self.attn(self.attn_norm(x))
        x = x + attn_out
        mlp_out = self.mlp(self.mlp_norm(x))
        x = x + mlp_out

        aux: dict[str, Any] = {}
        if return_aux:
            aux["block_output_norm"] = float(x.detach().float().norm(dim=-1).mean().item())
        return x, aux


class GPTBaseline(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [BaselineTransformerBlock(config, layer_index=index) for index in range(config.n_layers)]
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
            x = token_embeddings
        elif token_embeddings is None:
            x = prefix_embeddings
        else:
            x = torch.cat([prefix_embeddings, token_embeddings], dim=1)
        x = x + position_embeddings
        x = self.dropout(x)

        block_output_norms: list[float] = []
        for block in self.blocks:
            x, aux = block(x, return_aux=return_aux)
            if return_aux:
                block_output_norms.append(aux["block_output_norm"])

        x = self.final_norm(x)
        logits = self.lm_head(x)
        aux: dict[str, Any] = {}
        if return_aux:
            aux = {
                "block_output_norms": block_output_norms,
                "depth_attention_rows": [],
                "depth_source_indices": [],
            }
        return logits, aux
