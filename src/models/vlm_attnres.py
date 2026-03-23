from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from src.metrics.norms import perplexity_from_loss
from src.models.attnres import GPTAttnRes
from src.models.baseline import GPTBaseline
from src.utils.config import ModelConfig


def masked_language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)


@dataclass
class VisionLanguageAlphaSummary:
    vision_rows: list[list[float]]
    language_rows: list[list[float]]
    source_indices: list[list[int]]
    vision_entropy: list[float]
    language_entropy: list[float]
    vision_embedding: list[float]
    language_embedding: list[float]


class SiglipAttnResCaptioner(nn.Module):
    def __init__(
        self,
        *,
        vision_model_name: str,
        decoder_config: ModelConfig,
    ) -> None:
        super().__init__()
        processor = AutoProcessor.from_pretrained(vision_model_name)
        vision_backbone = AutoModel.from_pretrained(vision_model_name)
        if hasattr(vision_backbone, "vision_model"):
            vision_model = vision_backbone.vision_model
        else:
            vision_model = vision_backbone

        self.processor = processor
        self.vision_model_name = vision_model_name
        self.vision_encoder = vision_model.eval()
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False

        vision_hidden_size = int(getattr(self.vision_encoder.config, "hidden_size"))
        self.connector = nn.Linear(vision_hidden_size, decoder_config.d_model)
        if decoder_config.architecture == "attnres":
            self.decoder = GPTAttnRes(decoder_config)
        elif decoder_config.architecture == "baseline":
            self.decoder = GPTBaseline(decoder_config)
        else:
            raise ValueError(f"Unsupported decoder architecture: {decoder_config.architecture}")

    @property
    def decoder_config(self) -> ModelConfig:
        return self.decoder.config

    @property
    def supports_alpha_analysis(self) -> bool:
        return isinstance(self.decoder, GPTAttnRes)

    def set_weight_capture(self, enabled: bool) -> None:
        if hasattr(self.decoder, "set_weight_capture"):
            self.decoder.set_weight_capture(enabled)

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.vision_encoder(pixel_values=pixel_values)
        return outputs.last_hidden_state

    def forward(
        self,
        *,
        pixel_values: torch.Tensor | None = None,
        vision_hidden: torch.Tensor | None = None,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> dict[str, Any]:
        if vision_hidden is None:
            if pixel_values is None:
                raise ValueError("Provide pixel_values or precomputed vision_hidden")
            vision_hidden = self.encode_vision(pixel_values)
        vision_hidden = vision_hidden.to(dtype=self.connector.weight.dtype)
        prefix_embeddings = self.connector(vision_hidden)
        logits, aux = self.decoder(
            input_ids,
            return_aux=return_aux,
            prefix_embeddings=prefix_embeddings,
        )
        prefix_len = prefix_embeddings.size(1)
        target_len = input_ids.size(1) if targets is None else targets.size(1)
        text_logits = logits[:, prefix_len - 1 : prefix_len - 1 + target_len, :]

        payload: dict[str, Any] = {
            "logits": text_logits,
            "prefix_length": prefix_len,
            "aux": aux,
        }
        if targets is not None:
            loss = masked_language_model_loss(text_logits, targets)
            payload["loss"] = loss
            payload["perplexity"] = perplexity_from_loss(float(loss.item()))
        return payload


def _row_entropy(row: Sequence[float]) -> float:
    array = np.asarray(row, dtype=np.float64)
    array = np.clip(array, 1e-8, 1.0)
    return float(-(array * np.log(array)).sum())


def _embedding_contribution(row: Sequence[float], indices: Sequence[int]) -> float:
    if 0 not in indices:
        return 0.0
    return float(row[list(indices).index(0)])


@torch.no_grad()
def summarize_alpha_by_token_type(
    model: SiglipAttnResCaptioner,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
    mixed_precision: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> VisionLanguageAlphaSummary:
    if not model.supports_alpha_analysis:
        raise ValueError("Alpha summarization is only available for AttnRes decoders.")
    model.eval()
    model.set_weight_capture(True)

    vision_sums: list[torch.Tensor] = []
    language_sums: list[torch.Tensor] = []
    vision_counts: list[float] = []
    language_counts: list[float] = []
    source_indices: list[list[int]] = []

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        pixel_values = batch.get("pixel_values")
        input_ids = batch["input_ids"].to(device)
        text_mask = batch["text_mask"].cpu()
        vision_hidden = batch.get("vision_hidden")
        model_kwargs = {
            "pixel_values": pixel_values.to(device) if pixel_values is not None and vision_hidden is None else None,
            "vision_hidden": vision_hidden.to(device) if vision_hidden is not None else None,
            "input_ids": input_ids,
            "return_aux": False,
        }
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
            if device.type == "cuda" and mixed_precision
            else nullcontext()
        )
        with autocast_context:
            output = model(**model_kwargs)
        prefix_len = int(output["prefix_length"])

        for site_index, residual in enumerate(model.decoder.iter_depth_residuals()):
            weights = residual.last_weights
            indices = residual.last_source_indices
            if weights is None:
                continue

            seq_total = int(weights.size(2))
            batch_size = int(weights.size(1))
            vision_mask = torch.zeros((batch_size, seq_total), dtype=torch.float32)
            language_mask = torch.zeros((batch_size, seq_total), dtype=torch.float32)
            vision_mask[:, :prefix_len] = 1.0
            language_width = min(text_mask.size(1), seq_total - prefix_len)
            language_mask[:, prefix_len : prefix_len + language_width] = text_mask[:, :language_width].float()

            vision_denom = float(vision_mask.sum().item())
            language_denom = float(language_mask.sum().item())
            vision_row = (weights * vision_mask.unsqueeze(0)).sum(dim=(1, 2))
            language_row = (weights * language_mask.unsqueeze(0)).sum(dim=(1, 2))

            if len(vision_sums) <= site_index:
                vision_sums.append(vision_row.clone())
                language_sums.append(language_row.clone())
                vision_counts.append(vision_denom)
                language_counts.append(language_denom)
                source_indices.append(list(indices))
            else:
                vision_sums[site_index] += vision_row
                language_sums[site_index] += language_row
                vision_counts[site_index] += vision_denom
                language_counts[site_index] += language_denom

    model.set_weight_capture(False)

    vision_rows: list[list[float]] = []
    language_rows: list[list[float]] = []
    for site_index in range(len(vision_sums)):
        vision_rows.append((vision_sums[site_index] / max(vision_counts[site_index], 1.0)).tolist())
        language_rows.append((language_sums[site_index] / max(language_counts[site_index], 1.0)).tolist())

    vision_entropy = [_row_entropy(row) for row in vision_rows]
    language_entropy = [_row_entropy(row) for row in language_rows]
    vision_embedding = [_embedding_contribution(row, indices) for row, indices in zip(vision_rows, source_indices)]
    language_embedding = [_embedding_contribution(row, indices) for row, indices in zip(language_rows, source_indices)]

    return VisionLanguageAlphaSummary(
        vision_rows=vision_rows,
        language_rows=language_rows,
        source_indices=source_indices,
        vision_entropy=vision_entropy,
        language_entropy=language_entropy,
        vision_embedding=vision_embedding,
        language_embedding=language_embedding,
    )
