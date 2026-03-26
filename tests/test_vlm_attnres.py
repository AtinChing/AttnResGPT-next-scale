from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.vlm_attnres as vlm_module
from src.models.vlm_attnres import SiglipAttnResCaptioner
from src.utils.config import AttnResConfig, ModelConfig


class DummyProcessor:
    pass


class DummyVisionEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=pixel_values)


def _decoder_config(architecture: str) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        size_name="small",
        vocab_size=32,
        max_seq_len=16,
        d_model=12,
        n_layers=2,
        n_heads=3,
        d_ff=24,
        dropout=0.0,
        tie_weights=False,
        attnres=AttnResConfig(enabled=architecture == "attnres", final_readout=True),
    )


def _build_captioner(monkeypatch: pytest.MonkeyPatch, architecture: str) -> SiglipAttnResCaptioner:
    def fake_processor_from_pretrained(*args, **kwargs) -> DummyProcessor:
        return DummyProcessor()

    def fake_model_from_pretrained(*args, **kwargs) -> DummyVisionEncoder:
        return DummyVisionEncoder(hidden_size=6)

    monkeypatch.setattr(vlm_module.AutoProcessor, "from_pretrained", fake_processor_from_pretrained)
    monkeypatch.setattr(vlm_module.AutoModel, "from_pretrained", fake_model_from_pretrained)
    model = SiglipAttnResCaptioner(
        vision_model_name="dummy/siglip",
        decoder_config=_decoder_config(architecture),
    )
    model.eval()
    return model


@pytest.mark.parametrize("architecture", ["baseline", "attnres"])
def test_vlm_logits_start_at_bos_position(monkeypatch: pytest.MonkeyPatch, architecture: str) -> None:
    torch.manual_seed(0)
    model = _build_captioner(monkeypatch, architecture)
    vision_hidden = torch.randn(1, 3, 6)
    input_ids = torch.tensor([[1, 7, 8, 9]], dtype=torch.long)
    targets = torch.tensor([[7, 8, 9, 10]], dtype=torch.long)

    prefix_embeddings = model.connector(vision_hidden.to(dtype=model.connector.weight.dtype))
    full_logits, _ = model.decoder(
        input_ids,
        return_aux=False,
        prefix_embeddings=prefix_embeddings,
    )
    payload = model(
        vision_hidden=vision_hidden,
        input_ids=input_ids,
        targets=targets,
        return_aux=False,
    )

    prefix_len = prefix_embeddings.size(1)
    expected_logits = full_logits[:, prefix_len : prefix_len + targets.size(1), :]

    assert int(payload["prefix_length"]) == prefix_len
    torch.testing.assert_close(payload["logits"], expected_logits)


@pytest.mark.parametrize("architecture", ["baseline", "attnres"])
def test_vlm_loss_matches_corrected_caption_slice(monkeypatch: pytest.MonkeyPatch, architecture: str) -> None:
    torch.manual_seed(0)
    model = _build_captioner(monkeypatch, architecture)
    vision_hidden = torch.randn(1, 2, 6)
    input_ids = torch.tensor([[1, 4, 5]], dtype=torch.long)
    targets = torch.tensor([[4, 5, 6]], dtype=torch.long)

    prefix_embeddings = model.connector(vision_hidden.to(dtype=model.connector.weight.dtype))
    full_logits, _ = model.decoder(
        input_ids,
        return_aux=False,
        prefix_embeddings=prefix_embeddings,
    )
    payload = model(
        vision_hidden=vision_hidden,
        input_ids=input_ids,
        targets=targets,
        return_aux=False,
    )

    prefix_len = prefix_embeddings.size(1)
    expected_logits = full_logits[:, prefix_len : prefix_len + targets.size(1), :]
    manual_loss = F.cross_entropy(
        expected_logits.reshape(-1, expected_logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )

    torch.testing.assert_close(payload["logits"], expected_logits)
    torch.testing.assert_close(payload["loss"], manual_loss)
