from __future__ import annotations

import torch

from src.models.attnres import DepthAttentionResidual, build_model
from src.utils.config import AttnResConfig, ModelConfig
from src.utils.runtime import count_parameters


def _make_model_config(architecture: str) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        size_name="small",
        vocab_size=64,
        max_seq_len=16,
        d_model=48,
        n_layers=2,
        n_heads=4,
        d_ff=96,
        dropout=0.0,
        attnres=AttnResConfig(enabled=architecture == "attnres", final_readout=True),
    )


def _make_block_config(n_layers: int = 4, num_blocks: int = 2) -> ModelConfig:
    return ModelConfig(
        architecture="block_attnres",
        size_name="small",
        vocab_size=64,
        max_seq_len=16,
        d_model=48,
        n_layers=n_layers,
        n_heads=4,
        d_ff=96,
        dropout=0.0,
        attnres=AttnResConfig(enabled=True, mode="block", num_blocks=num_blocks, final_readout=True),
    )


def _make_full_config(n_layers: int = 4) -> ModelConfig:
    return ModelConfig(
        architecture="attnres",
        size_name="small",
        vocab_size=64,
        max_seq_len=16,
        d_model=48,
        n_layers=n_layers,
        n_heads=4,
        d_ff=96,
        dropout=0.0,
        attnres=AttnResConfig(enabled=True, final_readout=True),
    )


def test_baseline_and_attnres_shapes() -> None:
    input_ids = torch.randint(0, 64, (2, 16))
    baseline = build_model(_make_model_config("baseline"))
    attnres = build_model(_make_model_config("attnres"))

    baseline_logits, baseline_aux = baseline(input_ids, return_aux=True)
    attnres_logits, attnres_aux = attnres(input_ids, return_aux=True)

    assert baseline_logits.shape == (2, 16, 64)
    assert attnres_logits.shape == (2, 16, 64)
    assert len(baseline_aux["block_output_norms"]) == 2
    assert len(attnres_aux["block_output_norms"]) == 2
    assert len(attnres_aux["depth_attention_rows"]) == 5


def test_attnres_weights_sum_to_one() -> None:
    module = DepthAttentionResidual(
        d_model=16,
        temperature=1.0,
        window_size=None,
        zero_init_query=True,
        include_embedding=True,
        keep_embedding_in_window=True,
    )
    history = [torch.randn(2, 4, 16) for _ in range(5)]
    _, stats = module(history, return_stats=True)
    total = stats["mean_weights"].sum().item()
    assert abs(total - 1.0) < 1e-5
    assert stats["source_indices"] == [0, 1, 2, 3, 4]


def test_parameter_counts_stay_close() -> None:
    baseline = build_model(_make_model_config("baseline"))
    attnres = build_model(_make_model_config("attnres"))
    baseline_total = count_parameters(baseline)["total"]
    attnres_total = count_parameters(attnres)["total"]
    delta_pct = abs(attnres_total - baseline_total) / baseline_total
    assert attnres_total > baseline_total
    assert delta_pct < 0.05


def test_block_attnres_shapes() -> None:
    input_ids = torch.randint(0, 64, (2, 16))
    model = build_model(_make_block_config(n_layers=4, num_blocks=2))

    logits, aux = model(input_ids, return_aux=True)

    assert logits.shape == (2, 16, 64)
    # one stream norm per transformer layer, two mixers per layer plus final readout
    assert len(aux["block_output_norms"]) == 4
    assert len(aux["depth_attention_rows"]) == 2 * 4 + 1


def test_block_attnres_weights_sum_to_one() -> None:
    input_ids = torch.randint(0, 64, (2, 16))
    model = build_model(_make_block_config(n_layers=4, num_blocks=2))

    _, aux = model(input_ids, return_aux=True)

    for row in aux["depth_attention_rows"]:
        assert abs(float(row.sum().item()) - 1.0) < 1e-5


def test_block_attnres_param_count_matches_full() -> None:
    baseline_total = count_parameters(build_model(_make_model_config("baseline")))["total"]
    full_total = count_parameters(build_model(_make_full_config(n_layers=4)))["total"]
    block_total = count_parameters(build_model(_make_block_config(n_layers=4, num_blocks=2)))["total"]

    # Block mode uses the same number of depth mixers as Full, so parameter
    # counts must match exactly, and both exceed the baseline.
    assert block_total == full_total
    assert block_total > baseline_total


def test_block_reset_bounds_activation_norm() -> None:
    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (4, 16))
    # Two blocks of two layers each: boundaries after layer 1 and layer 3.
    model = build_model(_make_block_config(n_layers=4, num_blocks=2))

    _, aux = model(input_ids, return_aux=True)
    norms = aux["block_output_norms"]

    # Within the first block the additive stream grows.
    assert norms[0] < norms[1]
    # The block reset drops the accumulated magnitude at the start of block two.
    assert norms[2] < norms[1]
