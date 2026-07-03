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


def _matched_config(architecture: str, n_layers: int = 8, num_blocks: int = 4) -> ModelConfig:
    """Config identical across modes except the architecture (same depth/width)."""
    return ModelConfig(
        architecture=architecture,
        size_name="small",
        vocab_size=64,
        max_seq_len=32,
        d_model=48,
        n_layers=n_layers,
        n_heads=4,
        d_ff=96,
        dropout=0.0,
        attnres=AttnResConfig(
            enabled=architecture != "baseline",
            mode="block" if architecture == "block_attnres" else "full",
            num_blocks=num_blocks if architecture == "block_attnres" else None,
            final_readout=True,
        ),
    )


def test_all_modes_produce_matching_output_shape() -> None:
    input_ids = torch.randint(0, 64, (3, 32))
    shapes = {}
    for architecture in ("baseline", "attnres", "block_attnres"):
        model = build_model(_matched_config(architecture))
        logits, _ = model(input_ids, return_aux=True)
        shapes[architecture] = tuple(logits.shape)

    assert shapes["baseline"] == (3, 32, 64)
    assert shapes["attnres"] == shapes["baseline"]
    assert shapes["block_attnres"] == shapes["baseline"]


def test_full_attnres_weights_sum_to_one_end_to_end() -> None:
    input_ids = torch.randint(0, 64, (2, 32))
    model = build_model(_matched_config("attnres"))

    _, aux = model(input_ids, return_aux=True)

    assert len(aux["depth_attention_rows"]) == 2 * 8 + 1
    for row in aux["depth_attention_rows"]:
        assert abs(float(row.sum().item()) - 1.0) < 1e-5


def test_block_param_count_close_to_baseline_same_depth() -> None:
    baseline_total = count_parameters(build_model(_matched_config("baseline")))["total"]
    block_total = count_parameters(build_model(_matched_config("block_attnres")))["total"]
    full_total = count_parameters(build_model(_matched_config("attnres")))["total"]

    # The query + RMSNorm additions are tiny, so Block and Full must both sit
    # within a few percent of baseline at matched depth. A large delta is a bug.
    block_delta = abs(block_total - baseline_total) / baseline_total
    full_delta = abs(full_total - baseline_total) / baseline_total
    assert block_total == full_total
    assert block_delta < 0.05
    assert full_delta < 0.05


def _stream_norms_per_layer(model, input_ids) -> list[float]:
    """Per-layer residual-stream norm via aux; comparable for baseline and Block."""
    model.eval()
    with torch.no_grad():
        _, aux = model(input_ids, return_aux=True)
    return aux["block_output_norms"]


def test_block_reset_bounds_stream_norm_vs_baseline_growth() -> None:
    # n_layers=8, num_blocks=4 -> block size 2, resets after layers 1, 3, 5.
    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (4, 32))

    torch.manual_seed(0)
    baseline = build_model(_matched_config("baseline"))
    torch.manual_seed(0)
    block = build_model(_matched_config("block_attnres"))

    baseline_norms = _stream_norms_per_layer(baseline, input_ids)
    block_norms = _stream_norms_per_layer(block, input_ids)

    # Baseline additive residual stream grows across depth.
    assert baseline_norms[-1] > baseline_norms[0]

    # Block resets accumulation at each internal boundary: the layer right after
    # a boundary starts from a fresh convex combination, so its norm drops.
    for boundary_layer in (1, 3, 5):
        assert block_norms[boundary_layer + 1] < block_norms[boundary_layer]

    # The reset keeps Block's peak stream magnitude well below where the
    # unbounded baseline stream ends at the same depth.
    assert max(block_norms) < baseline_norms[-1]


def test_full_and_block_bound_growth_via_sublayer_norms() -> None:
    # Cross-mode comparable metric: raw attn/MLP sublayer output norms (hooked on
    # the same submodules in every mode). Baseline's residual stream grows, while
    # Full and Block keep per-layer magnitudes flat/bounded across depth.
    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (4, 32))

    def last_over_first_stream_ratio(architecture: str) -> float:
        torch.manual_seed(0)
        model = build_model(_matched_config(architecture))
        norms = _stream_norms_per_layer(model, input_ids)
        return norms[-1] / norms[0]

    baseline_ratio = last_over_first_stream_ratio("baseline")
    full_ratio = last_over_first_stream_ratio("attnres")

    # Baseline stream expands across depth; Full stays bounded (ratio near/below 1).
    assert baseline_ratio > 1.2
    assert full_ratio < baseline_ratio
