from __future__ import annotations

import torch

from src.metrics.norms import Layer1DepthAttentionProbe, LayerInputMagnitudeTracker, language_model_loss
from src.models.attnres import DepthAttentionResidual, _block_sizes, build_model
from src.training.eval import evaluate_model
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


def test_block_sizes_are_balanced() -> None:
    # Remainder is spread across the first blocks, not dumped into the last one.
    assert _block_sizes(12, 8) == [2, 2, 2, 2, 1, 1, 1, 1]
    assert _block_sizes(14, 8) == [2, 2, 2, 2, 2, 2, 1, 1]
    assert _block_sizes(16, 8) == [2, 2, 2, 2, 2, 2, 2, 2]
    assert _block_sizes(17, 8) == [3, 2, 2, 2, 2, 2, 2, 2]
    assert _block_sizes(21, 8) == [3, 3, 3, 3, 3, 2, 2, 2]
    assert _block_sizes(6, 3) == [2, 2, 2]


def test_block_attnres_uses_balanced_partition() -> None:
    model = build_model(_make_block_config(n_layers=12, num_blocks=8))
    assert model.block_sizes == [2, 2, 2, 2, 1, 1, 1, 1]
    assert max(model.block_sizes) - min(model.block_sizes) <= 1


def test_layer1_depth_attention_probe_full_and_block() -> None:
    """Layer-1 depth-attn weights + source mags for Full (3 sources) and Block (2)."""
    input_ids = torch.randint(0, 64, (2, 8))

    torch.manual_seed(0)
    full = build_model(_matched_config("attnres", n_layers=4, num_blocks=2))
    probe = Layer1DepthAttentionProbe()
    probe.register(full)
    full.train()
    full(input_ids, return_aux=False)
    snap = probe.snapshot()
    probe.close()
    assert snap["layer1_depth_attn/n_sources"] == 3.0
    for name in ("emb", "l0_attn", "l0_mlp"):
        assert f"layer1_depth_attn/weight_{name}" in snap
        assert f"layer1_depth_attn/mag_{name}" in snap
        assert snap[f"layer1_depth_attn/weight_{name}"] >= 0.0
        assert snap[f"layer1_depth_attn/mag_{name}"] >= 0.0
    weight_sum = sum(snap[f"layer1_depth_attn/weight_{name}"] for name in ("emb", "l0_attn", "l0_mlp"))
    assert abs(weight_sum - 1.0) < 1e-5
    assert snap["layer1_depth_attn/mixed_mag"] >= 0.0

    torch.manual_seed(0)
    block = build_model(_make_block_config(n_layers=4, num_blocks=2))
    probe = Layer1DepthAttentionProbe()
    probe.register(block)
    block.train()
    block(input_ids, return_aux=False)
    snap = probe.snapshot()
    probe.close()
    assert snap["layer1_depth_attn/n_sources"] == 2.0
    assert "layer1_depth_attn/weight_emb" in snap
    assert "layer1_depth_attn/weight_partial_after_l0" in snap
    assert "layer1_depth_attn/weight_l0_attn" not in snap

    baseline = build_model(_matched_config("baseline", n_layers=4))
    probe = Layer1DepthAttentionProbe()
    probe.register(baseline)
    baseline(input_ids)
    assert probe.snapshot() == {}
    probe.close()


def test_baseline_and_attnres_shapes() -> None:
    input_ids = torch.randint(0, 64, (2, 16))
    baseline = build_model(_make_model_config("baseline"))
    attnres = build_model(_make_model_config("attnres"))

    baseline_logits, baseline_aux = baseline(input_ids, return_aux=True)
    attnres_logits, attnres_aux = attnres(input_ids, return_aux=True)

    assert baseline_logits.shape == (2, 16, 64)
    assert attnres_logits.shape == (2, 16, 64)
    assert "block_output_magnitudes" not in baseline_aux
    assert "block_output_magnitudes" not in attnres_aux
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
    assert "block_output_magnitudes" not in aux
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


def test_layer_input_magnitude_uses_same_definition_all_variants() -> None:
    """Fig. 5b records the tensor entering each block's attention layer."""
    input_ids = torch.randint(0, 64, (3, 32))

    for architecture in ("baseline", "attnres", "block_attnres"):
        torch.manual_seed(0)
        model = build_model(_matched_config(architecture))
        observed_inputs: dict[str, torch.Tensor] = {}
        reference_handles = []
        for layer_index, block in enumerate(model.blocks):
            site = f"blocks.{layer_index}"
            reference_handles.append(
                block.attn_norm.register_forward_pre_hook(
                    lambda _module, inputs, site=site: observed_inputs.__setitem__(site, inputs[0].detach())
                )
            )

        tracker = LayerInputMagnitudeTracker()
        tracker.register(model)
        try:
            model.eval()
            with torch.no_grad():
                _, aux = model(input_ids, return_aux=True)
            magnitudes = tracker.snapshot()["layer_input_magnitudes"]
        finally:
            tracker.close()
            for handle in reference_handles:
                handle.remove()

        expected_keys = {f"blocks.{i}" for i in range(model.config.n_layers)}
        assert set(magnitudes) == set(observed_inputs) == expected_keys
        assert "block_output_magnitudes" not in aux
        for site, layer_input in observed_inputs.items():
            expected = float(layer_input.float().norm(dim=-1).mean().item())
            assert abs(magnitudes[site] - expected) < 1e-5


def test_layer_input_gradient_magnitude_uses_same_definition_all_variants() -> None:
    """Fig. 5c records ``dL/dh_l`` at the same layer-input sites as Fig. 5b."""
    input_ids = torch.randint(0, 64, (3, 32))
    targets = torch.randint(0, 64, (3, 32))

    key_sets = {}
    for architecture in ("baseline", "attnres", "block_attnres"):
        torch.manual_seed(0)
        model = build_model(_matched_config(architecture))
        observed_inputs: dict[str, torch.Tensor] = {}
        reference_handles = []
        for layer_index, block in enumerate(model.blocks):
            site = f"blocks.{layer_index}"

            def retain_layer_input(_module, inputs, *, site=site):
                inputs[0].retain_grad()
                observed_inputs[site] = inputs[0]

            reference_handles.append(block.attn_norm.register_forward_pre_hook(retain_layer_input))

        tracker = LayerInputMagnitudeTracker()
        tracker.register(model)
        try:
            model.train()
            tracker.reset_step()
            logits, _ = model(input_ids, return_aux=False)
            language_model_loss(logits, targets).backward()
            snapshot = tracker.snapshot()
            input_magnitudes = snapshot["layer_input_magnitudes"]
            gradient_magnitudes = snapshot["layer_input_gradient_magnitudes"]
        finally:
            tracker.close()
            for handle in reference_handles:
                handle.remove()

        expected_keys = {f"blocks.{i}" for i in range(model.config.n_layers)}
        assert set(input_magnitudes) == set(gradient_magnitudes) == expected_keys
        assert all(value >= 0.0 for value in gradient_magnitudes.values())
        for site, layer_input in observed_inputs.items():
            expected = float(layer_input.grad.detach().float().norm(dim=-1).mean().item())
            assert abs(gradient_magnitudes[site] - expected) < 1e-5
        key_sets[architecture] = set(gradient_magnitudes)

    # Identical measurement surface across all three variants.
    assert key_sets["baseline"] == key_sets["attnres"] == key_sets["block_attnres"]


def test_eval_averages_and_preserves_layer_input_curve() -> None:
    model = build_model(_matched_config("block_attnres", n_layers=4, num_blocks=2))
    batches = [
        {
            "input_ids": torch.randint(0, 64, (2, 32)),
            "targets": torch.randint(0, 64, (2, 32)),
        }
        for _ in range(2)
    ]
    observed: dict[str, list[float]] = {f"blocks.{i}": [] for i in range(model.config.n_layers)}
    handles = []
    for layer_index, block in enumerate(model.blocks):
        site = f"blocks.{layer_index}"

        def record_layer_input(_module, inputs, *, site=site):
            value = inputs[0].detach().float().norm(dim=-1).mean().item()
            observed[site].append(float(value))

        handles.append(block.attn_norm.register_forward_pre_hook(record_layer_input))

    try:
        metrics = evaluate_model(
            model,
            batches,
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
            collect_artifacts=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    curve = metrics["mean_layer_input_magnitudes"]
    assert list(curve) == [f"blocks.{i}" for i in range(model.config.n_layers)]
    for site, values in observed.items():
        assert len(values) == len(batches)
        assert abs(curve[site] - sum(values) / len(values)) < 1e-5
    assert metrics["mean_layer_input_magnitude_last_layer"] == curve["blocks.3"]
    assert "mean_block_output_magnitude" not in metrics
    assert "mean_activation_norms" not in metrics
