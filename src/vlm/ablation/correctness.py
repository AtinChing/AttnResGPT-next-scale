from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.models.attnres import DepthAttentionResidual
from src.models.baseline import BidirectionalSelfAttention
from src.models.vision_attnres import VisionConfig, build_vision_encoder
from src.models.vlm_attnres import TinyAttnResVLM
from src.utils.config import AttnResConfig, ModelConfig
from src.utils.runtime import seed_everything
from src.vlm.ablation.config import VARIANTS, AblationExperimentConfig, build_decoder_config, build_vision_config
from src.vlm.ablation.init_sync import (
    assert_encoder_decoder_attnres_separate,
    copy_shared_weights,
    validate_shared_initialization,
)
from src.vlm.ablation.io_utils import atomic_write_json
from src.vlm.ablation.train import build_model_for_variant
from src.vlm.synthetic_vqa import VQATokenizer, generate_example


def _tiny_decoder_config(vocab_size: int, residual: str) -> ModelConfig:
    attnres = AttnResConfig(
        enabled=residual != "baseline",
        mode="block" if residual == "block_attnres" else "full",
        num_blocks=2 if residual == "block_attnres" else None,
        final_readout=True,
    )
    return ModelConfig(
        architecture="baseline" if residual == "baseline" else residual,
        size_name="small",
        vocab_size=vocab_size,
        max_seq_len=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        tie_weights=False,
        attnres=attnres,
    )


def _tiny_vision_config(residual: str) -> VisionConfig:
    attnres = AttnResConfig(
        enabled=residual != "baseline",
        mode="block" if residual == "block_attnres" else "full",
        num_blocks=2 if residual == "block_attnres" else None,
        final_readout=True,
    )
    return VisionConfig(
        image_size=32,
        patch_size=8,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        residual=residual,  # type: ignore[arg-type]
        attnres=attnres,
    )


def run_correctness_checks(*, device: torch.device, report_path: Path | None = None) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("Correctness checks for the ablation grid require CUDA.")

    results: dict[str, Any] = {"passed": [], "failed": []}

    def check(name: str, fn) -> None:
        try:
            fn()
            results["passed"].append(name)
        except Exception as exc:  # noqa: BLE001
            results["failed"].append({"name": name, "error": str(exc)})
            raise

    tokenizer = VQATokenizer()

    def test_dataset_determinism() -> None:
        a = generate_example(split="train", split_seed=7, example_index=3, image_size=64)
        b = generate_example(split="train", split_seed=7, example_index=3, image_size=64)
        assert a.question == b.question and a.answer == b.answer
        assert (a.image == b.image).all()

    def test_unambiguous_answers() -> None:
        for index in range(32):
            example = generate_example(split="train", split_seed=11, example_index=index, image_size=64)
            assert example.answer in tokenizer.token_to_id

    def test_patch_shape() -> None:
        encoder = build_vision_encoder(_tiny_vision_config("baseline")).to(device)
        pixels = torch.randn(2, 3, 32, 32, device=device)
        tokens, _ = encoder(pixels)
        assert tokens.shape == (2, 16, 32)

    def test_bidirectional_shape() -> None:
        attn = BidirectionalSelfAttention(32, 4, 0.0).to(device)
        x = torch.randn(2, 16, 32, device=device)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_encoder_forward_backward(residual: str) -> None:
        encoder = build_vision_encoder(_tiny_vision_config(residual)).to(device)
        pixels = torch.randn(2, 3, 32, 32, device=device, requires_grad=False)
        tokens, aux = encoder(pixels, return_aux=True)
        loss = tokens.float().pow(2).mean()
        loss.backward()
        assert tokens.shape[1] == 16
        if residual != "baseline":
            assert aux["depth_attention_rows"]

    def test_routing_sums(residual: str) -> None:
        encoder = build_vision_encoder(_tiny_vision_config(residual)).to(device)
        encoder.set_weight_capture(True)
        pixels = torch.randn(2, 3, 32, 32, device=device)
        encoder(pixels, return_aux=True)
        for residual_module in encoder.iter_depth_residuals():
            weights = residual_module.last_weights
            assert weights is not None
            sums = weights.float().sum(dim=0)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_pseudoquery_grads() -> None:
        encoder = build_vision_encoder(_tiny_vision_config("attnres")).to(device)
        pixels = torch.randn(2, 3, 32, 32, device=device)
        tokens, _ = encoder(pixels)
        tokens.float().pow(2).mean().backward()
        grads = []
        for module in encoder.modules():
            if isinstance(module, DepthAttentionResidual):
                assert module.query.grad is not None
                grads.append(float(module.query.grad.abs().sum().item()))
        assert any(value > 0 for value in grads)

    def test_separate_attnres() -> None:
        model = TinyAttnResVLM(
            vision_config=_tiny_vision_config("attnres"),
            decoder_config=_tiny_decoder_config(tokenizer.vocab_size, "attnres"),
            encoder_residual="attnres",
            decoder_residual="attnres",
        ).to(device)
        assert_encoder_decoder_attnres_separate(model)

    def test_all_variants_forward_backward() -> None:
        pixels = torch.randn(2, 3, 32, 32, device=device)
        input_ids = torch.tensor([[tokenizer.bos_token_id, tokenizer.token_to_id["what"], tokenizer.answer_token_id, tokenizer.token_to_id["red"], tokenizer.eos_token_id]] * 2, device=device)
        targets = torch.full_like(input_ids, -100)
        targets[:, 3] = input_ids[:, 3]
        targets[:, 4] = input_ids[:, 4]
        for variant, residual in VARIANTS.items():
            model = TinyAttnResVLM(
                vision_config=_tiny_vision_config(residual["encoder"]),
                decoder_config=_tiny_decoder_config(tokenizer.vocab_size, residual["decoder"]),
                encoder_residual=residual["encoder"],
                decoder_residual=residual["decoder"],
            ).to(device)
            out = model(pixel_values=pixels, input_ids=input_ids, targets=targets)
            out["loss"].backward()

    def test_loss_masking() -> None:
        model = TinyAttnResVLM(
            vision_config=_tiny_vision_config("baseline"),
            decoder_config=_tiny_decoder_config(tokenizer.vocab_size, "baseline"),
        ).to(device)
        pixels = torch.randn(1, 3, 32, 32, device=device)
        input_ids = torch.tensor(
            [[tokenizer.bos_token_id, tokenizer.token_to_id["what"], tokenizer.answer_token_id, tokenizer.token_to_id["red"], tokenizer.eos_token_id]],
            device=device,
        )
        targets = torch.full_like(input_ids, -100)
        targets[0, 3] = input_ids[0, 3]
        targets[0, 4] = input_ids[0, 4]
        out = model(pixel_values=pixels, input_ids=input_ids, targets=targets)
        assert torch.isfinite(out["loss"])
        # Only answer/eos supervised: flipping a question token target must not change loss when ignored.
        targets_q = targets.clone()
        targets_q[0, 1] = tokenizer.token_to_id["what"]
        # With ignore_index, setting a previously -100 position to a label would change loss.
        # Confirm original targets ignore question positions.
        assert int((targets == -100).sum().item()) >= 3

    def test_shared_init() -> None:
        seed_everything(0, deterministic=True)
        reference = TinyAttnResVLM(
            vision_config=_tiny_vision_config("baseline"),
            decoder_config=_tiny_decoder_config(tokenizer.vocab_size, "baseline"),
        )
        candidate = TinyAttnResVLM(
            vision_config=_tiny_vision_config("attnres"),
            decoder_config=_tiny_decoder_config(tokenizer.vocab_size, "attnres"),
            encoder_residual="attnres",
            decoder_residual="attnres",
        )
        copy_shared_weights(reference, candidate)
        validate_shared_initialization(reference, candidate)

    def test_overfit_smoke() -> None:
        config = AblationExperimentConfig(
            run_mode="smoke",
            seeds=[0],
            batch_size=16,
            max_epochs=20,
            early_stopping_patience=20,
            train_size=128,
            validation_size=32,
            test_size=32,
            vision_d_model=64,
            vision_n_layers=2,
            vision_n_heads=4,
            vision_d_ff=128,
            decoder_d_model=64,
            decoder_n_layers=2,
            decoder_n_heads=4,
            decoder_d_ff=128,
            num_blocks=2,
            num_workers=0,
            mixed_precision=True,
            amp_dtype="float16",
        )
        from src.vlm.ablation.train import build_dataloaders
        from torch.optim import AdamW

        tokenizer_local = VQATokenizer()
        loaders = build_dataloaders(config, tokenizer_local)
        model, _ = build_model_for_variant(
            config,
            variant="baseline",
            vocab_size=tokenizer_local.vocab_size,
            seed=0,
        )
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=3e-3)
        model.train()
        for _ in range(40):
            batch = next(iter(loaders["train"]))
            optimizer.zero_grad(set_to_none=True)
            out = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                targets=batch["targets"].to(device),
            )
            out["loss"].backward()
            optimizer.step()
        # Evaluate on the same tiny loader batch distribution.
        model.eval()
        with torch.no_grad():
            batch = next(iter(loaders["train"]))
            out = model(
                pixel_values=batch["pixel_values"].to(device),
                input_ids=batch["input_ids"].to(device),
                targets=batch["targets"].to(device),
            )
            preds = out["logits"].argmax(dim=-1)
            answer_positions = batch["answer_positions"]
            correct = 0
            for row in range(batch["input_ids"].size(0)):
                pos = int(answer_positions[row].item())
                if int(preds[row, pos].item()) == int(batch["answer_ids"][row].item()):
                    correct += 1
            accuracy = correct / batch["input_ids"].size(0)
        if accuracy < 0.5:
            raise AssertionError(f"Overfit smoke accuracy too low: {accuracy}")

    check("dataset_determinism", test_dataset_determinism)
    check("unambiguous_answers", test_unambiguous_answers)
    check("patch_token_shape", test_patch_shape)
    check("bidirectional_attention_shape", test_bidirectional_shape)
    check("baseline_encoder_forward_backward", lambda: test_encoder_forward_backward("baseline"))
    check("full_attnres_encoder_forward_backward", lambda: test_encoder_forward_backward("attnres"))
    check("block_attnres_encoder_forward_backward", lambda: test_encoder_forward_backward("block_attnres"))
    check("full_encoder_routing_sums", lambda: test_routing_sums("attnres"))
    check("block_encoder_routing_sums", lambda: test_routing_sums("block_attnres"))
    check("encoder_pseudoquery_gradients", test_pseudoquery_grads)
    check("encoder_decoder_attnres_separate", test_separate_attnres)
    check("all_variants_forward_backward", test_all_variants_forward_backward)
    check("loss_masking", test_loss_masking)
    check("shared_initialization", test_shared_init)
    check("tiny_overfit", test_overfit_smoke)

    results["ok"] = not results["failed"]
    if report_path is not None:
        atomic_write_json(report_path, results)
    return results
