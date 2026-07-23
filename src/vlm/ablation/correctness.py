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
from src.vlm.ablation.config import VARIANTS
from src.vlm.ablation.init_sync import (
    assert_encoder_decoder_attnres_separate,
    copy_shared_weights,
    validate_shared_initialization,
)
from src.vlm.ablation.io_utils import atomic_write_json
from src.vlm.clevr.programs import analyze_program
from src.vlm.clevr.subsets import build_clevr_subset_manifest, build_cogent_subset_manifest
from src.vlm.clevr.tokenizer import CLEVRTokenizer
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
        max_seq_len=96,
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

    tokenizer = CLEVRTokenizer.build_from_training_questions(
        [
            {"question": "How many small spheres are there?", "answer": "2"},
            {"question": "What color is the metal cube?", "answer": "red"},
        ]
    )

    def test_tinyshapes_smoke_only() -> None:
        # TinyShapes remains only as an implementation smoke check.
        example = generate_example(split="train", split_seed=1, example_index=0, image_size=64)
        assert example.answer
        tok = VQATokenizer()
        tok.encode_supervised(example.question, example.answer)

    def test_clevr_tokenizer_unk() -> None:
        ids = tokenizer.encode("what color is the unseenword cube", allow_unk=True)
        assert tokenizer.unk_token_id in ids

    def test_program_stats() -> None:
        program = [
            {"function": "scene", "inputs": [], "value_inputs": []},
            {"function": "filter_color", "inputs": [0], "value_inputs": ["red"]},
            {"function": "count", "inputs": [1], "value_inputs": []},
        ]
        stats = analyze_program(program)
        assert stats.n_operations == 3
        assert stats.reasoning_category == "counting"
        assert stats.dependency_depth == 3

    def test_subset_manifest_no_overlap() -> None:
        train = [
            {
                "question_index": i,
                "image_index": i % 20,
                "image_filename": f"CLEVR_train_{i % 20:06d}.png",
                "question": "how many",
                "answer": "1",
                "question_family_index": 0,
                "program": [],
                "split": "train",
            }
            for i in range(100)
        ]
        val = [
            {
                "question_index": i,
                "image_index": i % 40,
                "image_filename": f"CLEVR_val_{i % 40:06d}.png",
                "question": "how many",
                "answer": "1",
                "question_family_index": 0,
                "program": [],
                "split": "val",
            }
            for i in range(200)
        ]
        # Use smoke sizes via temporary override by building with enough images.
        from src.vlm.clevr.official import CLEVR_SUBSETS

        # smoke needs 500 train images - generate denser fake pool
        train = [
            {
                "question_index": i,
                "image_index": i,
                "image_filename": f"CLEVR_train_{i:06d}.png",
                "question": "how many",
                "answer": "1",
                "question_family_index": 0,
                "program": [],
                "split": "train",
            }
            for i in range(CLEVR_SUBSETS["smoke"]["train_images"] + 10)
        ]
        val = [
            {
                "question_index": i,
                "image_index": i,
                "image_filename": f"CLEVR_val_{i:06d}.png",
                "question": "how many",
                "answer": "1",
                "question_family_index": 0,
                "program": [],
                "split": "val",
            }
            for i in range(
                CLEVR_SUBSETS["smoke"]["validation_images"] + CLEVR_SUBSETS["smoke"]["test_images"] + 10
            )
        ]
        manifest = build_clevr_subset_manifest(
            train_questions=train,
            val_questions=val,
            mode="smoke",
            subset_seed=0,
        )
        assert not (
            set(manifest["splits"]["validation"]["image_indices"])
            & set(manifest["splits"]["test"]["image_indices"])
        )
        cogent = build_cogent_subset_manifest(
            train_a_questions=train,
            val_a_questions=val[: CLEVR_SUBSETS["smoke"]["validation_images"] + 5],
            val_b_questions=[
                {
                    **item,
                    "image_filename": item["image_filename"].replace("val", "valB"),
                    "split": "valB",
                }
                for item in val[: CLEVR_SUBSETS["smoke"]["test_images"] + 5]
            ],
            mode="smoke",
            subset_seed=0,
        )
        assert cogent["splits"]["test"]["source_split"] == "valB"

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

    def test_all_variants_forward_backward() -> None:
        pixels = torch.randn(2, 3, 32, 32, device=device)
        input_ids = torch.tensor(
            [[tokenizer.bos_token_id, tokenizer.token_to_id["how"], tokenizer.answer_token_id, tokenizer.token_to_id["2"], tokenizer.eos_token_id]]
            * 2,
            device=device,
        )
        targets = torch.full_like(input_ids, -100)
        targets[:, 3] = input_ids[:, 3]
        targets[:, 4] = input_ids[:, 4]
        for _, residual in VARIANTS.items():
            model = TinyAttnResVLM(
                vision_config=_tiny_vision_config(residual["encoder"]),
                decoder_config=_tiny_decoder_config(tokenizer.vocab_size, residual["decoder"]),
                encoder_residual=residual["encoder"],
                decoder_residual=residual["decoder"],
            ).to(device)
            out = model(pixel_values=pixels, input_ids=input_ids, targets=targets)
            out["loss"].backward()

    def test_separate_attnres() -> None:
        model = TinyAttnResVLM(
            vision_config=_tiny_vision_config("attnres"),
            decoder_config=_tiny_decoder_config(tokenizer.vocab_size, "attnres"),
            encoder_residual="attnres",
            decoder_residual="attnres",
        ).to(device)
        assert_encoder_decoder_attnres_separate(model)

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

    def test_routing_sums() -> None:
        encoder = build_vision_encoder(_tiny_vision_config("attnres")).to(device)
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

    check("tinyshapes_smoke_only", test_tinyshapes_smoke_only)
    check("clevr_tokenizer_unk", test_clevr_tokenizer_unk)
    check("program_stats", test_program_stats)
    check("subset_manifest_no_overlap", test_subset_manifest_no_overlap)
    check("patch_token_shape", test_patch_shape)
    check("bidirectional_attention_shape", test_bidirectional_shape)
    check("all_variants_forward_backward", test_all_variants_forward_backward)
    check("encoder_decoder_attnres_separate", test_separate_attnres)
    check("shared_initialization", test_shared_init)
    check("full_encoder_routing_sums", test_routing_sums)
    check("encoder_pseudoquery_gradients", test_pseudoquery_grads)

    results["ok"] = not results["failed"]
    if report_path is not None:
        atomic_write_json(report_path, results)
    return results
