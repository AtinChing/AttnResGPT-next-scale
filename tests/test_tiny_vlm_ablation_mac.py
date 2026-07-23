from __future__ import annotations

from pathlib import Path

import torch

from src.models.baseline import BidirectionalSelfAttention, CausalSelfAttention
from src.models.vision_attnres import VisionConfig, build_vision_encoder
from src.models.vlm_attnres import TinyAttnResVLM
from src.utils.config import AttnResConfig, ModelConfig
from src.vlm.ablation.init_sync import copy_shared_weights, validate_shared_initialization
from src.vlm.ablation.source_hash import compare_source_hashes, hash_source_tree
from src.vlm.synthetic_vqa import VQATokenizer, generate_example


def test_bidirectional_attention_shape_cpu() -> None:
    attn = BidirectionalSelfAttention(32, 4, 0.0)
    x = torch.randn(2, 8, 32)
    out, _ = attn(x)
    assert out.shape == x.shape


def test_causal_attention_unchanged_api() -> None:
    attn = CausalSelfAttention(32, 4, 0.0, max_seq_len=16)
    x = torch.randn(2, 8, 32)
    out, _ = attn(x)
    assert out.shape == x.shape


def test_vision_encoders_tiny_forward_cpu() -> None:
    for residual in ("baseline", "attnres", "block_attnres"):
        config = VisionConfig(
            image_size=32,
            patch_size=8,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
            residual=residual,  # type: ignore[arg-type]
            attnres=AttnResConfig(
                enabled=residual != "baseline",
                mode="block" if residual == "block_attnres" else "full",
                num_blocks=2 if residual == "block_attnres" else None,
            ),
        )
        encoder = build_vision_encoder(config)
        tokens, aux = encoder(torch.randn(2, 3, 32, 32), return_aux=True)
        assert tokens.shape == (2, 16, 32)
        if residual != "baseline":
            assert aux["depth_attention_rows"]


def test_tiny_vlm_one_forward_backward_cpu() -> None:
    tokenizer = VQATokenizer()
    vision = VisionConfig(
        image_size=32,
        patch_size=8,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        residual="attnres",
        attnres=AttnResConfig(enabled=True, mode="full"),
    )
    decoder = ModelConfig(
        architecture="attnres",
        size_name="small",
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        tie_weights=False,
        attnres=AttnResConfig(enabled=True, mode="full"),
    )
    model = TinyAttnResVLM(
        vision_config=vision,
        decoder_config=decoder,
        encoder_residual="attnres",
        decoder_residual="attnres",
    )
    input_ids = torch.tensor(
        [[tokenizer.bos_token_id, tokenizer.token_to_id["what"], tokenizer.answer_token_id, tokenizer.token_to_id["red"], tokenizer.eos_token_id]]
    )
    targets = torch.full_like(input_ids, -100)
    targets[0, 3] = input_ids[0, 3]
    out = model(pixel_values=torch.randn(1, 3, 32, 32), input_ids=input_ids, targets=targets)
    out["loss"].backward()
    assert out["logits"].shape[0] == 1


def test_synthetic_vqa_determinism_and_small_sample() -> None:
    a = generate_example(split="train", split_seed=3, example_index=1, image_size=64)
    b = generate_example(split="train", split_seed=3, example_index=1, image_size=64)
    assert a.question == b.question
    assert a.answer == b.answer
    assert a.image.shape == (3, 64, 64)
    # Cap local generation well under the Colab dataset sizes.
    for index in range(8):
        example = generate_example(split="train", split_seed=5, example_index=index, image_size=64)
        assert example.answer


def test_shared_init_validation() -> None:
    tokenizer = VQATokenizer()
    vision_base = VisionConfig(image_size=32, patch_size=8, d_model=32, n_layers=2, n_heads=4, d_ff=64, residual="baseline")
    decoder_base = ModelConfig(
        architecture="baseline",
        size_name="small",
        vocab_size=tokenizer.vocab_size,
        max_seq_len=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        tie_weights=False,
    )
    torch.manual_seed(0)
    reference = TinyAttnResVLM(vision_config=vision_base, decoder_config=decoder_base)
    candidate = TinyAttnResVLM(
        vision_config=VisionConfig(
            image_size=32,
            patch_size=8,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            residual="attnres",
            attnres=AttnResConfig(enabled=True),
        ),
        decoder_config=ModelConfig(
            architecture="attnres",
            size_name="small",
            vocab_size=tokenizer.vocab_size,
            max_seq_len=64,
            d_model=32,
            n_layers=2,
            n_heads=4,
            d_ff=64,
            dropout=0.0,
            tie_weights=False,
            attnres=AttnResConfig(enabled=True),
        ),
        encoder_residual="attnres",
        decoder_residual="attnres",
    )
    copy_shared_weights(reference, candidate)
    validate_shared_initialization(reference, candidate)


def test_source_hash_self_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    hashes = hash_source_tree(root)
    assert hashes
    comparison = compare_source_hashes(root, root)
    assert comparison["match"] is True
