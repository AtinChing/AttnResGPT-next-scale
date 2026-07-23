from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.vlm.ablation.config import AblationExperimentConfig, config_hash, resolve_experiment_config
from src.vlm.ablation.source_hash import compare_source_hashes, hash_source_tree
from src.vlm.clevr.official import CLEVR_SUBSETS, CLEVR_V1_NO_IMAGES
from src.vlm.clevr.preprocess import PreprocessConfig, resize_and_pad
from src.vlm.clevr.programs import analyze_program
from src.vlm.clevr.subsets import build_clevr_subset_manifest, build_cogent_subset_manifest
from src.vlm.clevr.tokenizer import CLEVRTokenizer, tokenize_clevr_text
from src.vlm.synthetic_vqa import generate_example


def _fake_questions(prefix: str, n_images: int, questions_per_image: int = 2) -> list[dict]:
    items = []
    q_index = 0
    for image_index in range(n_images):
        for _ in range(questions_per_image):
            items.append(
                {
                    "question_index": q_index,
                    "image_index": image_index,
                    "image_filename": f"{prefix}_{image_index:06d}.png",
                    "question": "How many small red spheres are there?",
                    "answer": "2",
                    "question_family_index": 3,
                    "program": [
                        {"function": "scene", "inputs": [], "value_inputs": []},
                        {"function": "filter_color", "inputs": [0], "value_inputs": ["red"]},
                        {"function": "count", "inputs": [1], "value_inputs": []},
                    ],
                    "split": prefix.split("_")[-1],
                }
            )
            q_index += 1
    return items


def test_tokenize_clevr_text() -> None:
    assert tokenize_clevr_text("How many spheres?") == ["how", "many", "spheres"]


def test_clevr_tokenizer_unk_and_supervised() -> None:
    tok = CLEVRTokenizer.build_from_training_questions(
        [{"question": "how many spheres", "answer": "2"}]
    )
    encoded = tok.encode_supervised("how many spheres unseen", "2", allow_unk=True)
    assert tok.unk_token_id in encoded["input_ids"]
    assert encoded["targets"].count(-100) >= 3


def test_preprocess_deterministic_pad() -> None:
    image = Image.fromarray(np.zeros((90, 160, 3), dtype=np.uint8) + 120)
    cfg = PreprocessConfig(image_size=128)
    a = resize_and_pad(image, config=cfg)
    b = resize_and_pad(image, config=cfg)
    assert a.shape == (3, 128, 128)
    assert np.allclose(a, b)


def test_program_analysis_categories() -> None:
    stats = analyze_program(
        [
            {"function": "scene", "inputs": [], "value_inputs": []},
            {"function": "exist", "inputs": [0], "value_inputs": []},
        ]
    )
    assert stats.reasoning_category == "existence"
    assert stats.program_length_bin == "1-5"


def test_subset_manifests_fixed_and_non_overlapping() -> None:
    smoke = CLEVR_SUBSETS["smoke"]
    train = _fake_questions("CLEVR_train", smoke["train_images"] + 20)
    val = _fake_questions("CLEVR_val", smoke["validation_images"] + smoke["test_images"] + 20)
    manifest = build_clevr_subset_manifest(train_questions=train, val_questions=val, mode="smoke", subset_seed=7)
    again = build_clevr_subset_manifest(train_questions=train, val_questions=val, mode="smoke", subset_seed=7)
    assert manifest["manifest_hash"] == again["manifest_hash"]
    assert not (
        set(manifest["splits"]["validation"]["image_indices"])
        & set(manifest["splits"]["test"]["image_indices"])
    )
    cogent = build_cogent_subset_manifest(
        train_a_questions=train,
        val_a_questions=_fake_questions("CLEVR_valA", smoke["validation_images"] + 5),
        val_b_questions=_fake_questions("CLEVR_valB", smoke["test_images"] + 5),
        mode="smoke",
        subset_seed=7,
    )
    assert cogent["splits"]["train"]["source_split"] == "trainA"
    assert cogent["splits"]["test"]["source_split"] == "valB"


def test_config_defaults_clevr() -> None:
    cfg = resolve_experiment_config(AblationExperimentConfig(benchmark_mode="quick"))
    assert cfg.image_size == 128
    assert cfg.patch_size == 16
    assert cfg.vision_n_layers == 10
    assert cfg.decoder_n_layers == 10
    assert cfg.num_blocks == 5
    assert cfg.max_seq_len >= 64 + cfg.text_context_budget
    h1 = config_hash(cfg)
    cfg.subset_manifest_hash = "abc"
    assert config_hash(cfg) != h1


def test_official_urls_are_fbaipublicfiles() -> None:
    assert CLEVR_V1_NO_IMAGES.url.startswith("https://dl.fbaipublicfiles.com/clevr/")


def test_tinyshapes_remains_smoke_only() -> None:
    example = generate_example(split="train", split_seed=2, example_index=0, image_size=64)
    assert example.image.shape == (3, 64, 64)


def test_source_hash_self_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    hashes = hash_source_tree(root)
    assert "src/vlm/clevr/prepare.py" in hashes
    assert compare_source_hashes(root, root)["match"] is True


def test_clevr_model_shape_cpu() -> None:
    from src.models.vlm_attnres import TinyAttnResVLM
    from src.vlm.ablation.config import build_decoder_config, build_vision_config

    cfg = resolve_experiment_config(
        AblationExperimentConfig(
            benchmark_mode="smoke",
            vision_n_layers=2,
            decoder_n_layers=2,
            num_blocks=2,
            batch_size=2,
        )
    )
    # Keep Mac CPU test tiny.
    cfg.vision_n_layers = 2
    cfg.decoder_n_layers = 2
    cfg.num_blocks = 2
    tok = CLEVRTokenizer.build_from_training_questions(
        [{"question": "how many spheres", "answer": "2"}]
    )
    model = TinyAttnResVLM(
        vision_config=build_vision_config(cfg, "baseline"),
        decoder_config=build_decoder_config(cfg, "baseline", tok.vocab_size),
    )
    encoded = tok.encode_supervised("how many spheres", "2")
    input_ids = torch.tensor([encoded["input_ids"]])
    targets = torch.tensor([encoded["targets"]])
    out = model(pixel_values=torch.randn(1, 3, cfg.image_size, cfg.image_size), input_ids=input_ids, targets=targets)
    assert torch.isfinite(out["loss"])
