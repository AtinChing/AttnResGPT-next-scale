from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.vlm.ablation.io_utils import atomic_write_json, ensure_dir
from src.vlm.clevr.official import CLEVR_SUBSETS, COGENT_IMAGE_PREFIX, CLEVR_IMAGE_PREFIX

BenchmarkName = Literal["clevr_v1", "clevr_cogent_v1"]


def load_questions_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["questions"])


def extract_json_member(zip_path: Path, member: str, dest_path: Path) -> Path:
    dest_path = Path(dest_path)
    ensure_dir(dest_path.parent)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member) as source, dest_path.open("wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
    return dest_path


def _unique_sorted_image_indices(questions: list[dict[str, Any]]) -> list[int]:
    return sorted({int(item["image_index"]) for item in questions})


def _questions_for_images(
    questions: list[dict[str, Any]],
    image_indices: set[int],
) -> list[dict[str, Any]]:
    return [item for item in questions if int(item["image_index"]) in image_indices]


def _sample_indices(pool: list[int], count: int, rng: np.random.Generator) -> list[int]:
    if count > len(pool):
        raise ValueError(f"Requested {count} images but only {len(pool)} available")
    chosen = rng.choice(pool, size=count, replace=False)
    return sorted(int(value) for value in chosen)


def manifest_hash(payload: dict[str, Any]) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def build_clevr_subset_manifest(
    *,
    train_questions: list[dict[str, Any]],
    val_questions: list[dict[str, Any]],
    mode: str,
    subset_seed: int,
) -> dict[str, Any]:
    sizes = CLEVR_SUBSETS[mode]
    rng = np.random.default_rng(subset_seed)
    train_pool = _unique_sorted_image_indices(train_questions)
    val_pool = _unique_sorted_image_indices(val_questions)
    train_images = _sample_indices(train_pool, sizes["train_images"], rng)
    held = _sample_indices(
        val_pool,
        sizes["validation_images"] + sizes["test_images"],
        rng,
    )
    val_images = held[: sizes["validation_images"]]
    test_images = held[sizes["validation_images"] :]
    assert not (set(val_images) & set(test_images))

    def pack(split: str, source_split: str, images: list[int], questions: list[dict[str, Any]]):
        selected = _questions_for_images(questions, set(images))
        return {
            "split": split,
            "source_split": source_split,
            "image_indices": images,
            "question_indices": [int(item["question_index"]) for item in selected],
            "n_images": len(images),
            "n_questions": len(selected),
            "image_filenames": sorted({str(item["image_filename"]) for item in selected}),
        }

    body = {
        "benchmark": "clevr_v1",
        "dataset_version": "CLEVR_v1.0",
        "mode": mode,
        "subset_seed": subset_seed,
        "note": (
            "Reported test accuracy uses a held-out subset of the official validation split; "
            "official CLEVR test answers are not used."
        ),
        "splits": {
            "train": pack("train", "train", train_images, train_questions),
            "validation": pack("validation", "val", val_images, val_questions),
            "test": pack("test", "val", test_images, val_questions),
        },
    }
    body["manifest_hash"] = manifest_hash({k: v for k, v in body.items() if k != "manifest_hash"})
    return body


def build_cogent_subset_manifest(
    *,
    train_a_questions: list[dict[str, Any]],
    val_a_questions: list[dict[str, Any]],
    val_b_questions: list[dict[str, Any]],
    mode: str,
    subset_seed: int,
) -> dict[str, Any]:
    sizes = CLEVR_SUBSETS[mode]
    rng = np.random.default_rng(subset_seed)
    train_images = _sample_indices(
        _unique_sorted_image_indices(train_a_questions),
        sizes["train_images"],
        rng,
    )
    val_images = _sample_indices(
        _unique_sorted_image_indices(val_a_questions),
        sizes["validation_images"],
        rng,
    )
    test_images = _sample_indices(
        _unique_sorted_image_indices(val_b_questions),
        sizes["test_images"],
        rng,
    )

    def pack(split: str, source_split: str, images: list[int], questions: list[dict[str, Any]]):
        selected = _questions_for_images(questions, set(images))
        return {
            "split": split,
            "source_split": source_split,
            "image_indices": images,
            "question_indices": [int(item["question_index"]) for item in selected],
            "n_images": len(images),
            "n_questions": len(selected),
            "image_filenames": sorted({str(item["image_filename"]) for item in selected}),
        }

    body = {
        "benchmark": "clevr_cogent_v1",
        "dataset_version": "CLEVR_CoGenT_v1.0",
        "mode": mode,
        "subset_seed": subset_seed,
        "note": (
            "Train/val use Condition A; reported test is Condition B validation "
            "(compositional generalization). Condition B is never mixed into training."
        ),
        "splits": {
            "train": pack("train", "trainA", train_images, train_a_questions),
            "validation": pack("validation", "valA", val_images, val_a_questions),
            "test": pack("test", "valB", test_images, val_b_questions),
        },
    }
    body["manifest_hash"] = manifest_hash({k: v for k, v in body.items() if k != "manifest_hash"})
    return body


def load_or_create_manifest(path: Path, builder) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    manifest = builder()
    ensure_dir(path.parent)
    atomic_write_json(path, manifest)
    return manifest


def image_zip_members_for_manifest(benchmark: BenchmarkName, manifest: dict[str, Any]) -> list[str]:
    members: list[str] = []
    for split_name, split in manifest["splits"].items():
        source = split["source_split"]
        if benchmark == "clevr_v1":
            prefix = CLEVR_IMAGE_PREFIX[source]
        else:
            prefix = COGENT_IMAGE_PREFIX[source]
        for filename in split["image_filenames"]:
            members.append(prefix + filename)
    return sorted(set(members))


def questions_by_index(questions: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["question_index"]): item for item in questions}


def select_examples(
    questions: list[dict[str, Any]],
    question_indices: list[int],
) -> list[dict[str, Any]]:
    by_index = questions_by_index(questions)
    missing = [index for index in question_indices if index not in by_index]
    if missing:
        raise KeyError(f"Missing question indices: {missing[:10]}")
    return [by_index[index] for index in question_indices]


def group_questions_by_image(questions: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in questions:
        grouped[int(item["image_index"])].append(item)
    return grouped
