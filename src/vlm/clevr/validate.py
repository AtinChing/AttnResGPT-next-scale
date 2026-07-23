from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.vlm.ablation.io_utils import atomic_write_json
from src.vlm.clevr.official import COGENT_IMAGE_PREFIX, CLEVR_IMAGE_PREFIX
from src.vlm.clevr.subsets import select_examples


def validate_benchmark_bundle(
    *,
    benchmark: str,
    manifest: dict[str, Any],
    question_tables: dict[str, list[dict[str, Any]]],
    image_root: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    split_image_sets: dict[str, set[int]] = {}
    split_question_sets: dict[str, set[int]] = {}

    for split_name, split in manifest["splits"].items():
        source = split["source_split"]
        questions = question_tables[source]
        selected = select_examples(questions, split["question_indices"])
        by_q = {int(item["question_index"]): item for item in selected}
        image_indices = set(split["image_indices"])
        split_image_sets[split_name] = set(image_indices)
        split_question_sets[split_name] = set(split["question_indices"])

        if benchmark == "clevr_v1":
            prefix = CLEVR_IMAGE_PREFIX[source]
        else:
            prefix = COGENT_IMAGE_PREFIX[source]

        for item in selected:
            q_index = int(item["question_index"])
            image_index = int(item["image_index"])
            filename = str(item["image_filename"])
            if image_index not in image_indices:
                errors.append(f"{split_name}: question {q_index} image_index not in split images")
            if filename not in split["image_filenames"]:
                errors.append(f"{split_name}: question {q_index} filename missing from manifest")
            image_path = Path(image_root) / prefix / filename
            if not image_path.exists():
                errors.append(f"Missing image file: {image_path}")
            # Official answer preserved.
            if "answer" not in item or item["answer"] is None or item["answer"] == "":
                errors.append(f"{split_name}: empty answer for question {q_index}")
            if int(by_q[q_index]["image_index"]) != image_index:
                errors.append(f"{split_name}: image_index mismatch for question {q_index}")

        # Every selected image has at least one question.
        images_with_q = {int(item["image_index"]) for item in selected}
        orphan_images = image_indices - images_with_q
        if orphan_images:
            errors.append(f"{split_name}: {len(orphan_images)} images have no selected questions")

    if benchmark == "clevr_v1":
        if split_image_sets["validation"] & split_image_sets["test"]:
            errors.append("Standard CLEVR validation/test image subsets overlap")
        if split_question_sets["validation"] & split_question_sets["test"]:
            errors.append("Standard CLEVR validation/test question subsets overlap")
        train_files = set(manifest["splits"]["train"]["image_filenames"])
        val_files = set(manifest["splits"]["validation"]["image_filenames"])
        test_files = set(manifest["splits"]["test"]["image_filenames"])
        if train_files & (val_files | test_files):
            errors.append("Standard CLEVR train filenames overlap val/test")
    else:
        train_files = set(manifest["splits"]["train"]["image_filenames"])
        test_files = set(manifest["splits"]["test"]["image_filenames"])
        if any(name.startswith("CLEVR_valB_") or name.startswith("CLEVR_testB_") for name in train_files):
            errors.append("CoGenT Condition B images leaked into training")
        if train_files & test_files:
            errors.append("CoGenT train/test image filename overlap")
        if manifest["splits"]["test"]["source_split"] != "valB":
            errors.append("CoGenT test split must be valB")
        if manifest["splits"]["train"]["source_split"] != "trainA":
            errors.append("CoGenT train split must be trainA")

    report = {
        "benchmark": benchmark,
        "manifest_hash": manifest.get("manifest_hash"),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "split_sizes": {
            name: {
                "n_images": split["n_images"],
                "n_questions": split["n_questions"],
                "source_split": split["source_split"],
            }
            for name, split in manifest["splits"].items()
        },
    }
    if report_path is not None:
        atomic_write_json(report_path, report)
    if errors:
        raise RuntimeError(f"CLEVR dataset validation failed: {errors[:5]}")
    return report


def majority_answer_baseline(train_examples: list[dict[str, Any]], eval_examples: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(item["answer"]) for item in train_examples)
    majority, _ = counts.most_common(1)[0]
    correct = sum(1 for item in eval_examples if str(item["answer"]) == majority)
    return {
        "majority_answer": majority,
        "train_answer_support": counts[majority],
        "accuracy": correct / max(1, len(eval_examples)),
        "correct": correct,
        "total": len(eval_examples),
    }
