from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.vlm.ablation.io_utils import atomic_write_json, ensure_dir
from src.vlm.clevr.download import download_official_archive, verify_archive
from src.vlm.clevr.extract import extract_selected_members
from src.vlm.clevr.official import (
    COGENT_IMAGE_PREFIX,
    COGENT_ROOT_NAME,
    COGENT_V1_FULL,
    COGENT_V1_NO_IMAGES,
    CLEVR_IMAGE_PREFIX,
    CLEVR_ROOT_NAME,
    CLEVR_V1_FULL,
    CLEVR_V1_NO_IMAGES,
)
from src.vlm.clevr.preprocess import PreprocessConfig
from src.vlm.clevr.subsets import (
    build_clevr_subset_manifest,
    build_cogent_subset_manifest,
    extract_json_member,
    image_zip_members_for_manifest,
    load_or_create_manifest,
    load_questions_json,
    select_examples,
)
from src.vlm.clevr.tokenizer import CLEVRTokenizer
from src.vlm.clevr.validate import validate_benchmark_bundle

BenchmarkName = Literal["clevr_v1", "clevr_cogent_v1"]


@dataclass
class PreparedBenchmark:
    benchmark: BenchmarkName
    dataset_version: str
    mode: str
    data_root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    tokenizer_path: Path
    tokenizer: CLEVRTokenizer
    preprocess: PreprocessConfig
    image_root: Path
    question_tables: dict[str, list[dict[str, Any]]]
    split_examples: dict[str, list[dict[str, Any]]]
    image_prefix_by_split: dict[str, str]
    validation_report: dict[str, Any]
    download_report: dict[str, Any]

    def subset_manifest_hash(self) -> str:
        return str(self.manifest["manifest_hash"])

    def vocab_hash(self) -> str:
        return self.tokenizer.vocab_hash()

    def to_meta(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "dataset_version": self.dataset_version,
            "mode": self.mode,
            "manifest_path": str(self.manifest_path),
            "manifest_hash": self.subset_manifest_hash(),
            "tokenizer_path": str(self.tokenizer_path),
            "vocab_hash": self.vocab_hash(),
            "preprocess": self.preprocess.to_dict(),
            "preprocess_hash": self.preprocess.config_hash(),
            "split_sizes": {
                name: {"n_images": split["n_images"], "n_questions": split["n_questions"]}
                for name, split in self.manifest["splits"].items()
            },
            "download_report": self.download_report,
            "validation_report": self.validation_report,
        }


def _progress_printer(label: str):
    last = {"pct": -1}

    def callback(written: int, total: int) -> None:
        pct = int(100 * written / max(1, total))
        if pct >= last["pct"] + 1 or written == total:
            last["pct"] = pct
            print(f"[download] {label}: {pct}% ({written}/{total} bytes)", flush=True)

    return callback


def prepare_benchmark(
    *,
    project_root: Path,
    benchmark: BenchmarkName,
    mode: str,
    subset_seed: int = 17,
    preprocess: PreprocessConfig | None = None,
    validation_report_path: Path | None = None,
) -> PreparedBenchmark:
    project_root = Path(project_root)
    data_root = ensure_dir(project_root / "data" / benchmark)
    archives_dir = ensure_dir(data_root / "archives")
    meta_dir = ensure_dir(data_root / "metadata")
    images_dir = ensure_dir(data_root / "images")
    manifests_dir = ensure_dir(data_root / "manifests")
    preprocess = preprocess or PreprocessConfig()
    download_report: dict[str, Any] = {}

    if benchmark == "clevr_v1":
        no_images = CLEVR_V1_NO_IMAGES
        full = CLEVR_V1_FULL
        root_name = CLEVR_ROOT_NAME
        question_members = {
            "train": f"{root_name}/questions/CLEVR_train_questions.json",
            "val": f"{root_name}/questions/CLEVR_val_questions.json",
        }
    else:
        no_images = COGENT_V1_NO_IMAGES
        full = COGENT_V1_FULL
        root_name = COGENT_ROOT_NAME
        question_members = {
            "trainA": f"{root_name}/questions/CLEVR_trainA_questions.json",
            "valA": f"{root_name}/questions/CLEVR_valA_questions.json",
            "valB": f"{root_name}/questions/CLEVR_valB_questions.json",
        }

    print(f"Downloading official metadata archive: {no_images.url}")
    meta_zip = download_official_archive(
        no_images,
        archives_dir,
        progress_callback=_progress_printer(no_images.name),
    )
    download_report["metadata_archive"] = verify_archive(meta_zip, no_images)

    question_tables: dict[str, list[dict[str, Any]]] = {}
    for key, member in question_members.items():
        dest = meta_dir / Path(member).name
        extract_json_member(meta_zip, member, dest)
        question_tables[key] = load_questions_json(dest)

    manifest_path = manifests_dir / f"subset_{mode}_seed{subset_seed}.json"

    def _builder():
        if benchmark == "clevr_v1":
            return build_clevr_subset_manifest(
                train_questions=question_tables["train"],
                val_questions=question_tables["val"],
                mode=mode,
                subset_seed=subset_seed,
            )
        return build_cogent_subset_manifest(
            train_a_questions=question_tables["trainA"],
            val_a_questions=question_tables["valA"],
            val_b_questions=question_tables["valB"],
            mode=mode,
            subset_seed=subset_seed,
        )

    # Once created, never resample automatically.
    manifest = load_or_create_manifest(manifest_path, _builder)
    print(f"Using subset manifest {manifest_path} hash={manifest['manifest_hash']}")

    print(f"Downloading official image archive: {full.url}")
    print("This is large; downloads resume after disconnects.")
    image_zip = download_official_archive(
        full,
        archives_dir,
        progress_callback=_progress_printer(full.name),
    )
    download_report["image_archive"] = verify_archive(image_zip, full, compute_sha256=False)

    members = image_zip_members_for_manifest(benchmark, manifest)
    extract_stats = extract_selected_members(image_zip, members, images_dir, skip_existing=True)
    download_report["extract"] = extract_stats
    print(f"Selective extract: {extract_stats}")

    # Vocabulary from official training questions (full train split annotations).
    tokenizer_path = manifests_dir / f"tokenizer_{benchmark}.json"
    if tokenizer_path.exists():
        tokenizer = CLEVRTokenizer.load(tokenizer_path)
    else:
        train_key = "train" if benchmark == "clevr_v1" else "trainA"
        tokenizer = CLEVRTokenizer.build_from_training_questions(question_tables[train_key])
        tokenizer.save(tokenizer_path)

    split_examples: dict[str, list[dict[str, Any]]] = {}
    image_prefix_by_split: dict[str, str] = {}
    for split_name, split in manifest["splits"].items():
        source = split["source_split"]
        split_examples[split_name] = select_examples(question_tables[source], split["question_indices"])
        if benchmark == "clevr_v1":
            image_prefix_by_split[split_name] = CLEVR_IMAGE_PREFIX[source]
        else:
            image_prefix_by_split[split_name] = COGENT_IMAGE_PREFIX[source]

    report_path = validation_report_path or (project_root / "summaries" / "clevr_dataset_validation.json")
    # Keep per-benchmark reports without clobbering the other.
    if validation_report_path is None:
        report_path = project_root / "summaries" / f"{benchmark}_dataset_validation.json"
    ensure_dir(report_path.parent)
    validation_report = validate_benchmark_bundle(
        benchmark=benchmark,
        manifest=manifest,
        question_tables=question_tables,
        image_root=images_dir,
        report_path=report_path,
    )
    # Also write the path requested by the user for the active/latest validation.
    atomic_write_json(project_root / "summaries" / "clevr_dataset_validation.json", validation_report)

    prepared = PreparedBenchmark(
        benchmark=benchmark,
        dataset_version=manifest["dataset_version"],
        mode=mode,
        data_root=data_root,
        manifest_path=manifest_path,
        manifest=manifest,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        preprocess=preprocess,
        image_root=images_dir,
        question_tables=question_tables,
        split_examples=split_examples,
        image_prefix_by_split=image_prefix_by_split,
        validation_report=validation_report,
        download_report=download_report,
    )
    atomic_write_json(manifests_dir / f"prepared_{mode}_{manifest['manifest_hash']}.json", prepared.to_meta())
    return prepared
