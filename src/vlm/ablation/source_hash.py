from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

# Canonical ablation-critical modules that must stay synchronized between the
# repository and the notebook-written Drive source tree.
CANONICAL_SOURCE_RELPATHS: tuple[str, ...] = (
    "src/models/baseline.py",
    "src/models/attnres.py",
    "src/models/vision_attnres.py",
    "src/models/vlm_attnres.py",
    "src/metrics/depth_metrics.py",
    "src/metrics/norms.py",
    "src/utils/config.py",
    "src/utils/runtime.py",
    "src/vlm/synthetic_vqa.py",
    "src/vlm/ablation/__init__.py",
    "src/vlm/ablation/config.py",
    "src/vlm/ablation/io_utils.py",
    "src/vlm/ablation/source_hash.py",
    "src/vlm/ablation/init_sync.py",
    "src/vlm/ablation/manifest.py",
    "src/vlm/ablation/checkpoint.py",
    "src/vlm/ablation/eval.py",
    "src/vlm/ablation/routing.py",
    "src/vlm/ablation/train.py",
    "src/vlm/ablation/correctness.py",
    "src/vlm/ablation/aggregate.py",
    "src/vlm/ablation/plots.py",
    "src/vlm/ablation/runner.py",
    "src/vlm/ablation/wandb_logger.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_source_tree(root: Path, relpaths: Iterable[str] = CANONICAL_SOURCE_RELPATHS) -> dict[str, str]:
    payload: dict[str, str] = {}
    for relpath in relpaths:
        path = Path(root) / relpath
        if not path.exists():
            payload[relpath] = "MISSING"
            continue
        payload[relpath] = file_sha256(path)
    return payload


def combined_source_hash(hashes: dict[str, str]) -> str:
    material = json_dumps_sorted(hashes).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def json_dumps_sorted(payload: dict[str, str]) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compare_source_hashes(
    left_root: Path,
    right_root: Path,
    *,
    relpaths: Iterable[str] = CANONICAL_SOURCE_RELPATHS,
) -> dict[str, object]:
    left = hash_source_tree(left_root, relpaths)
    right = hash_source_tree(right_root, relpaths)
    mismatches = {
        relpath: {"left": left[relpath], "right": right[relpath]}
        for relpath in left
        if left.get(relpath) != right.get(relpath)
    }
    return {
        "left_hash": combined_source_hash(left),
        "right_hash": combined_source_hash(right),
        "match": not mismatches,
        "mismatches": mismatches,
        "left": left,
        "right": right,
    }
