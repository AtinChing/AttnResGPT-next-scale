from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


def atomic_torch_save(path: Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
        handle.flush()


def create_project_layout(project_root: Path) -> dict[str, Path]:
    names = [
        "source",
        "configs",
        "checkpoints",
        "runs",
        "logs",
        "metrics",
        "summaries",
        "plots",
        "tables",
        "cache",
        "manifests",
        "data",
    ]
    paths = {name: ensure_dir(project_root / name) for name in names}
    return paths
