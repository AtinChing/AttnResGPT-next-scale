from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.vlm.ablation.io_utils import atomic_write_json, ensure_dir


class ExperimentManifest:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.data: dict[str, Any] = {"runs": {}, "updated_at": None}
        if self.path.exists():
            import json

            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def run_key(self, variant: str, seed: int, config_hash: str) -> str:
        return f"{variant}/seed_{seed}/{config_hash}"

    def get(self, variant: str, seed: int, config_hash: str) -> dict[str, Any] | None:
        return self.data["runs"].get(self.run_key(variant, seed, config_hash))

    def upsert(self, variant: str, seed: int, config_hash: str, **fields: Any) -> dict[str, Any]:
        key = self.run_key(variant, seed, config_hash)
        row = dict(self.data["runs"].get(key, {}))
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if "start_time" not in row and fields.get("status") == "running":
            row["start_time"] = now
        row.update(fields)
        row["variant"] = variant
        row["seed"] = seed
        row["config_hash"] = config_hash
        row["last_update_time"] = now
        if fields.get("status") == "completed":
            row["completion_time"] = now
        self.data["runs"][key] = row
        self.data["updated_at"] = now
        self.save()
        return row

    def save(self) -> None:
        atomic_write_json(self.path, self.data)

    def summarize(self) -> dict[str, list[str]]:
        buckets = {
            "pending": [],
            "running": [],
            "interrupted": [],
            "completed": [],
            "failed": [],
        }
        for key, row in self.data["runs"].items():
            status = str(row.get("status", "pending"))
            buckets.setdefault(status, []).append(key)
        return buckets
