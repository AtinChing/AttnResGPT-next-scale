from __future__ import annotations

import zipfile
from pathlib import Path

from src.vlm.ablation.io_utils import ensure_dir


def extract_selected_members(
    zip_path: Path,
    members: list[str],
    dest_root: Path,
    *,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Extract only the requested zip members under dest_root, preserving archive paths."""
    zip_path = Path(zip_path)
    dest_root = Path(dest_root)
    ensure_dir(dest_root)
    extracted = 0
    skipped = 0
    missing = 0
    with zipfile.ZipFile(zip_path) as archive:
        namelist = set(archive.namelist())
        for member in members:
            if member not in namelist:
                missing += 1
                continue
            target = dest_root / member
            if skip_existing and target.exists() and target.stat().st_size > 0:
                skipped += 1
                continue
            ensure_dir(target.parent)
            with archive.open(member) as source, target.open("wb") as handle:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            extracted += 1
    if missing:
        raise FileNotFoundError(f"{missing} requested image members missing from {zip_path.name}")
    return {"extracted": extracted, "skipped": skipped, "missing": missing, "requested": len(members)}
