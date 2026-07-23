from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from src.vlm.clevr.official import OfficialArchive


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sidecar_path(archive_path: Path) -> Path:
    return archive_path.with_suffix(archive_path.suffix + ".integrity.json")


def write_integrity_sidecar(archive_path: Path, *, expected_bytes: int, sha256: str) -> None:
    payload = {
        "path": str(archive_path),
        "bytes": int(archive_path.stat().st_size),
        "expected_bytes": int(expected_bytes),
        "sha256": sha256,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "dl.fbaipublicfiles.com",
    }
    _sidecar_path(archive_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def verify_archive(archive_path: Path, official: OfficialArchive, *, compute_sha256: bool = True) -> dict:
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing archive: {archive_path}")
    size = int(archive_path.stat().st_size)
    if size != official.expected_bytes:
        raise RuntimeError(
            f"Archive size mismatch for {archive_path.name}: got {size}, "
            f"expected {official.expected_bytes} from official listing"
        )
    sidecar = _sidecar_path(archive_path)
    sha256 = None
    if sidecar.exists():
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        if int(recorded.get("bytes", -1)) == size and recorded.get("sha256"):
            sha256 = recorded["sha256"]
    if sha256 is None and compute_sha256:
        sha256 = _sha256_file(archive_path)
        write_integrity_sidecar(archive_path, expected_bytes=official.expected_bytes, sha256=sha256)
    if official.sha256 is not None and sha256 is not None and sha256 != official.sha256:
        raise RuntimeError(f"SHA256 mismatch for {archive_path.name}")
    return {"path": str(archive_path), "bytes": size, "sha256": sha256, "ok": True}


def _stream_to_file(
    url: str,
    partial_path: Path,
    *,
    start_at: int,
    total: int,
    chunk_size: int,
    progress_callback: Callable[[int, int], None] | None,
) -> None:
    headers = {"Range": f"bytes={start_at}-"} if start_at > 0 else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        status = getattr(response, "status", 200)
        mode = "ab"
        written = start_at
        if start_at > 0 and status == 200:
            # Server ignored Range; restart cleanly.
            partial_path.unlink(missing_ok=True)
            written = 0
            mode = "wb"
        with partial_path.open(mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if progress_callback is not None:
                    progress_callback(written, total)
                if written % (64 * 1024 * 1024) < chunk_size:
                    handle.flush()
                    os.fsync(handle.fileno())
            handle.flush()
            os.fsync(handle.fileno())


def download_official_archive(
    official: OfficialArchive,
    dest_dir: Path,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    chunk_size: int = 8 * 1024 * 1024,
) -> Path:
    """Resume-safe download of an official archive into dest_dir."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_path = dest_dir / official.name
    partial_path = dest_dir / f"{official.name}.partial"

    if final_path.exists():
        verify_archive(final_path, official, compute_sha256=False)
        return final_path

    existing = partial_path.stat().st_size if partial_path.exists() else 0
    if existing > official.expected_bytes:
        partial_path.unlink()
        existing = 0

    try:
        _stream_to_file(
            official.url,
            partial_path,
            start_at=existing,
            total=official.expected_bytes,
            chunk_size=chunk_size,
            progress_callback=progress_callback,
        )
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Official download failed for {official.url}: HTTP {error.code}. "
            "Refusing unofficial mirrors."
        ) from error

    if int(partial_path.stat().st_size) != official.expected_bytes:
        raise RuntimeError(
            f"Incomplete download for {official.name}: "
            f"{partial_path.stat().st_size} != {official.expected_bytes}"
        )
    os.replace(partial_path, final_path)
    verify_archive(final_path, official, compute_sha256=True)
    return final_path
