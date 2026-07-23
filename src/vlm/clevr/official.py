from __future__ import annotations

"""Official CLEVR / CLEVR-CoGenT download endpoints (Facebook AI / Meta).

Never substitute unofficial mirrors.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OfficialArchive:
    name: str
    url: str
    expected_bytes: int
    # No published MD5/SHA on the project page; size is the primary integrity check.
    # After a successful download we persist a local sha256 sidecar for reuse checks.
    sha256: str | None = None


# Verified via HTTP HEAD against dl.fbaipublicfiles.com (2026-07-23).
CLEVR_V1_NO_IMAGES = OfficialArchive(
    name="CLEVR_v1.0_no_images.zip",
    url="https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip",
    expected_bytes=89_363_837,
)
CLEVR_V1_FULL = OfficialArchive(
    name="CLEVR_v1.0.zip",
    url="https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
    expected_bytes=19_021_600_724,
)
COGENT_V1_NO_IMAGES = OfficialArchive(
    name="CLEVR_CoGenT_v1.0_no_images.zip",
    url="https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0_no_images.zip",
    expected_bytes=110_502_280,
)
COGENT_V1_FULL = OfficialArchive(
    name="CLEVR_CoGenT_v1.0.zip",
    url="https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip",
    expected_bytes=24_719_655_191,
)

CLEVR_ROOT_NAME = "CLEVR_v1.0"
COGENT_ROOT_NAME = "CLEVR_CoGenT_v1.0"

CLEVR_SUBSETS = {
    "smoke": {"train_images": 500, "validation_images": 100, "test_images": 100},
    "quick": {"train_images": 5_000, "validation_images": 1_000, "test_images": 1_000},
    "full": {"train_images": 15_000, "validation_images": 3_000, "test_images": 3_000},
}

CLEVR_IMAGE_PREFIX = {
    "train": f"{CLEVR_ROOT_NAME}/images/train/",
    "val": f"{CLEVR_ROOT_NAME}/images/val/",
    "test": f"{CLEVR_ROOT_NAME}/images/test/",
}
COGENT_IMAGE_PREFIX = {
    "trainA": f"{COGENT_ROOT_NAME}/images/trainA/",
    "valA": f"{COGENT_ROOT_NAME}/images/valA/",
    "valB": f"{COGENT_ROOT_NAME}/images/valB/",
    "testA": f"{COGENT_ROOT_NAME}/images/testA/",
    "testB": f"{COGENT_ROOT_NAME}/images/testB/",
}
