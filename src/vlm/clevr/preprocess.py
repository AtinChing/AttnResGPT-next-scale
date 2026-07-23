from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 128
    pad_value: float = 0.0
    # ImageNet-ish constants kept fixed; recorded for hash identity.
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    preserve_aspect_ratio: bool = True
    deterministic_pad: bool = True
    augmentation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        material = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(material).hexdigest()[:16]


def load_rgb_image(path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def resize_and_pad(
    image: Image.Image,
    *,
    config: PreprocessConfig,
) -> np.ndarray:
    """Resize preserving aspect ratio, then deterministic center pad to square."""
    target = int(config.image_size)
    width, height = image.size
    scale = target / max(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target, target), color=(0, 0, 0))
    # Deterministic centered pad (floor for left/top).
    left = (target - new_w) // 2
    top = (target - new_h) // 2
    canvas.paste(resized, (left, top))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    mean = np.asarray(config.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config.std, dtype=np.float32).reshape(1, 1, 3)
    array = (array - mean) / std
    # CHW
    return np.transpose(array, (2, 0, 1)).astype(np.float32)
