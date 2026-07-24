from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

COLORS = ("red", "green", "blue", "yellow")
SHAPES = ("circle", "square", "triangle")
SHAPE_PLURALS = ("circles", "squares", "triangles")
DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
COUNT_WORDS = ("one", "two", "three", "four", "five", "six")
LOCATIONS = (
    "top_left",
    "top",
    "top_right",
    "left",
    "center",
    "right",
    "bottom_left",
    "bottom",
    "bottom_right",
)
FAMILIES = (
    "local_detail",
    "attribute",
    "counting",
    "location",
    "relation",
    "compositional",
    "multi_hop",
)
YES_NO = ("yes", "no")
DIFFICULTY_LEVELS = (1, 2, 3, 4, 5)

SHAPE_TO_PLURAL = {
    "circle": "circles",
    "square": "squares",
    "triangle": "triangles",
}

COLOR_BASE_RGB = {
    "red": np.array([0.92, 0.12, 0.12], dtype=np.float32),
    "green": np.array([0.12, 0.78, 0.18], dtype=np.float32),
    "blue": np.array([0.15, 0.35, 0.95], dtype=np.float32),
    "yellow": np.array([0.95, 0.88, 0.12], dtype=np.float32),
}

# Held-out combinations reserved for validation/test.
HELD_OUT_COLOR_SHAPES = (("yellow", "triangle"), ("blue", "circle"))
HELD_OUT_OBJECT_COUNTS = (6,)
HELD_OUT_TEMPLATE_IDS = ("comp_v2", "hop_v2")
TRAIN_LEVEL_MIX = {1: 0.10, 2: 0.20, 3: 0.25, 4: 0.25, 5: 0.20}

DIGIT_GLYPHS: dict[int, tuple[str, ...]] = {
    0: ("111", "101", "101", "101", "111"),
    1: ("010", "110", "010", "010", "111"),
    2: ("111", "001", "111", "100", "111"),
    3: ("111", "001", "111", "001", "111"),
    4: ("101", "101", "111", "001", "001"),
    5: ("111", "100", "111", "001", "111"),
    6: ("111", "100", "111", "101", "111"),
    7: ("111", "001", "010", "010", "010"),
    8: ("111", "101", "111", "101", "111"),
    9: ("111", "101", "111", "001", "111"),
}

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<answer>")
QUESTION_WORDS = (
    "what",
    "digit",
    "is",
    "inside",
    "in",
    "the",
    "color",
    "shape",
    "how",
    "many",
    "objects",
    "object",
    "are",
    "there",
    "where",
    "left",
    "of",
    "above",
    "below",
    "right",
    "contains",
    "containing",
    "with",
)


@dataclass
class DifficultyProfile:
    """Mutable knobs used by the sanity gate to harden the benchmark."""

    dataset_version: str = "hard_v1"
    bump_level: int = 0
    max_objects: int = 6
    min_objects_level2: int = 3
    digit_scale: float = 1.0
    distractor_count: int = 4
    noise_scale: float = 1.0
    blur_chance: float = 0.35
    occlusion_chance: float = 0.35
    contrast_chance: float = 0.35
    overlap_jitter: float = 0.35
    relation_extra_hops: int = 0

    def bumped(self) -> "DifficultyProfile":
        return DifficultyProfile(
            dataset_version=self.dataset_version,
            bump_level=self.bump_level + 1,
            max_objects=min(8, self.max_objects + 1),
            min_objects_level2=min(4, self.min_objects_level2 + 1),
            digit_scale=max(0.45, self.digit_scale * 0.8),
            distractor_count=self.distractor_count + 3,
            noise_scale=min(2.0, self.noise_scale * 1.25),
            blur_chance=min(0.7, self.blur_chance + 0.1),
            occlusion_chance=min(0.7, self.occlusion_chance + 0.1),
            contrast_chance=min(0.7, self.contrast_chance + 0.1),
            overlap_jitter=min(0.8, self.overlap_jitter + 0.1),
            relation_extra_hops=min(1, self.relation_extra_hops + 1),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneObject:
    color: str
    shape: str
    location: str
    digit: int | None
    size_scale: float = 1.0
    offset_yx: tuple[float, float] = (0.0, 0.0)
    shade: float = 0.0
    digit_offset_yx: tuple[float, float] = (0.0, 0.0)
    digit_scale: float = 1.0


@dataclass(frozen=True)
class VQAExample:
    example_index: int
    split: str
    family: str
    question: str
    answer: str
    objects: tuple[SceneObject, ...]
    image: np.ndarray
    difficulty_level: int
    hops: int
    degradation_strength: float
    held_out: bool
    template_id: str
    visual_degraded: bool


class VQATokenizer:
    def __init__(self) -> None:
        vocab = list(SPECIAL_TOKENS)
        for token in (
            *QUESTION_WORDS,
            *COLORS,
            *SHAPES,
            *SHAPE_PLURALS,
            *DIGITS,
            *COUNT_WORDS,
            *LOCATIONS,
            *YES_NO,
        ):
            if token not in vocab:
                vocab.append(token)
        self.token_to_id = {token: index for index, token in enumerate(vocab)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.answer_token_id = self.token_to_id["<answer>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def vocab(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        tokens = text.split()
        missing = [token for token in tokens if token not in self.token_to_id]
        if missing:
            raise KeyError(f"Unknown tokenizer tokens: {missing} in text={text!r}")
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(self.id_to_token[int(token_id)] for token_id in ids)

    def encode_supervised(
        self,
        question: str,
        answer: str,
        *,
        supervise_eos: bool = True,
    ) -> dict[str, list[int]]:
        """Build teacher-forced VQA sequence without answer leakage.

        Supervise next-token prediction at ``<answer>``, not at the answer token.
        """
        question_ids = self.encode(question)
        answer_id = self.token_to_id[answer]
        input_ids = [self.bos_token_id, *question_ids, self.answer_token_id, answer_id, self.eos_token_id]
        targets = [-100] * len(input_ids)
        answer_position = len(input_ids) - 3
        assert input_ids[answer_position] == self.answer_token_id
        targets[answer_position] = answer_id
        if supervise_eos:
            targets[answer_position + 1] = self.eos_token_id
        return {
            "input_ids": input_ids,
            "targets": targets,
            "answer_position": answer_position,
            "answer_id": answer_id,
        }


def _rng(split_seed: int, example_index: int, salt: str = "") -> np.random.Generator:
    material = f"{split_seed}:{example_index}:{salt}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def _choose_level(rng: np.random.Generator) -> int:
    levels = list(TRAIN_LEVEL_MIX.keys())
    probs = np.array([TRAIN_LEVEL_MIX[level] for level in levels], dtype=np.float64)
    probs = probs / probs.sum()
    return int(rng.choice(levels, p=probs))


def _cell_center(location: str, image_size: int) -> tuple[float, float]:
    row = LOCATIONS.index(location) // 3
    col = LOCATIONS.index(location) % 3
    cell = image_size / 3.0
    return row * cell + cell / 2.0, col * cell + cell / 2.0


def _row_col(location: str) -> tuple[int, int]:
    index = LOCATIONS.index(location)
    return divmod(index, 3)


def _shade_color(color: str, shade: float) -> np.ndarray:
    base = COLOR_BASE_RGB[color]
    # shade in [-0.25, 0.25] moves toward white/black while staying recognizable.
    if shade >= 0:
        return np.clip(base + shade * (1.0 - base), 0.0, 1.0)
    return np.clip(base * (1.0 + shade), 0.0, 1.0)


def _draw_circle(canvas: np.ndarray, cy: float, cx: float, radius: float, color: np.ndarray) -> None:
    yy, xx = np.ogrid[: canvas.shape[1], : canvas.shape[2]]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    for channel in range(3):
        canvas[channel][mask] = color[channel]


def _draw_square(canvas: np.ndarray, cy: float, cx: float, half: float, color: np.ndarray) -> None:
    y0, y1 = max(0, int(cy - half)), min(canvas.shape[1], int(cy + half))
    x0, x1 = max(0, int(cx - half)), min(canvas.shape[2], int(cx + half))
    for channel in range(3):
        canvas[channel, y0:y1, x0:x1] = color[channel]


def _draw_triangle(canvas: np.ndarray, cy: float, cx: float, half: float, color: np.ndarray) -> None:
    half_i = max(1, int(round(half)))
    for row in range(-half_i, half_i + 1):
        width = int(half_i * (1.0 - abs(row) / max(half_i, 1)))
        y = int(cy + row)
        if y < 0 or y >= canvas.shape[1]:
            continue
        x0, x1 = max(0, int(cx - width)), min(canvas.shape[2], int(cx + width + 1))
        for channel in range(3):
            canvas[channel, y, x0:x1] = color[channel]


def _draw_digit(
    canvas: np.ndarray,
    cy: float,
    cx: float,
    digit: int,
    *,
    scale: float,
) -> None:
    glyph = DIGIT_GLYPHS[digit]
    ink = np.array([0.04, 0.04, 0.04], dtype=np.float32)
    scale = max(0.4, scale)
    for row, line in enumerate(glyph):
        for col, bit in enumerate(line):
            if bit != "1":
                continue
            for dy in range(max(1, int(round(scale)))):
                for dx in range(max(1, int(round(scale)))):
                    y = int(cy - 2 * scale + row * scale + dy)
                    x = int(cx - 1 * scale + col * scale + dx)
                    if 0 <= y < canvas.shape[1] and 0 <= x < canvas.shape[2]:
                        canvas[:, y, x] = ink


def _add_distractors(canvas: np.ndarray, rng: np.random.Generator, count: int) -> None:
    h, w = canvas.shape[1], canvas.shape[2]
    for _ in range(count):
        y = int(rng.integers(0, h))
        x = int(rng.integers(0, w))
        value = float(rng.uniform(0.0, 1.0))
        radius = int(rng.integers(1, 3))
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        canvas[:, y0:y1, x0:x1] = value


def _apply_degradations(
    canvas: np.ndarray,
    rng: np.random.Generator,
    profile: DifficultyProfile,
    *,
    level: int,
) -> tuple[np.ndarray, float]:
    strength = 0.0
    out = canvas.copy()
    if level < 2:
        return out, strength

    if level >= 2:
        noise = rng.normal(0.0, 0.03 * profile.noise_scale, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)
        strength += 0.2

    if level >= 3:
        if rng.random() < profile.blur_chance:
            # Cheap 3x3 box blur.
            padded = np.pad(out, ((0, 0), (1, 1), (1, 1)), mode="edge")
            blurred = (
                padded[:, 0:-2, 0:-2]
                + padded[:, 0:-2, 1:-1]
                + padded[:, 0:-2, 2:]
                + padded[:, 1:-1, 0:-2]
                + padded[:, 1:-1, 1:-1]
                + padded[:, 1:-1, 2:]
                + padded[:, 2:, 0:-2]
                + padded[:, 2:, 1:-1]
                + padded[:, 2:, 2:]
            ) / 9.0
            out = blurred.astype(np.float32)
            strength += 0.25
        if rng.random() < profile.contrast_chance:
            factor = float(rng.uniform(0.55, 0.85))
            out = np.clip((out - 0.5) * factor + 0.5, 0.0, 1.0)
            strength += 0.2
        if rng.random() < profile.occlusion_chance:
            h, w = out.shape[1], out.shape[2]
            bh = int(rng.integers(max(4, h // 10), max(5, h // 4)))
            bw = int(rng.integers(max(4, w // 10), max(5, w // 4)))
            y0 = int(rng.integers(0, max(1, h - bh)))
            x0 = int(rng.integers(0, max(1, w - bw)))
            out[:, y0 : y0 + bh, x0 : x0 + bw] = float(rng.uniform(0.2, 0.8))
            strength += 0.25
        noise = rng.normal(0.0, 0.05 * profile.noise_scale, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)
        strength += 0.1
    return out, float(min(1.0, strength))


def render_scene(
    objects: tuple[SceneObject, ...],
    *,
    image_size: int,
    rng: np.random.Generator,
    profile: DifficultyProfile,
    level: int,
) -> tuple[np.ndarray, float]:
    canvas = np.full((3, image_size, image_size), 0.90, dtype=np.float32)
    base_half = max(3.0, image_size / 10.0)
    order = list(range(len(objects)))
    rng.shuffle(order)
    for index in order:
        obj = objects[index]
        cy, cx = _cell_center(obj.location, image_size)
        cy += obj.offset_yx[0] * (image_size / 12.0)
        cx += obj.offset_yx[1] * (image_size / 12.0)
        if level >= 3:
            cy += float(rng.uniform(-profile.overlap_jitter, profile.overlap_jitter))
            cx += float(rng.uniform(-profile.overlap_jitter, profile.overlap_jitter))
        half = base_half * obj.size_scale
        color = _shade_color(obj.color, obj.shade)
        if obj.shape == "circle":
            _draw_circle(canvas, cy, cx, half, color)
        elif obj.shape == "square":
            _draw_square(canvas, cy, cx, half, color)
        else:
            _draw_triangle(canvas, cy, cx, half, color)
        if obj.digit is not None:
            dcy = cy + obj.digit_offset_yx[0] * 2.0
            dcx = cx + obj.digit_offset_yx[1] * 2.0
            _draw_digit(canvas, dcy, dcx, obj.digit, scale=obj.digit_scale * profile.digit_scale)
    if level >= 2:
        _add_distractors(canvas, rng, profile.distractor_count if level >= 2 else 0)
    return _apply_degradations(canvas, rng, profile, level=level)


def _allowed_color_shapes(split: str) -> list[tuple[str, str]]:
    all_pairs = [(color, shape) for color in COLORS for shape in SHAPES]
    held = set(HELD_OUT_COLOR_SHAPES)
    if split == "train":
        return [pair for pair in all_pairs if pair not in held]
    return all_pairs


def _sample_objects(
    rng: np.random.Generator,
    *,
    split: str,
    level: int,
    profile: DifficultyProfile,
) -> tuple[SceneObject, ...] | None:
    if level == 1:
        low, high = 2, 5
    else:
        low = profile.min_objects_level2
        high = profile.max_objects + 1
    count = int(rng.integers(low, high))
    if split == "train" and count in HELD_OUT_OBJECT_COUNTS:
        candidates = [value for value in range(low, high) if value not in HELD_OUT_OBJECT_COUNTS]
        if not candidates:
            return None
        count = int(rng.choice(candidates))

    locations = list(rng.choice(LOCATIONS, size=min(count, len(LOCATIONS)), replace=False))
    pairs = _allowed_color_shapes(split)
    rng.shuffle(pairs)
    if len(pairs) < count:
        return None

    objects: list[SceneObject] = []
    used_pairs: set[tuple[str, str]] = set()
    for location, pair in zip(locations, pairs):
        color, shape = pair
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        digit = int(rng.integers(0, 10)) if (level == 1 and rng.random() < 0.7) or level >= 2 else None
        if level >= 4 and digit is None and rng.random() < 0.85:
            digit = int(rng.integers(0, 10))
        size_scale = float(rng.uniform(0.7, 1.25)) if level >= 2 else 1.0
        offset = (
            float(rng.uniform(-0.8, 0.8)) if level >= 2 else 0.0,
            float(rng.uniform(-0.8, 0.8)) if level >= 2 else 0.0,
        )
        shade = float(rng.uniform(-0.22, 0.22)) if level >= 2 else 0.0
        digit_offset = (
            float(rng.uniform(-1.0, 1.0)) if level >= 2 else 0.0,
            float(rng.uniform(-1.0, 1.0)) if level >= 2 else 0.0,
        )
        digit_scale = float(rng.uniform(0.55, 1.0)) if level >= 2 else 1.0
        objects.append(
            SceneObject(
                color=color,
                shape=shape,
                location=str(location),
                digit=digit,
                size_scale=size_scale,
                offset_yx=offset,
                shade=shade,
                digit_offset_yx=digit_offset,
                digit_scale=digit_scale,
            )
        )
    if len(objects) < 2:
        return None
    return tuple(objects)


def _unique_by(objects: tuple[SceneObject, ...], attr: str, value: Any) -> SceneObject | None:
    matches = [obj for obj in objects if getattr(obj, attr) == value]
    return matches[0] if len(matches) == 1 else None


def _find_relative(objects: tuple[SceneObject, ...], anchor: SceneObject, direction: str) -> SceneObject | None:
    a_row, a_col = _row_col(anchor.location)
    matches: list[SceneObject] = []
    for obj in objects:
        if obj is anchor:
            continue
        row, col = _row_col(obj.location)
        if direction == "left" and row == a_row and col < a_col:
            matches.append(obj)
        elif direction == "right" and row == a_row and col > a_col:
            matches.append(obj)
        elif direction == "above" and col == a_col and row < a_row:
            matches.append(obj)
        elif direction == "below" and col == a_col and row > a_row:
            matches.append(obj)
    if direction in {"left", "above"}:
        # Closest unique neighbor on that axis.
        if not matches:
            return None
        if direction == "left":
            matches.sort(key=lambda obj: -_row_col(obj.location)[1])
        else:
            matches.sort(key=lambda obj: -_row_col(obj.location)[0])
        # Require uniqueness of the nearest neighbor.
        nearest = matches[0]
        nearest_pos = _row_col(nearest.location)
        tied = [obj for obj in matches if _row_col(obj.location) == nearest_pos]
        return nearest if len(tied) == 1 else None
    if direction in {"right", "below"}:
        if not matches:
            return None
        if direction == "right":
            matches.sort(key=lambda obj: _row_col(obj.location)[1])
        else:
            matches.sort(key=lambda obj: _row_col(obj.location)[0])
        nearest = matches[0]
        nearest_pos = _row_col(nearest.location)
        tied = [obj for obj in matches if _row_col(obj.location) == nearest_pos]
        return nearest if len(tied) == 1 else None
    return None


def _build_level1_question(rng: np.random.Generator, objects: tuple[SceneObject, ...]) -> tuple[str, str, str, str, int] | None:
    families = ["local_detail", "attribute", "counting", "location", "relation"]
    rng.shuffle(families)
    for family in families:
        if family == "local_detail":
            candidates = [
                obj
                for obj in objects
                if obj.digit is not None
                and sum(item.color == obj.color and item.shape == obj.shape for item in objects) == 1
            ]
            if not candidates:
                continue
            obj = candidates[int(rng.integers(0, len(candidates)))]
            return family, f"what digit is inside the {obj.color} {obj.shape}", DIGITS[obj.digit], "easy_local", 0
        if family == "attribute":
            if rng.random() < 0.5:
                shape = str(rng.choice(SHAPES))
                obj = _unique_by(objects, "shape", shape)
                if obj is None:
                    continue
                return family, f"what color is the {shape}", obj.color, "easy_attr", 0
            color = str(rng.choice(COLORS))
            obj = _unique_by(objects, "color", color)
            if obj is None:
                continue
            return family, f"what shape is {color}", obj.shape, "easy_attr", 0
        if family == "counting":
            if rng.random() < 0.5:
                shape = str(rng.choice(SHAPES))
                count = sum(obj.shape == shape for obj in objects)
                if count == 0 or count > len(COUNT_WORDS):
                    continue
                return family, f"how many {SHAPE_TO_PLURAL[shape]} are there", COUNT_WORDS[count - 1], "easy_count", 0
            color = str(rng.choice(COLORS))
            count = sum(obj.color == color for obj in objects)
            if count == 0 or count > len(COUNT_WORDS):
                continue
            return family, f"how many {color} objects are there", COUNT_WORDS[count - 1], "easy_count", 0
        if family == "location":
            unique = [
                obj
                for obj in objects
                if sum(item.color == obj.color and item.shape == obj.shape for item in objects) == 1
            ]
            if not unique:
                continue
            obj = unique[int(rng.integers(0, len(unique)))]
            return family, f"where is the {obj.color} {obj.shape}", obj.location, "easy_loc", 0
        if family == "relation" and len(objects) >= 2:
            indices = rng.choice(len(objects), size=2, replace=False)
            left, right = objects[int(indices[0])], objects[int(indices[1])]
            left_row, left_col = _row_col(left.location)
            right_row, right_col = _row_col(right.location)
            if left.location == right.location:
                continue
            if rng.random() < 0.5:
                question = f"is the {left.color} {left.shape} left of the {right.color} {right.shape}"
                answer = "yes" if left_col < right_col else "no"
            else:
                question = f"is the {left.color} {left.shape} above the {right.color} {right.shape}"
                answer = "yes" if left_row < right_row else "no"
            return family, question, answer, "easy_rel", 1
    return None


def _build_compositional(
    rng: np.random.Generator,
    objects: tuple[SceneObject, ...],
    *,
    split: str,
) -> tuple[str, str, str, str, int] | None:
    templates = ["comp_v1", "comp_digit", "comp_count", "comp_v2"]
    if split == "train":
        templates = [template for template in templates if template not in HELD_OUT_TEMPLATE_IDS]
    rng.shuffle(templates)
    for template in templates:
        if template in {"comp_v1", "comp_v2"}:
            anchors = [
                obj
                for obj in objects
                if sum(item.color == obj.color and item.shape == obj.shape for item in objects) == 1
            ]
            if not anchors:
                continue
            anchor = anchors[int(rng.integers(0, len(anchors)))]
            direction = str(rng.choice(["left", "right", "above", "below"]))
            target = _find_relative(objects, anchor, direction)
            if target is None or target.digit is None:
                continue
            if template == "comp_v1":
                question = f"what digit is in the shape {direction} of the {anchor.color} {anchor.shape}"
            else:
                question = f"what digit is inside the object {direction} of the {anchor.color} {anchor.shape}"
            return "compositional", question, DIGITS[target.digit], template, 1
        if template == "comp_digit":
            digit_objs = [obj for obj in objects if obj.digit is not None]
            if not digit_objs:
                continue
            # Prefer unique digits.
            unique = []
            for obj in digit_objs:
                if sum(item.digit == obj.digit for item in objects if item.digit is not None) == 1:
                    unique.append(obj)
            if not unique:
                continue
            obj = unique[int(rng.integers(0, len(unique)))]
            question = f"what shape contains the digit {DIGITS[obj.digit]}"
            return "compositional", question, obj.shape, template, 1
        if template == "comp_count":
            anchors = [
                obj
                for obj in objects
                if sum(item.color == obj.color and item.shape == obj.shape for item in objects) == 1
            ]
            if not anchors:
                continue
            anchor = anchors[int(rng.integers(0, len(anchors)))]
            direction = str(rng.choice(["left", "right", "above", "below"]))
            a_row, a_col = _row_col(anchor.location)
            count = 0
            for obj in objects:
                if obj is anchor:
                    continue
                row, col = _row_col(obj.location)
                if direction == "left" and row == a_row and col < a_col:
                    count += 1
                elif direction == "right" and row == a_row and col > a_col:
                    count += 1
                elif direction == "above" and col == a_col and row < a_row:
                    count += 1
                elif direction == "below" and col == a_col and row > a_row:
                    count += 1
            if count == 0 or count > len(COUNT_WORDS):
                continue
            question = f"how many objects are {direction} the {anchor.color} {anchor.shape}"
            return "compositional", question, COUNT_WORDS[count - 1], template, 1
    return None


def _build_multihop(
    rng: np.random.Generator,
    objects: tuple[SceneObject, ...],
    *,
    split: str,
    profile: DifficultyProfile,
) -> tuple[str, str, str, str, int] | None:
    templates = ["hop_v1", "hop_color", "hop_rel", "hop_v2"]
    if split == "train":
        templates = [template for template in templates if template not in HELD_OUT_TEMPLATE_IDS]
    rng.shuffle(templates)
    hops = 2 + profile.relation_extra_hops
    for template in templates:
        anchors = [
            obj
            for obj in objects
            if sum(item.color == obj.color and item.shape == obj.shape for item in objects) == 1
        ]
        if len(anchors) < 1:
            continue
        if template in {"hop_v1", "hop_v2"}:
            anchor = anchors[int(rng.integers(0, len(anchors)))]
            d1 = str(rng.choice(["left", "right", "above", "below"]))
            mid = _find_relative(objects, anchor, d1)
            if mid is None:
                continue
            d2 = str(rng.choice(["left", "right", "above", "below"]))
            target = _find_relative(objects, mid, d2)
            if target is None or target.digit is None:
                continue
            if template == "hop_v1":
                question = (
                    f"what digit is in the object {d2} of the shape {d1} of the {anchor.color} {anchor.shape}"
                )
            else:
                question = (
                    f"what digit is inside the object {d2} of the object {d1} of the {anchor.color} {anchor.shape}"
                )
            return "multi_hop", question, DIGITS[target.digit], template, hops
        if template == "hop_color":
            digit_objs = [
                obj
                for obj in objects
                if obj.digit is not None
                and sum(item.digit == obj.digit for item in objects if item.digit is not None) == 1
            ]
            if not digit_objs:
                continue
            anchor = digit_objs[int(rng.integers(0, len(digit_objs)))]
            direction = str(rng.choice(["left", "right", "above", "below"]))
            target = _find_relative(objects, anchor, direction)
            if target is None:
                continue
            question = f"what color is the shape {direction} the object containing {DIGITS[anchor.digit]}"
            return "multi_hop", question, target.color, template, hops
        if template == "hop_rel" and len(objects) >= 3:
            a = anchors[int(rng.integers(0, len(anchors)))]
            d1 = str(rng.choice(["left", "right"]))
            b = _find_relative(objects, a, d1)
            if b is None:
                continue
            c_candidates = [obj for obj in objects if obj is not a and obj is not b]
            if not c_candidates:
                continue
            c = c_candidates[int(rng.integers(0, len(c_candidates)))]
            b_row, _ = _row_col(b.location)
            c_row, _ = _row_col(c.location)
            question = (
                f"is the object {d1} of the {a.color} {a.shape} above the {c.color} {c.shape}"
            )
            answer = "yes" if b_row < c_row else "no"
            return "multi_hop", question, answer, template, hops
    return None


def _build_question(
    rng: np.random.Generator,
    objects: tuple[SceneObject, ...],
    *,
    level: int,
    split: str,
    profile: DifficultyProfile,
) -> tuple[str, str, str, str, int] | None:
    if level <= 3:
        return _build_level1_question(rng, objects)
    if level == 4:
        return _build_compositional(rng, objects, split=split)
    return _build_multihop(rng, objects, split=split, profile=profile)


def _is_held_out(objects: tuple[SceneObject, ...], template_id: str) -> bool:
    if template_id in HELD_OUT_TEMPLATE_IDS:
        return True
    if len(objects) in HELD_OUT_OBJECT_COUNTS:
        return True
    for obj in objects:
        if (obj.color, obj.shape) in HELD_OUT_COLOR_SHAPES:
            return True
    return False


def generate_example(
    *,
    split: str,
    split_seed: int,
    example_index: int,
    image_size: int = 64,
    profile: DifficultyProfile | None = None,
    forced_level: int | None = None,
) -> VQAExample:
    profile = profile or DifficultyProfile()
    rng = _rng(split_seed, example_index, salt=profile.dataset_version)
    level = forced_level if forced_level is not None else _choose_level(rng)

    for attempt in range(48):
        attempt_rng = _rng(split_seed, example_index, salt=f"{profile.dataset_version}:{level}:{attempt}")
        objects = _sample_objects(attempt_rng, split=split, level=level, profile=profile)
        if objects is None:
            continue
        built = _build_question(attempt_rng, objects, level=level, split=split, profile=profile)
        if built is None:
            continue
        family, question, answer, template_id, hops = built
        # Reject train examples that accidentally use held-out markers.
        held = _is_held_out(objects, template_id)
        if split == "train" and held:
            continue
        if split != "train" and not held and attempt < 16 and level >= 4:
            # Bias val/test toward held-out combinations for levels 4/5.
            continue
        image, degradation = render_scene(
            objects,
            image_size=image_size,
            rng=attempt_rng,
            profile=profile,
            level=level,
        )
        return VQAExample(
            example_index=example_index,
            split=split,
            family=family,
            question=question,
            answer=answer,
            objects=objects,
            image=image,
            difficulty_level=level,
            hops=hops,
            degradation_strength=degradation,
            held_out=held,
            template_id=template_id,
            visual_degraded=level >= 3 or degradation > 0.05,
        )

    # Deterministic safe fallback: easy unambiguous attribute question.
    fallback_objects = (
        SceneObject(color="red", shape="circle", location="center", digit=1),
        SceneObject(color="green", shape="square", location="left", digit=None),
    )
    image, degradation = render_scene(
        fallback_objects,
        image_size=image_size,
        rng=rng,
        profile=profile,
        level=1,
    )
    return VQAExample(
        example_index=example_index,
        split=split,
        family="attribute",
        question="what color is the circle",
        answer="red",
        objects=fallback_objects,
        image=image,
        difficulty_level=1,
        hops=0,
        degradation_strength=degradation,
        held_out=False,
        template_id="fallback",
        visual_degraded=False,
    )


class SyntheticVQADataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        split: Literal["train", "validation", "test"],
        size: int,
        split_seed: int,
        tokenizer: VQATokenizer,
        image_size: int = 64,
        supervise_eos: bool = True,
        profile: DifficultyProfile | None = None,
        forced_level: int | None = None,
    ) -> None:
        self.split = split
        self.size = size
        self.split_seed = split_seed
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.supervise_eos = supervise_eos
        self.profile = profile or DifficultyProfile()
        self.forced_level = forced_level

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = generate_example(
            split=self.split,
            split_seed=self.split_seed,
            example_index=index,
            image_size=self.image_size,
            profile=self.profile,
            forced_level=self.forced_level,
        )
        encoded = self.tokenizer.encode_supervised(
            example.question,
            example.answer,
            supervise_eos=self.supervise_eos,
        )
        return {
            "pixel_values": torch.from_numpy(example.image.copy()),
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "targets": torch.tensor(encoded["targets"], dtype=torch.long),
            "answer_position": encoded["answer_position"],
            "answer_id": encoded["answer_id"],
            "family": example.family,
            "question": example.question,
            "answer": example.answer,
            "example_index": example.example_index,
            "difficulty_level": example.difficulty_level,
            "hops": example.hops,
            "degradation_strength": example.degradation_strength,
            "held_out": example.held_out,
            "template_id": example.template_id,
            "visual_degraded": example.visual_degraded,
        }


def collate_vqa_batch(examples: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    max_len = max(int(example["input_ids"].numel()) for example in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), -100, dtype=torch.long)
    answer_positions = torch.zeros(batch_size, dtype=torch.long)
    answer_ids = torch.zeros(batch_size, dtype=torch.long)
    pixel_values = torch.stack([example["pixel_values"] for example in examples], dim=0)
    families: list[str] = []
    questions: list[str] = []
    answers: list[str] = []
    difficulty_levels = torch.zeros(batch_size, dtype=torch.long)
    hops = torch.zeros(batch_size, dtype=torch.long)
    degradation = torch.zeros(batch_size, dtype=torch.float32)
    held_out = torch.zeros(batch_size, dtype=torch.bool)
    visual_degraded = torch.zeros(batch_size, dtype=torch.bool)
    template_ids: list[str] = []
    for row, example in enumerate(examples):
        width = int(example["input_ids"].numel())
        input_ids[row, :width] = example["input_ids"]
        targets[row, :width] = example["targets"]
        answer_positions[row] = int(example["answer_position"])
        answer_ids[row] = int(example["answer_id"])
        families.append(example["family"])
        questions.append(example["question"])
        answers.append(example["answer"])
        difficulty_levels[row] = int(example["difficulty_level"])
        hops[row] = int(example["hops"])
        degradation[row] = float(example["degradation_strength"])
        held_out[row] = bool(example["held_out"])
        visual_degraded[row] = bool(example["visual_degraded"])
        template_ids.append(example["template_id"])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "targets": targets,
        "answer_positions": answer_positions,
        "answer_ids": answer_ids,
        "families": families,
        "questions": questions,
        "answers": answers,
        "difficulty_levels": difficulty_levels,
        "hops": hops,
        "degradation_strength": degradation,
        "held_out": held_out,
        "visual_degraded": visual_degraded,
        "template_ids": template_ids,
    }
