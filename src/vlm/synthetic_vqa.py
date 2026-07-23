from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

COLORS = ("red", "green", "blue", "yellow")
SHAPES = ("circle", "square", "triangle")
DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
COUNT_WORDS = ("one", "two", "three", "four")
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
FAMILIES = ("local_detail", "attribute", "counting", "location", "relation")
YES_NO = ("yes", "no")

COLOR_RGB = {
    "red": (1.0, 0.1, 0.1),
    "green": (0.1, 0.8, 0.1),
    "blue": (0.1, 0.3, 1.0),
    "yellow": (0.95, 0.9, 0.1),
}

# Hard-coded 3x5 digit glyphs. 1 = ink, 0 = background.
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
    "the",
    "color",
    "shape",
    "how",
    "many",
    "objects",
    "are",
    "there",
    "where",
    "left",
    "of",
    "above",
    "below",
    "right",
)


@dataclass(frozen=True)
class SceneObject:
    color: str
    shape: str
    location: str
    digit: int | None


@dataclass(frozen=True)
class VQAExample:
    example_index: int
    split: str
    family: str
    question: str
    answer: str
    objects: tuple[SceneObject, ...]
    image: np.ndarray  # float32 CHW in [0, 1]


class VQATokenizer:
    def __init__(self) -> None:
        vocab = list(SPECIAL_TOKENS)
        for token in (
            *QUESTION_WORDS,
            *COLORS,
            *SHAPES,
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
        return [self.token_to_id[token] for token in text.split()]

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
        question_ids = self.encode(question)
        answer_id = self.token_to_id[answer]
        input_ids = [self.bos_token_id, *question_ids, self.answer_token_id, answer_id, self.eos_token_id]
        targets = [-100] * len(input_ids)
        answer_position = len(input_ids) - 2
        targets[answer_position] = answer_id
        if supervise_eos:
            targets[answer_position + 1] = self.eos_token_id
        return {
            "input_ids": input_ids,
            "targets": targets,
            "answer_position": answer_position,
            "answer_id": answer_id,
        }


def _rng(split_seed: int, example_index: int) -> np.random.Generator:
    material = f"{split_seed}:{example_index}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def _cell_center(location: str, image_size: int) -> tuple[int, int]:
    row = LOCATIONS.index(location) // 3
    col = LOCATIONS.index(location) % 3
    cell = image_size // 3
    return row * cell + cell // 2, col * cell + cell // 2


def _draw_circle(canvas: np.ndarray, cy: int, cx: int, radius: int, color: tuple[float, float, float]) -> None:
    yy, xx = np.ogrid[: canvas.shape[1], : canvas.shape[2]]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
    for channel, value in enumerate(color):
        canvas[channel][mask] = value


def _draw_square(canvas: np.ndarray, cy: int, cx: int, half: int, color: tuple[float, float, float]) -> None:
    y0, y1 = max(0, cy - half), min(canvas.shape[1], cy + half)
    x0, x1 = max(0, cx - half), min(canvas.shape[2], cx + half)
    for channel, value in enumerate(color):
        canvas[channel, y0:y1, x0:x1] = value


def _draw_triangle(canvas: np.ndarray, cy: int, cx: int, half: int, color: tuple[float, float, float]) -> None:
    for row in range(-half, half + 1):
        width = int(half * (1.0 - abs(row) / max(half, 1)))
        y = cy + row
        if y < 0 or y >= canvas.shape[1]:
            continue
        x0, x1 = max(0, cx - width), min(canvas.shape[2], cx + width + 1)
        for channel, value in enumerate(color):
            canvas[channel, y, x0:x1] = value


def _draw_digit(canvas: np.ndarray, cy: int, cx: int, digit: int) -> None:
    glyph = DIGIT_GLYPHS[digit]
    ink = (0.05, 0.05, 0.05)
    for row, line in enumerate(glyph):
        for col, bit in enumerate(line):
            if bit != "1":
                continue
            y = cy - 2 + row
            x = cx - 1 + col
            if 0 <= y < canvas.shape[1] and 0 <= x < canvas.shape[2]:
                for channel, value in enumerate(ink):
                    canvas[channel, y, x] = value


def render_scene(objects: tuple[SceneObject, ...], *, image_size: int = 64) -> np.ndarray:
    canvas = np.full((3, image_size, image_size), 0.92, dtype=np.float32)
    half = max(4, image_size // 10)
    for obj in objects:
        cy, cx = _cell_center(obj.location, image_size)
        color = COLOR_RGB[obj.color]
        if obj.shape == "circle":
            _draw_circle(canvas, cy, cx, half, color)
        elif obj.shape == "square":
            _draw_square(canvas, cy, cx, half, color)
        else:
            _draw_triangle(canvas, cy, cx, half, color)
        if obj.digit is not None:
            _draw_digit(canvas, cy, cx, obj.digit)
    return canvas


def _sample_objects(rng: np.random.Generator) -> tuple[SceneObject, ...]:
    count = int(rng.integers(2, 5))
    locations = list(rng.choice(LOCATIONS, size=count, replace=False))
    objects: list[SceneObject] = []
    used_color_shape: set[tuple[str, str]] = set()
    for location in locations:
        for _ in range(32):
            color = str(rng.choice(COLORS))
            shape = str(rng.choice(SHAPES))
            if (color, shape) in used_color_shape:
                continue
            used_color_shape.add((color, shape))
            digit = int(rng.integers(0, 10)) if rng.random() < 0.7 else None
            objects.append(SceneObject(color=color, shape=shape, location=str(location), digit=digit))
            break
        else:
            color = str(rng.choice(COLORS))
            shape = str(rng.choice(SHAPES))
            objects.append(SceneObject(color=color, shape=shape, location=str(location), digit=None))
    return tuple(objects)


def _unique_by(objects: tuple[SceneObject, ...], attr: str, value: str) -> SceneObject | None:
    matches = [obj for obj in objects if getattr(obj, attr) == value]
    return matches[0] if len(matches) == 1 else None


def _build_question(rng: np.random.Generator, objects: tuple[SceneObject, ...]) -> tuple[str, str, str]:
    families = list(FAMILIES)
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
            question = f"what digit is inside the {obj.color} {obj.shape}"
            return family, question, DIGITS[obj.digit]

        if family == "attribute":
            if rng.random() < 0.5:
                shape = str(rng.choice(SHAPES))
                obj = _unique_by(objects, "shape", shape)
                if obj is None:
                    continue
                return family, f"what color is the {shape}", obj.color
            color = str(rng.choice(COLORS))
            obj = _unique_by(objects, "color", color)
            if obj is None:
                continue
            return family, f"what shape is {color}", obj.shape

        if family == "counting":
            if rng.random() < 0.5:
                shape = str(rng.choice(SHAPES))
                count = sum(obj.shape == shape for obj in objects)
                if count == 0:
                    continue
                return family, f"how many {shape}s are there", COUNT_WORDS[count - 1]
            color = str(rng.choice(COLORS))
            count = sum(obj.color == color for obj in objects)
            if count == 0:
                continue
            return family, f"how many {color} objects are there", COUNT_WORDS[count - 1]

        if family == "location":
            unique = [
                obj
                for obj in objects
                if sum(item.color == obj.color and item.shape == obj.shape for item in objects) == 1
            ]
            if not unique:
                continue
            obj = unique[int(rng.integers(0, len(unique)))]
            return family, f"where is the {obj.color} {obj.shape}", obj.location

        if family == "relation":
            if len(objects) < 2:
                continue
            indices = rng.choice(len(objects), size=2, replace=False)
            left, right = objects[int(indices[0])], objects[int(indices[1])]
            if left.location == right.location:
                continue
            left_row, left_col = divmod(LOCATIONS.index(left.location), 3)
            right_row, right_col = divmod(LOCATIONS.index(right.location), 3)
            if rng.random() < 0.5:
                question = f"is the {left.color} {left.shape} left of the {right.color} {right.shape}"
                answer = "yes" if left_col < right_col else "no"
            else:
                question = f"is the {left.color} {left.shape} above the {right.color} {right.shape}"
                answer = "yes" if left_row < right_row else "no"
            return family, question, answer

    # Deterministic fallback that is always unambiguous.
    for obj in objects:
        if sum(item.shape == obj.shape for item in objects) == 1:
            return "attribute", f"what color is the {obj.shape}", obj.color
    obj = objects[0]
    return "location", f"where is the {obj.color} {obj.shape}", obj.location


def generate_example(
    *,
    split: str,
    split_seed: int,
    example_index: int,
    image_size: int = 64,
) -> VQAExample:
    rng = _rng(split_seed, example_index)
    objects = _sample_objects(rng)
    family, question, answer = _build_question(rng, objects)
    image = render_scene(objects, image_size=image_size)
    return VQAExample(
        example_index=example_index,
        split=split,
        family=family,
        question=question,
        answer=answer,
        objects=objects,
        image=image,
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
    ) -> None:
        self.split = split
        self.size = size
        self.split_seed = split_seed
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.supervise_eos = supervise_eos

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = generate_example(
            split=self.split,
            split_seed=self.split_seed,
            example_index=index,
            image_size=self.image_size,
        )
        encoded = self.tokenizer.encode_supervised(
            example.question,
            example.answer,
            supervise_eos=self.supervise_eos,
        )
        return {
            "pixel_values": torch.from_numpy(example.image),
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "targets": torch.tensor(encoded["targets"], dtype=torch.long),
            "answer_position": encoded["answer_position"],
            "answer_id": encoded["answer_id"],
            "family": example.family,
            "question": example.question,
            "answer": example.answer,
            "example_index": example.example_index,
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
    for row, example in enumerate(examples):
        width = int(example["input_ids"].numel())
        input_ids[row, :width] = example["input_ids"]
        targets[row, :width] = example["targets"]
        answer_positions[row] = int(example["answer_position"])
        answer_ids[row] = int(example["answer_id"])
        families.append(example["family"])
        questions.append(example["question"])
        answers.append(example["answer"])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "targets": targets,
        "answer_positions": answer_positions,
        "answer_ids": answer_ids,
        "families": families,
        "questions": questions,
        "answers": answers,
    }
