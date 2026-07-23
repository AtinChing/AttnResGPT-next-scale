from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.vlm.clevr.preprocess import PreprocessConfig, load_rgb_image, resize_and_pad
from src.vlm.clevr.programs import analyze_program
from src.vlm.clevr.tokenizer import CLEVRTokenizer


class CLEVRExampleDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        *,
        examples: list[dict[str, Any]],
        image_root: Path,
        image_prefix: str,
        tokenizer: CLEVRTokenizer,
        preprocess: PreprocessConfig,
        supervise_eos: bool = True,
        allow_unk: bool = True,
        control_mode: str = "none",  # none | question_only | blank_question
    ) -> None:
        self.examples = examples
        self.image_root = Path(image_root)
        self.image_prefix = image_prefix
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.supervise_eos = supervise_eos
        self.allow_unk = allow_unk
        self.control_mode = control_mode

    def __len__(self) -> int:
        return len(self.examples)

    def _image_path(self, filename: str) -> Path:
        return self.image_root / self.image_prefix / filename

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.examples[index]
        stats = analyze_program(item.get("program"))
        question = str(item["question"])
        answer = str(item["answer"])
        if self.control_mode == "blank_question":
            question = ""
        encoded = self.tokenizer.encode_supervised(
            question,
            answer,
            supervise_eos=self.supervise_eos,
            allow_unk=self.allow_unk,
        )
        image_path = self._image_path(str(item["image_filename"]))
        if self.control_mode == "question_only":
            pixels = torch.zeros(
                (3, self.preprocess.image_size, self.preprocess.image_size),
                dtype=torch.float32,
            )
        else:
            image = load_rgb_image(image_path)
            array = resize_and_pad(image, config=self.preprocess)
            pixels = torch.from_numpy(array.copy())

        shape_flags = {
            "mentions_cube": "cube" in question.lower() or "cubes" in question.lower(),
            "mentions_cylinder": "cylinder" in question.lower() or "cylinders" in question.lower(),
            "mentions_sphere": "sphere" in question.lower() or "spheres" in question.lower()
            or "ball" in question.lower()
            or "balls" in question.lower(),
        }
        return {
            "pixel_values": pixels,
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "targets": torch.tensor(encoded["targets"], dtype=torch.long),
            "answer_position": encoded["answer_position"],
            "answer_id": encoded["answer_id"],
            "family": stats.reasoning_category,
            "question": question,
            "answer": answer,
            "official_answer": answer,
            "image_filename": str(item["image_filename"]),
            "image_index": int(item["image_index"]),
            "question_index": int(item["question_index"]),
            "question_family_index": int(item["question_family_index"]),
            "program": item.get("program") or [],
            "n_operations": stats.n_operations,
            "dependency_depth": stats.dependency_depth,
            "n_relational_ops": stats.n_relational_ops,
            "terminal_function": stats.terminal_function,
            "program_length_bin": stats.program_length_bin,
            "reasoning_category": stats.reasoning_category,
            **shape_flags,
        }


def collate_clevr_batch(examples: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    max_len = max(int(example["input_ids"].numel()) for example in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_len), -100, dtype=torch.long)
    answer_positions = torch.zeros(batch_size, dtype=torch.long)
    answer_ids = torch.zeros(batch_size, dtype=torch.long)
    pixel_values = torch.stack([example["pixel_values"] for example in examples], dim=0)
    payload: dict[str, Any] = {
        "families": [],
        "questions": [],
        "answers": [],
        "image_filenames": [],
        "image_indices": [],
        "question_indices": [],
        "question_family_indices": [],
        "n_operations": [],
        "dependency_depths": [],
        "program_length_bins": [],
        "reasoning_categories": [],
        "terminal_functions": [],
        "mentions_cube": [],
        "mentions_cylinder": [],
        "mentions_sphere": [],
        "programs": [],
    }
    for row, example in enumerate(examples):
        width = int(example["input_ids"].numel())
        input_ids[row, :width] = example["input_ids"]
        targets[row, :width] = example["targets"]
        answer_positions[row] = int(example["answer_position"])
        answer_ids[row] = int(example["answer_id"])
        payload["families"].append(example["family"])
        payload["questions"].append(example["question"])
        payload["answers"].append(example["answer"])
        payload["image_filenames"].append(example["image_filename"])
        payload["image_indices"].append(example["image_index"])
        payload["question_indices"].append(example["question_index"])
        payload["question_family_indices"].append(example["question_family_index"])
        payload["n_operations"].append(example["n_operations"])
        payload["dependency_depths"].append(example["dependency_depth"])
        payload["program_length_bins"].append(example["program_length_bin"])
        payload["reasoning_categories"].append(example["reasoning_category"])
        payload["terminal_functions"].append(example["terminal_function"])
        payload["mentions_cube"].append(example["mentions_cube"])
        payload["mentions_cylinder"].append(example["mentions_cylinder"])
        payload["mentions_sphere"].append(example["mentions_sphere"])
        payload["programs"].append(example["program"])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "targets": targets,
        "answer_positions": answer_positions,
        "answer_ids": answer_ids,
        **payload,
    }
