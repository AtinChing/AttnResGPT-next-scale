from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from src.data.tokenizer import TokenizerWrapper


@dataclass
class CaptionExample:
    image: Any
    caption: str


@dataclass
class CaptionRecord:
    row_index: int
    caption: str


def _extract_captions(payload: Any) -> Iterable[str]:
    if isinstance(payload, str):
        text = payload.strip()
        if text:
            yield text
    elif isinstance(payload, list):
        for item in payload:
            yield from _extract_captions(item)
    elif isinstance(payload, dict):
        for key in ("caption", "text", "raw", "sentence"):
            if key in payload:
                yield from _extract_captions(payload[key])


def _row_captions(row: dict[str, Any]) -> list[str]:
    for key in ("caption", "captions", "sentences", "alt_text", "original_alt_text"):
        if key in row:
            captions = list(_extract_captions(row[key]))
            if captions:
                return captions
    return []


def _row_image(row: dict[str, Any]) -> Any:
    for key in ("image", "img", "jpg"):
        if key in row:
            return row[key]
    raise KeyError("Could not find image field in Flickr30K row")


def load_flickr30k_examples(
    *,
    dataset_name: str,
    split: str,
    max_examples: int,
    seed: int,
) -> tuple[Any, list[CaptionRecord], list[CaptionRecord]]:
    try:
        dataset = load_dataset(dataset_name, split=split)
        requested_row_split = None
    except Exception:
        dataset_dict = load_dataset(dataset_name)
        hf_split_name = split if split in dataset_dict else next(iter(dataset_dict.keys()))
        dataset = dataset_dict[hf_split_name]
        requested_row_split = split
    if hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)

    examples: list[CaptionRecord] = []
    for row_index, row in enumerate(dataset):
        if requested_row_split is not None and "split" in row:
            row_split = row["split"]
            if isinstance(row_split, str) and row_split.strip().lower() != requested_row_split.strip().lower():
                continue
        for caption in _row_captions(row):
            examples.append(CaptionRecord(row_index=row_index, caption=caption))
            if len(examples) >= max_examples:
                break
        if len(examples) >= max_examples:
            break

    if len(examples) < 2:
        raise ValueError("Not enough Flickr30K examples were loaded")

    cutoff = max(1, int(0.9 * len(examples)))
    return dataset, examples[:cutoff], examples[cutoff:]


class CaptionExampleDataset(Dataset[CaptionExample]):
    def __init__(self, dataset: Any, examples: Sequence[CaptionRecord]) -> None:
        self.dataset = dataset
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> CaptionExample:
        record = self.examples[index]
        row = self.dataset[int(record.row_index)]
        return CaptionExample(image=_row_image(row), caption=record.caption)


class Flickr30KCollator:
    def __init__(
        self,
        *,
        processor: Any,
        tokenizer: TokenizerWrapper,
        max_text_tokens: int,
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_text_tokens = max_text_tokens
        backend = tokenizer.backend
        self.pad_token_id = int(backend.pad_token_id if backend.pad_token_id is not None else 0)
        bos_token_id = backend.bos_token_id
        if bos_token_id is None:
            bos_token_id = backend.eos_token_id
        if bos_token_id is None:
            bos_token_id = self.pad_token_id
        self.bos_token_id = int(bos_token_id)

    def __call__(self, examples: Sequence[CaptionExample]) -> dict[str, torch.Tensor]:
        images = [example.image for example in examples]
        captions = [example.caption for example in examples]
        pixel_values = self.processor(images=images, return_tensors="pt")["pixel_values"]

        input_rows: list[list[int]] = []
        target_rows: list[list[int]] = []
        for caption in captions:
            token_ids = self.tokenizer.encode(caption)[: self.max_text_tokens]
            if not token_ids:
                token_ids = [self.bos_token_id]
            decoder_input = [self.bos_token_id, *token_ids[:-1]]
            input_rows.append(decoder_input)
            target_rows.append(token_ids)

        max_len = max(len(row) for row in input_rows)
        input_ids = torch.full((len(input_rows), max_len), self.pad_token_id, dtype=torch.long)
        targets = torch.full((len(input_rows), max_len), -100, dtype=torch.long)
        text_mask = torch.zeros((len(input_rows), max_len), dtype=torch.bool)

        for row_index, (decoder_input, decoder_target) in enumerate(zip(input_rows, target_rows)):
            input_ids[row_index, : len(decoder_input)] = torch.tensor(decoder_input, dtype=torch.long)
            targets[row_index, : len(decoder_target)] = torch.tensor(decoder_target, dtype=torch.long)
            text_mask[row_index, : len(decoder_input)] = True

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "targets": targets,
            "text_mask": text_mask,
        }


def build_flickr30k_dataloaders(
    *,
    dataset_name: str,
    split: str,
    processor: Any,
    tokenizer: TokenizerWrapper,
    max_examples: int,
    max_text_tokens: int,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    dataset, train_examples, val_examples = load_flickr30k_examples(
        dataset_name=dataset_name,
        split=split,
        max_examples=max_examples,
        seed=seed,
    )
    collator = Flickr30KCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_text_tokens=max_text_tokens,
    )
    train_loader = DataLoader(
        CaptionExampleDataset(dataset, train_examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        CaptionExampleDataset(dataset, val_examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
