from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.data.tokenizer import TokenizerWrapper, build_tokenizer
from src.utils.config import Config


def _extract_text_fields(payload: Any) -> Iterable[str]:
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, list):
        for item in payload:
            yield from _extract_text_fields(item)
    elif isinstance(payload, dict):
        for key in ("text", "story", "content", "prompt", "completion"):
            if key in payload:
                yield from _extract_text_fields(payload[key])


def _read_json_file(path: Path) -> str:
    if path.suffix == ".jsonl":
        rows: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.extend(_extract_text_fields(json.loads(line)))
        return "\n".join(rows)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return "\n".join(_extract_text_fields(payload))


def read_local_corpus(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Could not find corpus at {resolved}")
    if resolved.is_file():
        if resolved.suffix in {".json", ".jsonl"}:
            return _read_json_file(resolved)
        return resolved.read_text(encoding="utf-8")

    texts: list[str] = []
    for file_path in sorted(resolved.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix in {".txt", ".md"}:
            texts.append(file_path.read_text(encoding="utf-8"))
        elif file_path.suffix in {".json", ".jsonl"}:
            texts.append(_read_json_file(file_path))
    if not texts:
        raise ValueError(f"No readable text files found under {resolved}")
    return "\n".join(texts)


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if len(token_ids) <= block_size:
            raise ValueError("Token corpus is too small for the configured block size")
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x = torch.tensor(self.token_ids[index : index + self.block_size], dtype=torch.long)
        y = torch.tensor(self.token_ids[index + 1 : index + 1 + self.block_size], dtype=torch.long)
        return {"input_ids": x, "targets": y}


class StreamingTokenBlockDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        tokenizer: TokenizerWrapper,
        text_iterator_factory: Callable[[], Iterator[str]],
        *,
        block_size: int,
        max_examples: Optional[int],
        max_tokens: Optional[int],
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.text_iterator_factory = text_iterator_factory
        self.block_size = block_size
        self.max_examples = max_examples
        self.max_tokens = max_tokens

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        tokens_seen = 0
        examples_seen = 0
        buffer: list[int] = []
        for text in self.text_iterator_factory():
            if self.max_examples is not None and examples_seen >= self.max_examples:
                break
            token_ids = self.tokenizer.encode(text)
            if not token_ids:
                continue
            if self.max_tokens is not None:
                remaining = self.max_tokens - tokens_seen
                if remaining <= 0:
                    break
                token_ids = token_ids[:remaining]
            buffer.extend(token_ids)
            tokens_seen += len(token_ids)
            examples_seen += 1

            while len(buffer) >= self.block_size + 1:
                x = torch.tensor(buffer[: self.block_size], dtype=torch.long)
                y = torch.tensor(buffer[1 : self.block_size + 1], dtype=torch.long)
                yield {"input_ids": x, "targets": y}
                buffer = buffer[self.block_size :]

            if self.max_tokens is not None and tokens_seen >= self.max_tokens:
                break


def _tinystories_text_iterator(split: str) -> Iterator[str]:
    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for row in dataset:
        text = row.get("text") or row.get("story") or ""
        if text:
            yield str(text)


def _split_train_val_text(text: str) -> tuple[str, str]:
    cutoff = max(1, int(0.9 * len(text)))
    train_text = text[:cutoff]
    val_text = text[cutoff:]
    if len(val_text) < 512:
        tail = min(len(text), 4096)
        train_text = text[:-tail]
        val_text = text[-tail:]
    return train_text, val_text


def build_datasets(config: Config) -> tuple[TokenizerWrapper, Dataset | IterableDataset, Dataset | IterableDataset, dict[str, Any]]:
    tokenizer = build_tokenizer(config.data.tokenizer_name)
    block_size = config.data.block_size

    if config.data.dataset_type == "tinystories":
        train_dataset = StreamingTokenBlockDataset(
            tokenizer,
            text_iterator_factory=lambda: _tinystories_text_iterator(config.data.train_split),
            block_size=block_size,
            max_examples=config.data.max_train_examples,
            max_tokens=config.data.max_train_tokens,
        )
        val_dataset = StreamingTokenBlockDataset(
            tokenizer,
            text_iterator_factory=lambda: _tinystories_text_iterator(config.data.val_split),
            block_size=block_size,
            max_examples=config.data.max_val_examples,
            max_tokens=config.data.max_val_tokens,
        )
        metadata = {
            "dataset": config.data.dataset_name,
            "tokenizer_name": tokenizer.name,
            "train_cap_examples": config.data.max_train_examples,
            "train_cap_tokens": config.data.max_train_tokens,
            "val_cap_examples": config.data.max_val_examples,
            "val_cap_tokens": config.data.max_val_tokens,
        }
        return tokenizer, train_dataset, val_dataset, metadata

    if config.data.train_text_path:
        train_text = read_local_corpus(config.data.train_text_path)
        val_path = config.data.val_text_path or config.data.train_text_path
        val_text = read_local_corpus(val_path)
    elif config.data.text_path:
        train_text, val_text = _split_train_val_text(read_local_corpus(config.data.text_path))
    else:
        raise ValueError("local_text dataset requires text_path or train_text_path")

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    if config.data.max_train_examples is not None:
        train_ids = train_ids[: block_size + config.data.max_train_examples]
    if config.data.max_val_examples is not None:
        val_ids = val_ids[: block_size + config.data.max_val_examples]
    if config.data.max_train_tokens is not None:
        train_ids = train_ids[: config.data.max_train_tokens]
    if config.data.max_val_tokens is not None:
        val_ids = val_ids[: config.data.max_val_tokens]

    train_dataset = TokenBlockDataset(train_ids, block_size)
    val_dataset = TokenBlockDataset(val_ids, block_size)
    metadata = {
        "dataset": config.data.dataset_name,
        "tokenizer_name": tokenizer.name,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }
    return tokenizer, train_dataset, val_dataset, metadata


def build_dataloaders(
    config: Config,
) -> tuple[TokenizerWrapper, DataLoader, DataLoader, dict[str, Any]]:
    tokenizer, train_dataset, val_dataset, metadata = build_datasets(config)
    train_is_iterable = isinstance(train_dataset, IterableDataset)
    val_is_iterable = isinstance(val_dataset, IterableDataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=False if train_is_iterable else True,
        num_workers=0 if train_is_iterable else config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        num_workers=0 if val_is_iterable else config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )
    return tokenizer, train_loader, val_loader, metadata
