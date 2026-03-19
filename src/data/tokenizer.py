from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class TokenizerWrapper:
    backend: PreTrainedTokenizerBase
    name: str

    @property
    def vocab_size(self) -> int:
        return int(len(self.backend))

    def encode(self, text: str) -> list[int]:
        return list(self.backend(text, add_special_tokens=False)["input_ids"])

    def save(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.backend.save_pretrained(str(path))


def build_tokenizer(tokenizer_name: str) -> TokenizerWrapper:
    backend = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if backend.pad_token is None and backend.eos_token is not None:
        backend.pad_token = backend.eos_token
    return TokenizerWrapper(backend=backend, name=tokenizer_name)
