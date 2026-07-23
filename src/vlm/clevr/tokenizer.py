from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<answer>", "<unk>")


def tokenize_clevr_text(text: str) -> list[str]:
    """Whitespace/punctuation tokenizer for official CLEVR questions/answers."""
    lowered = text.lower().strip()
    # Keep alphanumeric tokens; drop punctuation.
    return re.findall(r"[a-z0-9]+", lowered)


class CLEVRTokenizer:
    def __init__(self, token_to_id: dict[str, int] | None = None) -> None:
        if token_to_id is None:
            self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        else:
            self.token_to_id = dict(token_to_id)
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["<pad>"]
        self.bos_token_id = self.token_to_id["<bos>"]
        self.eos_token_id = self.token_to_id["<eos>"]
        self.answer_token_id = self.token_to_id["<answer>"]
        self.unk_token_id = self.token_to_id["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def vocab_hash(self) -> str:
        material = json.dumps(self.token_to_id, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(material).hexdigest()[:16]

    def add_token(self, token: str) -> None:
        if token not in self.token_to_id:
            index = len(self.token_to_id)
            self.token_to_id[token] = index
            self.id_to_token[index] = token

    def encode(self, text: str, *, allow_unk: bool = True) -> list[int]:
        ids: list[int] = []
        for token in tokenize_clevr_text(text):
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            elif allow_unk:
                ids.append(self.unk_token_id)
            else:
                raise KeyError(f"Unknown token {token!r}")
        return ids

    def encode_answer(self, answer: str, *, allow_unk: bool = False) -> int:
        tokens = tokenize_clevr_text(answer)
        if len(tokens) != 1:
            # Official CLEVR answers are single tokens after tokenization
            # (e.g. "yes", "2", "blue"). Keep joined fallback for safety.
            token = "".join(tokens) if tokens else answer.lower().strip()
        else:
            token = tokens[0]
        if token in self.token_to_id:
            return self.token_to_id[token]
        if allow_unk:
            return self.unk_token_id
        raise KeyError(f"Unknown answer token {token!r} from answer={answer!r}")

    def encode_supervised(
        self,
        question: str,
        answer: str,
        *,
        supervise_eos: bool = True,
        allow_unk: bool = True,
    ) -> dict[str, Any]:
        question_ids = self.encode(question, allow_unk=allow_unk)
        answer_id = self.encode_answer(answer, allow_unk=allow_unk)
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

    def to_dict(self) -> dict[str, Any]:
        return {"token_to_id": self.token_to_id, "vocab_hash": self.vocab_hash()}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CLEVRTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(token_to_id=payload["token_to_id"])

    @classmethod
    def build_from_training_questions(cls, questions: Iterable[dict[str, Any]]) -> "CLEVRTokenizer":
        tokenizer = cls()
        for item in questions:
            for token in tokenize_clevr_text(str(item["question"])):
                tokenizer.add_token(token)
            for token in tokenize_clevr_text(str(item["answer"])):
                tokenizer.add_token(token)
        return tokenizer
