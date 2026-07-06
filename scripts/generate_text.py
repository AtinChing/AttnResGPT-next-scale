from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.tokenizer import TokenizerWrapper, build_tokenizer
from src.training.eval import load_checkpoint_model
from src.utils.config import Config, load_config
from src.utils.runtime import amp_dtype_from_string, get_device, seed_everything


def latest_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    checkpoints = sorted(path.glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No step_*.pt checkpoints found in {path}")
    return checkpoints[-1]


def infer_config_path(checkpoint_path: Path, output_root: Path) -> Path:
    run_name = checkpoint_path.parent.name
    config_path = output_root / "runs" / run_name / "config.snapshot.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            "Could not infer config snapshot. Pass --config explicitly. "
            f"Tried: {config_path}"
        )
    return config_path


def load_run_tokenizer(config: Config, config_path: Path) -> TokenizerWrapper:
    tokenizer_dir = config_path.parent / "tokenizer"
    if tokenizer_dir.exists():
        backend = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
        if backend.pad_token is None and backend.eos_token is not None:
            backend.pad_token = backend.eos_token
        return TokenizerWrapper(backend=backend, name=str(tokenizer_dir))
    return build_tokenizer(config.data.tokenizer_name)


def filter_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("--temperature must be positive")
    logits = logits / temperature

    if top_k is not None and top_k > 0 and top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumulative_probs > top_p
        remove_sorted[1:] = remove_sorted[:-1].clone()
        remove_sorted[0] = False
        remove_indices = sorted_indices[remove_sorted]
        logits = logits.scatter(0, remove_indices, float("-inf"))

    return logits


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: list[int],
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    greedy: bool,
    eos_token_id: Optional[int],
) -> list[int]:
    model.eval()
    generated = list(input_ids)
    max_seq_len = int(model.config.max_seq_len)
    use_autocast = device.type == "cuda" and amp_dtype in {torch.float16, torch.bfloat16}

    for _ in range(max_new_tokens):
        context = generated[-max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        autocast_context = torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_autocast,
        )
        with autocast_context if use_autocast else nullcontext():
            logits, _ = model(x, return_aux=False)
        next_logits = logits[0, -1, :].float()

        if greedy:
            next_id = int(torch.argmax(next_logits).item())
        else:
            filtered = filter_logits(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            probs = torch.softmax(filtered, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        generated.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a pretraining checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to step_*.pt, or a checkpoint directory.")
    parser.add_argument("--config", default=None, help="Path to config.snapshot.yaml. Inferred from checkpoint when omitted.")
    parser.add_argument("--output-root", default="outputs", help="Root containing runs/ and checkpoints/.")
    parser.add_argument("--model", choices=["baseline", "attnres", "block_attnres"], default=None)
    parser.add_argument("--prompt", default="", help="Prompt text. Empty prompt starts from EOS when available.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true", help="Use argmax decoding instead of sampling.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp-dtype", default=None, help="Override config.training.amp_dtype.")
    parser.add_argument("--overrides", nargs="*", default=[], help="Optional key=value config overrides.")
    args = parser.parse_args()

    checkpoint_path = latest_checkpoint(Path(args.checkpoint))
    config_path = Path(args.config) if args.config else infer_config_path(checkpoint_path, Path(args.output_root))

    overrides = list(args.overrides)
    if args.model is not None:
        overrides.append(f"model.architecture={args.model}")
    config = load_config(config_path, overrides=overrides)
    tokenizer = load_run_tokenizer(config, config_path)
    config.model.vocab_size = tokenizer.vocab_size

    device = get_device(args.device)
    amp_dtype = amp_dtype_from_string(args.amp_dtype or config.training.amp_dtype)
    seed_everything(args.seed)

    model = load_checkpoint_model(config, checkpoint_path, device)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        fallback_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        prompt_ids = [fallback_id]

    output_ids = generate(
        model,
        prompt_ids,
        device=device,
        amp_dtype=amp_dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        greedy=args.greedy,
        eos_token_id=tokenizer.eos_token_id,
    )

    print(f"checkpoint: {checkpoint_path}")
    print(f"config: {config_path}")
    print(f"model: {config.model.architecture}")
    print()
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
