from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.attnres_wandb import pad_alpha_rows, save_figure
from src.data.tokenizer import build_tokenizer
from src.models.vlm_attnres import SiglipAttnResCaptioner, summarize_alpha_by_token_type
from src.training.train import build_scheduler
from src.utils.config import Config, load_config
from src.utils.runtime import amp_dtype_from_string, ensure_dir, get_device
from src.vlm.flickr30k import build_flickr30k_dataloaders


def _repo_root() -> Path:
    return PROJECT_ROOT


def _load_decoder_config(
    config_path: str | Path,
    tokenizer_name: str,
    decoder_architecture: str,
) -> Config:
    config = load_config(
        config_path,
        overrides=[
            f"model.architecture={decoder_architecture}",
            "data.block_size=512",
            "model.max_seq_len=512",
        ],
    )
    tokenizer = build_tokenizer(tokenizer_name)
    config.model.vocab_size = tokenizer.vocab_size
    return config


def _choose_batch_size(args: argparse.Namespace) -> int:
    return args.batch_size if args.batch_size > 0 else 1


def _language_model_metrics(losses: list[float]) -> dict[str, float]:
    mean_loss = float(sum(losses) / max(1, len(losses)))
    return {
        "eval_loss": mean_loss,
        "perplexity": float(torch.exp(torch.tensor(min(mean_loss, 20.0))).item()),
    }


def _global_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().float().norm().item())
        total += grad_norm * grad_norm
    return float(total ** 0.5)


def _flatten_alpha_metrics(summary) -> dict[str, float]:
    metrics: dict[str, float] = {}
    metrics["alpha/vision_entropy_mean"] = float(np.mean(summary.vision_entropy))
    metrics["alpha/language_entropy_mean"] = float(np.mean(summary.language_entropy))
    metrics["alpha/vision_embedding_mean"] = float(np.mean(summary.vision_embedding))
    metrics["alpha/language_embedding_mean"] = float(np.mean(summary.language_embedding))
    for site_index, value in enumerate(summary.vision_entropy):
        metrics[f"alpha/vision_entropy/site_{site_index:02d}"] = float(value)
    for site_index, value in enumerate(summary.language_entropy):
        metrics[f"alpha/language_entropy/site_{site_index:02d}"] = float(value)
    for site_index, value in enumerate(summary.vision_embedding):
        metrics[f"alpha/vision_embedding/site_{site_index:02d}"] = float(value)
    for site_index, value in enumerate(summary.language_embedding):
        metrics[f"alpha/language_embedding/site_{site_index:02d}"] = float(value)
    return metrics


@dataclass
class CachedVisionExample:
    vision_hidden: torch.Tensor
    input_ids: torch.Tensor
    targets: torch.Tensor
    text_mask: torch.Tensor


class CachedVisionDataset(Dataset[CachedVisionExample]):
    def __init__(self, examples: list[CachedVisionExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> CachedVisionExample:
        return self.examples[index]


class CachedVisionCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[CachedVisionExample]) -> dict[str, torch.Tensor]:
        max_len = max(int(example.input_ids.size(0)) for example in examples)
        batch_size = len(examples)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)
        text_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        vision_hidden = torch.stack([example.vision_hidden for example in examples], dim=0)

        for row_index, example in enumerate(examples):
            width = int(example.input_ids.size(0))
            input_ids[row_index, :width] = example.input_ids
            targets[row_index, :width] = example.targets
            text_mask[row_index, :width] = example.text_mask

        return {
            "vision_hidden": vision_hidden,
            "input_ids": input_ids,
            "targets": targets,
            "text_mask": text_mask,
        }


@torch.no_grad()
def _cache_vision_features(
    model: SiglipAttnResCaptioner,
    dataloader: DataLoader,
    *,
    device: torch.device,
    label: str,
    mixed_precision: bool,
    amp_dtype: torch.dtype,
) -> list[CachedVisionExample]:
    model.eval()
    cached: list[CachedVisionExample] = []
    total_batches = len(dataloader)
    started = time.perf_counter()
    for batch_index, batch in enumerate(dataloader, start=1):
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
            if device.type == "cuda" and mixed_precision
            else nullcontext()
        )
        with autocast_context:
            vision_hidden = model.encode_vision(batch["pixel_values"].to(device))
        vision_hidden = vision_hidden.to(torch.float16).cpu()
        input_ids = batch["input_ids"].cpu()
        targets = batch["targets"].cpu()
        text_mask = batch["text_mask"].cpu()
        for row_index in range(vision_hidden.size(0)):
            width = int(text_mask[row_index].sum().item())
            cached.append(
                CachedVisionExample(
                    vision_hidden=vision_hidden[row_index].clone(),
                    input_ids=input_ids[row_index, :width].clone(),
                    targets=targets[row_index, :width].clone(),
                    text_mask=text_mask[row_index, :width].clone(),
                )
            )
        if batch_index == 1 or batch_index % 50 == 0 or batch_index == total_batches:
            elapsed = time.perf_counter() - started
            print(
                f"[cache:{label}] batch {batch_index}/{total_batches} "
                f"examples={len(cached)} elapsed={elapsed:.1f}s",
                flush=True,
            )
    return cached


def _build_cached_loader(
    examples: list[CachedVisionExample],
    *,
    batch_size: int,
    shuffle: bool,
    pad_token_id: int,
) -> DataLoader:
    return DataLoader(
        CachedVisionDataset(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=CachedVisionCollator(pad_token_id),
    )


@torch.no_grad()
def evaluate_captioner(
    model: SiglipAttnResCaptioner,
    dataloader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    max_batches: int | None,
    mixed_precision: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        vision_hidden = batch.get("vision_hidden")
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
            if device.type == "cuda" and mixed_precision
            else nullcontext()
        )
        with autocast_context:
            output = model(
                pixel_values=batch["pixel_values"].to(device) if "pixel_values" in batch else None,
                vision_hidden=vision_hidden.to(device) if vision_hidden is not None else None,
                input_ids=batch["input_ids"].to(device),
                targets=batch["targets"].to(device),
                return_aux=False,
            )
        losses.append(float(output["loss"].item()))
    return _language_model_metrics(losses)


def _plot_vlm_alpha(summary, *, title_prefix: str) -> tuple[plt.Figure, plt.Figure, plt.Figure]:
    heatmap_fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=True)
    vision_image = axes[0].imshow(pad_alpha_rows(summary.vision_rows), aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0].set_title(f"{title_prefix} Vision tokens")
    axes[0].set_xlabel("Source slot")
    axes[0].set_ylabel("Depth-mixing site")
    language_image = axes[1].imshow(pad_alpha_rows(summary.language_rows), aspect="auto", interpolation="nearest", cmap="viridis")
    axes[1].set_title(f"{title_prefix} Language tokens")
    axes[1].set_xlabel("Source slot")
    axes[1].set_ylabel("Depth-mixing site")
    vmax = max(float(np.nanmax(pad_alpha_rows(summary.vision_rows))), float(np.nanmax(pad_alpha_rows(summary.language_rows))))
    vision_image.set_clim(0.0, vmax)
    language_image.set_clim(0.0, vmax)
    heatmap_fig.colorbar(language_image, ax=axes, label="Mean alpha weight")

    entropy_fig, entropy_ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    entropy_ax.plot(summary.vision_entropy, label="vision")
    entropy_ax.plot(summary.language_entropy, label="language")
    entropy_ax.set_title(f"{title_prefix} Depth-attention Entropy")
    entropy_ax.set_xlabel("Depth-mixing site")
    entropy_ax.set_ylabel("Entropy")
    entropy_ax.grid(True, alpha=0.3)
    entropy_ax.legend()

    embedding_fig, embedding_ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    embedding_ax.plot(summary.vision_embedding, label="vision")
    embedding_ax.plot(summary.language_embedding, label="language")
    embedding_ax.set_title(f"{title_prefix} Embedding Contribution")
    embedding_ax.set_xlabel("Depth-mixing site")
    embedding_ax.set_ylabel("Embedding contribution")
    embedding_ax.grid(True, alpha=0.3)
    embedding_ax.legend()
    return heatmap_fig, entropy_fig, embedding_fig


def _save_alpha_summary(summary, output_dir: Path, step: int) -> Path:
    payload = {
        "vision_rows": summary.vision_rows,
        "language_rows": summary.language_rows,
        "source_indices": summary.source_indices,
        "vision_entropy": summary.vision_entropy,
        "language_entropy": summary.language_entropy,
        "vision_embedding": summary.vision_embedding,
        "language_embedding": summary.language_embedding,
    }
    path = output_dir / f"alpha_summary_step_{step:07d}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _save_checkpoint(
    *,
    path: Path,
    model: SiglipAttnResCaptioner,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    epoch: int,
    best_eval_loss: float,
    decoder_backend: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_eval_loss": best_eval_loss,
            "decoder_backend": decoder_backend,
        },
        path,
    )


def _log_checkpoint_artifact(
    run: wandb.sdk.wandb_run.Run,
    *,
    checkpoint_path: Path,
    output_paths: list[Path],
    step: int,
) -> None:
    artifact = wandb.Artifact(f"{run.name}_step_{step:07d}", type="checkpoint")
    artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    for path in output_paths:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact, aliases=["latest", f"step-{step:07d}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny SigLIP-prefix VLM with either a baseline or AttnRes GPT decoder.")
    parser.add_argument("--project", default="attnres-next-scale")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--run-name", default="vlm_attnres_flickr30k")
    parser.add_argument("--vision-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--dataset-name", default="Mozilla/flickr30k-transformed-captions")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--tokenizer-name", default="gpt2")
    parser.add_argument("--decoder-config", default="configs/large_ctx512_3000.yaml")
    parser.add_argument("--decoder-architecture", choices=["baseline", "attnres"], default="attnres")
    parser.add_argument("--decoder-backend", default="gpt_attnres_large")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=20000)
    parser.add_argument("--max-text-tokens", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--alpha-eval-max-batches", type=int, default=None)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--mixed-precision", dest="mixed_precision", action="store_true")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false")
    parser.set_defaults(mixed_precision=True)
    parser.add_argument("--amp-dtype", default="float16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_vlm(args: argparse.Namespace) -> None:
    repo_root = _repo_root()
    output_dir = ensure_dir(repo_root / "outputs" / args.run_name)
    checkpoint_dir = ensure_dir(repo_root / "checkpoints" / args.run_name)
    device = get_device(args.device)
    amp_dtype = amp_dtype_from_string(args.amp_dtype)
    resolved_decoder_backend = f"gpt_{args.decoder_architecture}_large"
    if args.decoder_backend != resolved_decoder_backend:
        print(
            f"Requested decoder backend {args.decoder_backend!r}, "
            f"but this repo currently uses the {resolved_decoder_backend!r} fallback for VLM training."
        )

    decoder_config = _load_decoder_config(
        repo_root / args.decoder_config,
        args.tokenizer_name,
        args.decoder_architecture,
    )
    tokenizer = build_tokenizer(args.tokenizer_name)
    model = SiglipAttnResCaptioner(
        vision_model_name=args.vision_model,
        decoder_config=decoder_config.model,
    ).to(device)
    supports_alpha_analysis = model.supports_alpha_analysis
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    use_autocast = device.type == "cuda" and args.mixed_precision
    scaler = torch.amp.GradScaler("cuda", enabled=use_autocast and amp_dtype == torch.float16)

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        id=args.run_name,
        resume="allow",
        config=vars(args),
        job_type="train",
    )
    run.summary["decoder_backend_requested"] = args.decoder_backend
    run.summary["decoder_backend_resolved"] = resolved_decoder_backend
    run.summary["decoder_architecture"] = args.decoder_architecture
    run.summary["supports_alpha_analysis"] = supports_alpha_analysis
    run.summary["device"] = str(device)
    run.summary["mixed_precision"] = bool(args.mixed_precision)
    run.summary["amp_dtype"] = args.amp_dtype
    run.summary["decoder_backend_note"] = "Using the existing GPT decoder implementations in this repo for the VLM path."
    run.summary["dataset_name"] = args.dataset_name
    run.summary["dataset_split"] = args.dataset_split
    run.summary["total_parameters"] = total_parameter_count
    run.summary["trainable_parameters"] = trainable_parameter_count
    run.summary["frozen_parameters"] = total_parameter_count - trainable_parameter_count

    train_loader, val_loader = build_flickr30k_dataloaders(
        dataset_name=args.dataset_name,
        split=args.dataset_split,
        processor=model.processor,
        tokenizer=tokenizer,
        max_examples=args.max_examples,
        max_text_tokens=args.max_text_tokens,
        batch_size=_choose_batch_size(args),
        seed=args.seed,
        num_workers=args.num_workers,
    )
    cache_started = time.perf_counter()
    pad_token_id = int(getattr(tokenizer.backend, "pad_token_id", 0) or 0)
    cached_train_examples = _cache_vision_features(
        model,
        train_loader,
        device=device,
        label="train",
        mixed_precision=args.mixed_precision,
        amp_dtype=amp_dtype,
    )
    cached_val_examples = _cache_vision_features(
        model,
        val_loader,
        device=device,
        label="val",
        mixed_precision=args.mixed_precision,
        amp_dtype=amp_dtype,
    )
    train_loader = _build_cached_loader(
        cached_train_examples,
        batch_size=_choose_batch_size(args),
        shuffle=True,
        pad_token_id=pad_token_id,
    )
    val_loader = _build_cached_loader(
        cached_val_examples,
        batch_size=_choose_batch_size(args),
        shuffle=False,
        pad_token_id=pad_token_id,
    )
    cache_elapsed = time.perf_counter() - cache_started
    run.log(
        {
            "cache/seconds": float(cache_elapsed),
            "cache/train_examples": len(cached_train_examples),
            "cache/val_examples": len(cached_val_examples),
        },
        step=0,
    )

    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = build_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        base_lr=args.learning_rate,
        min_lr=args.learning_rate * 0.1,
    )
    run.summary["train_examples"] = len(train_loader.dataset)
    run.summary["val_examples"] = len(val_loader.dataset)
    run.summary["vision_feature_cache_seconds"] = float(cache_elapsed)
    run.summary["cached_train_examples"] = len(cached_train_examples)
    run.summary["cached_val_examples"] = len(cached_val_examples)

    global_step = 0
    best_eval_loss = float("inf")
    plateau_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            global_step += 1
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)
                if use_autocast
                else nullcontext()
            )
            with autocast_context:
                output = model(
                    pixel_values=batch["pixel_values"].to(device) if "pixel_values" in batch else None,
                    vision_hidden=batch.get("vision_hidden").to(device) if batch.get("vision_hidden") is not None else None,
                    input_ids=batch["input_ids"].to(device),
                    targets=batch["targets"].to(device),
                    return_aux=False,
                )
                loss = output["loss"]

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = _global_grad_norm(trainable_parameters)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            run.log(
                {
                    "train/loss": float(loss.item()),
                    "train/perplexity": float(output["perplexity"]),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train/grad_norm": grad_norm,
                    "epoch": epoch,
                },
                step=global_step,
            )

            should_checkpoint = global_step % args.checkpoint_interval == 0 or global_step == total_steps
            if should_checkpoint:
                metrics = evaluate_captioner(
                    model,
                    val_loader,
                    device=device,
                    max_batches=args.eval_max_batches,
                    mixed_precision=args.mixed_precision,
                    amp_dtype=amp_dtype,
                )
                output_paths: list[Path] = []
                alpha_log_payload: dict[str, Any] = {}
                alpha_summary_path: Path | None = None

                if supports_alpha_analysis:
                    alpha_summary = summarize_alpha_by_token_type(
                        model,
                        val_loader,
                        device=device,
                        max_batches=args.alpha_eval_max_batches,
                        mixed_precision=args.mixed_precision,
                        amp_dtype=amp_dtype,
                    )
                    title_prefix = f"step {global_step}"
                    heatmap_fig, entropy_fig, embedding_fig = _plot_vlm_alpha(alpha_summary, title_prefix=title_prefix)
                    heatmap_path = save_figure(heatmap_fig, output_dir / f"vision_vs_language_heatmap_step_{global_step:07d}.png")
                    entropy_path = save_figure(entropy_fig, output_dir / f"vision_vs_language_entropy_step_{global_step:07d}.png")
                    embedding_path = save_figure(embedding_fig, output_dir / f"vision_vs_language_embedding_step_{global_step:07d}.png")
                    alpha_summary_path = _save_alpha_summary(alpha_summary, output_dir, global_step)
                    output_paths = [heatmap_path, entropy_path, embedding_path, alpha_summary_path]
                    alpha_log_payload = {
                        "alpha/vision_language_heatmap": wandb.Image(str(heatmap_path)),
                        "alpha/vision_language_entropy": wandb.Image(str(entropy_path)),
                        "alpha/vision_language_embedding": wandb.Image(str(embedding_path)),
                        "alpha/vision_mean_embedding_final_site": alpha_summary.vision_embedding[-1],
                        "alpha/language_mean_embedding_final_site": alpha_summary.language_embedding[-1],
                        **_flatten_alpha_metrics(alpha_summary),
                    }

                checkpoint_path = checkpoint_dir / f"step_{global_step:07d}.pt"
                _save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    epoch=epoch,
                    best_eval_loss=min(best_eval_loss, metrics["eval_loss"]),
                    decoder_backend=resolved_decoder_backend,
                )
                _log_checkpoint_artifact(
                    run,
                    checkpoint_path=checkpoint_path,
                    output_paths=output_paths,
                    step=global_step,
                )

                run.log(
                    {
                        "eval/loss": metrics["eval_loss"],
                        "eval/perplexity": metrics["perplexity"],
                        **alpha_log_payload,
                    },
                    step=global_step,
                )
                run.summary["best_eval_loss"] = min(best_eval_loss, metrics["eval_loss"])
                run.summary["last_checkpoint_path"] = str(checkpoint_path)
                if alpha_summary_path is not None:
                    run.summary["last_alpha_summary_path"] = str(alpha_summary_path)

                if metrics["eval_loss"] < best_eval_loss - 1e-6:
                    best_eval_loss = metrics["eval_loss"]
                    plateau_count = 0
                else:
                    plateau_count += 1
                    if plateau_count >= args.plateau_patience:
                        run.summary["stopped_early"] = True
                        run.summary["stop_step"] = global_step
                        run.finish()
                        return

    run.summary["stopped_early"] = False
    run.summary["final_step"] = global_step
    run.finish()


def main() -> None:
    run_vlm(parse_args())


if __name__ == "__main__":
    main()
