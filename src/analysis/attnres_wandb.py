from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from src.data.dataset import build_dataloaders
from src.training.eval import load_checkpoint_model
from src.utils.config import Config, load_config
from src.utils.runtime import ensure_dir, get_device


CHECKPOINT_RE = re.compile(r"(?:^|/)step_(\d+)\.pt$")


@dataclass
class AlphaMatrixResult:
    checkpoint_step: int
    rows: list[list[float]]
    source_indices: list[list[int]]
    entropy_per_site: list[float]
    embedding_contribution_per_site: list[float]


def login_wandb_from_env() -> wandb.Api:
    wandb.login()
    return wandb.Api()


def _artifact_entry_names(artifact: Any) -> list[str]:
    return sorted(artifact.manifest.entries.keys())


def _checkpoint_entries(artifact: Any) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    for name in _artifact_entry_names(artifact):
        match = CHECKPOINT_RE.search(name)
        if match:
            entries.append((int(match.group(1)), name))
    return sorted(entries)


def find_logged_checkpoint_artifact(
    api: wandb.Api,
    *,
    run_path: str,
    explicit_artifact: str | None = None,
) -> Any:
    if explicit_artifact:
        return api.artifact(explicit_artifact)

    run = api.run(run_path)
    candidates = []
    for artifact in run.logged_artifacts():
        checkpoints = _checkpoint_entries(artifact)
        if checkpoints:
            candidates.append((len(checkpoints), artifact))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-bearing logged artifacts found for {run_path}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def download_checkpoint_from_artifact(
    artifact: Any,
    *,
    step: int | None,
    target_dir: str | Path,
) -> tuple[Path, int]:
    checkpoints = _checkpoint_entries(artifact)
    if not checkpoints:
        raise FileNotFoundError(f"No step_XXXX.pt checkpoints found in artifact {artifact.name}")

    if step is None:
        resolved_step, entry_name = checkpoints[-1]
    else:
        matches = [item for item in checkpoints if item[0] == step]
        if not matches:
            available = [item[0] for item in checkpoints]
            raise FileNotFoundError(f"Checkpoint step {step} not found. Available steps: {available}")
        resolved_step, entry_name = matches[0]

    local_path = Path(artifact.get_path(entry_name).download(root=str(target_dir)))
    return local_path, resolved_step


def prepare_text_config(
    config_path: str | Path,
    *,
    context: int,
    eval_batch_size: int,
    max_val_tokens: int | None = None,
) -> Config:
    overrides = [
        "model.architecture=attnres",
        f"data.block_size={context}",
        f"model.max_seq_len={context}",
        f"data.eval_batch_size={eval_batch_size}",
        f"data.batch_size={eval_batch_size}",
    ]
    if max_val_tokens is not None:
        overrides.append(f"data.max_val_tokens={max_val_tokens}")
    return load_config(config_path, overrides=overrides)


def first_val_batch(config: Config) -> dict[str, torch.Tensor]:
    _tokenizer, _train_loader, val_loader, _meta = build_dataloaders(config)
    return next(iter(val_loader))


def _row_entropy(row: Sequence[float]) -> float:
    array = np.asarray(row, dtype=np.float64)
    array = np.clip(array, 1e-8, 1.0)
    return float(-(array * np.log(array)).sum())


def _embedding_contribution_per_site(
    rows: Sequence[Sequence[float]],
    source_indices: Sequence[Sequence[int]],
) -> list[float]:
    values: list[float] = []
    for row, indices in zip(rows, source_indices):
        if 0 in indices:
            values.append(float(row[indices.index(0)]))
        else:
            values.append(0.0)
    return values


def alpha_matrix_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: Config,
    device: torch.device,
    batch: dict[str, torch.Tensor],
) -> AlphaMatrixResult:
    working = deepcopy(config)
    tokenizer, _, _, _ = build_dataloaders(working)
    working.model.vocab_size = tokenizer.vocab_size
    model = load_checkpoint_model(working, checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        _logits, aux = model(input_ids, return_aux=True)

    rows = [row.tolist() if hasattr(row, "tolist") else list(row) for row in aux.get("depth_attention_rows", [])]
    source_indices = [list(indices) for indices in aux.get("depth_source_indices", [])]
    entropy = [_row_entropy(row) for row in rows]
    embedding = _embedding_contribution_per_site(rows, source_indices)
    checkpoint_step = int(CHECKPOINT_RE.search(str(checkpoint_path)).group(1)) if CHECKPOINT_RE.search(str(checkpoint_path)) else -1
    return AlphaMatrixResult(
        checkpoint_step=checkpoint_step,
        rows=rows,
        source_indices=source_indices,
        entropy_per_site=entropy,
        embedding_contribution_per_site=embedding,
    )


def load_alpha_results_from_run(
    api: wandb.Api,
    *,
    run_path: str,
    config_path: str | Path,
    context: int,
    steps: Sequence[int | None],
    eval_batch_size: int,
    download_root: str | Path,
    explicit_artifact: str | None = None,
    max_val_tokens: int | None = None,
    device: str = "auto",
) -> list[AlphaMatrixResult]:
    artifact = find_logged_checkpoint_artifact(api, run_path=run_path, explicit_artifact=explicit_artifact)
    config = prepare_text_config(
        config_path,
        context=context,
        eval_batch_size=eval_batch_size,
        max_val_tokens=max_val_tokens,
    )
    batch = first_val_batch(config)
    torch_device = get_device(device)
    results: list[AlphaMatrixResult] = []
    for step in steps:
        checkpoint_path, resolved_step = download_checkpoint_from_artifact(
            artifact,
            step=step,
            target_dir=Path(download_root) / artifact.name.replace(":", "_"),
        )
        result = alpha_matrix_from_checkpoint(
            checkpoint_path,
            config=config,
            device=torch_device,
            batch=batch,
        )
        result.checkpoint_step = resolved_step
        results.append(result)
    return results


def pad_alpha_rows(rows: Sequence[Sequence[float]]) -> np.ndarray:
    width = max(len(row) for row in rows)
    padded = np.full((len(rows), width), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        padded[index, : len(row)] = np.asarray(row, dtype=np.float64)
    return padded


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def plot_temporal_heatmaps(results: Sequence[AlphaMatrixResult], *, cmap: str = "viridis") -> plt.Figure:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 8), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    vmax = max(float(np.nanmax(pad_alpha_rows(result.rows))) for result in results)
    for axis, result in zip(axes, results):
        image = axis.imshow(pad_alpha_rows(result.rows), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
        axis.set_title(f"step {result.checkpoint_step}")
        axis.set_xlabel("Source slot")
        axis.set_ylabel("Depth-mixing site")
    fig.colorbar(image, ax=axes, label="Mean alpha weight")
    fig.suptitle("Temporal Evolution of AttnRes Alpha Heatmaps")
    return fig


def plot_entropy_by_site(results: Sequence[AlphaMatrixResult]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for result in results:
        ax.plot(
            list(range(len(result.entropy_per_site))),
            result.entropy_per_site,
            label=f"step {result.checkpoint_step}",
        )
    ax.set_xlabel("Depth-mixing site")
    ax.set_ylabel("Entropy")
    ax.set_title("Depth-attention Entropy by Mixing Site")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_scale_heatmaps(results: Sequence[tuple[str, AlphaMatrixResult]], *, cmap: str = "viridis") -> plt.Figure:
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 8), constrained_layout=True)
    if len(results) == 1:
        axes = [axes]
    vmax = max(float(np.nanmax(pad_alpha_rows(result.rows))) for _, result in results)
    for axis, (label, result) in zip(axes, results):
        image = axis.imshow(pad_alpha_rows(result.rows), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=vmax)
        axis.set_title(label)
        axis.set_xlabel("Source slot")
        axis.set_ylabel("Depth-mixing site")
    fig.colorbar(image, ax=axes, label="Mean alpha weight")
    fig.suptitle("Scale Comparison of AttnRes Alpha Heatmaps")
    return fig


def plot_embedding_contribution(results: Sequence[tuple[str, AlphaMatrixResult]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for label, result in results:
        ax.plot(
            list(range(len(result.embedding_contribution_per_site))),
            result.embedding_contribution_per_site,
            label=label,
        )
    ax.set_xlabel("Depth-mixing site")
    ax.set_ylabel("Embedding contribution")
    ax.set_title("Embedding Contribution by Mixing Site")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def log_figure_to_run(
    *,
    entity: str,
    project: str,
    run_id: str | None,
    run_name: str,
    local_path: str | Path,
    key: str,
    notes: dict[str, Any] | None = None,
) -> None:
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        name=run_name,
        resume="allow" if run_id else None,
        job_type="analysis",
    )
    payload: dict[str, Any] = {key: wandb.Image(str(local_path))}
    if notes:
        for note_key, note_value in notes.items():
            if isinstance(note_value, (int, float, str, bool)):
                run.summary[note_key] = note_value
    run.log(payload)
    artifact_name = f"{run_name.replace('/', '_')}_{Path(local_path).stem}"
    artifact = wandb.Artifact(artifact_name, type="analysis-figure")
    artifact.add_file(str(local_path), name=Path(local_path).name)
    run.log_artifact(artifact, aliases=["latest"])
    run.finish()
