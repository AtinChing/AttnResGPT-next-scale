from __future__ import annotations

import argparse
import sys
from pathlib import Path

import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vlm_attnres_flickr30k import run_vlm
from src.analysis.attnres_wandb import (
    load_alpha_results_from_run,
    log_figure_to_run,
    login_wandb_from_env,
    plot_embedding_contribution,
    plot_entropy_by_site,
    plot_scale_heatmaps,
    plot_temporal_heatmaps,
    save_figure,
)


def _repo_root() -> Path:
    return PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AttnRes paper-style analysis pipeline and then start VLM training.")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default="attnres-next-scale")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--max-val-tokens", type=int, default=200000)
    parser.add_argument("--download-root", default="outputs/research_pipeline/downloads")

    parser.add_argument("--large-run", default=None)
    parser.add_argument("--large-artifact", default=None)
    parser.add_argument("--large-config", default="configs/large_ctx512_3000.yaml")

    parser.add_argument("--small-run", default=None)
    parser.add_argument("--small-artifact", default=None)
    parser.add_argument("--small-config", default="configs/small.yaml")

    parser.add_argument("--medium-run", default=None)
    parser.add_argument("--medium-artifact", default=None)
    parser.add_argument("--medium-config", default="configs/medium.yaml")

    parser.add_argument("--scale-run-name", default="attnres_scale_comparison")
    parser.add_argument("--skip-part3", action="store_true")

    parser.add_argument("--vlm-run-name", default="vlm_attnres_flickr30k")
    parser.add_argument("--vlm-vision-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--vlm-dataset-name", default="Mozilla/flickr30k-transformed-captions")
    parser.add_argument("--vlm-dataset-split", default="train")
    parser.add_argument("--vlm-tokenizer-name", default="gpt2")
    parser.add_argument("--vlm-decoder-config", default="configs/large_ctx512_3000.yaml")
    parser.add_argument("--vlm-decoder-backend", default="gpt_attnres_large")
    parser.add_argument("--vlm-batch-size", type=int, default=1)
    parser.add_argument("--vlm-max-examples", type=int, default=20000)
    parser.add_argument("--vlm-max-text-tokens", type=int, default=128)
    parser.add_argument("--vlm-num-workers", type=int, default=4)
    parser.add_argument("--vlm-epochs", type=int, default=3)
    parser.add_argument("--vlm-learning-rate", type=float, default=1e-4)
    parser.add_argument("--vlm-weight-decay", type=float, default=0.01)
    parser.add_argument("--vlm-warmup-steps", type=int, default=100)
    parser.add_argument("--vlm-checkpoint-interval", type=int, default=500)
    parser.add_argument("--vlm-eval-max-batches", type=int, default=None)
    parser.add_argument("--vlm-alpha-eval-max-batches", type=int, default=None)
    parser.add_argument("--vlm-plateau-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _default_run(entity: str | None, project: str, size: str) -> str:
    if entity is None:
        raise ValueError(f"Provide --entity or --{size}-run explicitly")
    return f"{entity}/{project}/tinystories_{size}_attnres_ctx512_steps3000_seed42"


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    outputs_root = repo_root / "outputs" / "research_pipeline"
    outputs_root.mkdir(parents=True, exist_ok=True)
    download_root = repo_root / args.download_root

    api = login_wandb_from_env()
    entity = args.entity
    if entity is None and not all([args.large_run, args.small_run, args.medium_run]):
        raise ValueError("Provide --entity or explicitly pass --small-run, --medium-run, and --large-run.")

    large_run = args.large_run or _default_run(entity, args.project, "large")
    small_run = args.small_run or _default_run(entity, args.project, "small")
    medium_run = args.medium_run or _default_run(entity, args.project, "medium")

    temporal_results = load_alpha_results_from_run(
        api,
        run_path=large_run,
        config_path=repo_root / args.large_config,
        context=args.context,
        steps=[1000, 2000, 3000],
        eval_batch_size=args.eval_batch_size,
        download_root=download_root / "part1",
        explicit_artifact=args.large_artifact,
        max_val_tokens=args.max_val_tokens,
        device=args.device,
    )
    temporal_heatmap = save_figure(
        plot_temporal_heatmaps(temporal_results),
        outputs_root / "part1_temporal_alpha_heatmaps.png",
    )
    temporal_entropy = save_figure(
        plot_entropy_by_site(temporal_results),
        outputs_root / "part1_temporal_entropy.png",
    )
    large_run_id = large_run.split("/")[-1]
    log_figure_to_run(
        entity=entity,
        project=args.project,
        run_id=large_run_id,
        run_name=large_run_id,
        local_path=temporal_heatmap,
        key="analysis/temporal_alpha_heatmaps",
        notes={"analysis/latest_temporal_heatmaps_path": str(temporal_heatmap)},
    )
    log_figure_to_run(
        entity=entity,
        project=args.project,
        run_id=large_run_id,
        run_name=large_run_id,
        local_path=temporal_entropy,
        key="analysis/temporal_alpha_entropy",
        notes={"analysis/latest_temporal_entropy_path": str(temporal_entropy)},
    )

    scale_specs = [
        ("20M", small_run, repo_root / args.small_config, args.small_artifact),
        ("60M", medium_run, repo_root / args.medium_config, args.medium_artifact),
        ("90M", large_run, repo_root / args.large_config, args.large_artifact),
    ]
    scale_results = []
    for label, run_path, config_path, artifact_name in scale_specs:
        result = load_alpha_results_from_run(
            api,
            run_path=run_path,
            config_path=config_path,
            context=args.context,
            steps=[None],
            eval_batch_size=args.eval_batch_size,
            download_root=download_root / "part2" / label,
            explicit_artifact=artifact_name,
            max_val_tokens=args.max_val_tokens,
            device=args.device,
        )[0]
        scale_results.append((label, result))

    scale_heatmap = save_figure(
        plot_scale_heatmaps(scale_results),
        outputs_root / "part2_scale_alpha_heatmaps.png",
    )
    scale_embedding = save_figure(
        plot_embedding_contribution(scale_results),
        outputs_root / "part2_scale_embedding_contribution.png",
    )
    scale_run = wandb.init(
        entity=entity,
        project=args.project,
        name=args.scale_run_name,
        id=args.scale_run_name,
        resume="allow",
        job_type="analysis",
        config=vars(args),
    )
    scale_run.log(
        {
            "analysis/scale_alpha_heatmaps": wandb.Image(str(scale_heatmap)),
            "analysis/scale_embedding_contribution": wandb.Image(str(scale_embedding)),
        }
    )
    artifact = wandb.Artifact(f"{args.scale_run_name}_figures", type="analysis-figure")
    artifact.add_file(str(scale_heatmap), name=scale_heatmap.name)
    artifact.add_file(str(scale_embedding), name=scale_embedding.name)
    scale_run.log_artifact(artifact, aliases=["latest"])
    scale_run.summary["part2_scale_heatmaps_path"] = str(scale_heatmap)
    scale_run.summary["part2_scale_embedding_path"] = str(scale_embedding)
    scale_run.finish()

    if args.skip_part3:
        return

    vlm_args = argparse.Namespace(
        project=args.project,
        entity=entity,
        run_name=args.vlm_run_name,
        vision_model=args.vlm_vision_model,
        dataset_name=args.vlm_dataset_name,
        dataset_split=args.vlm_dataset_split,
        tokenizer_name=args.vlm_tokenizer_name,
        decoder_config=args.vlm_decoder_config,
        decoder_backend=args.vlm_decoder_backend,
        device=args.device,
        batch_size=args.vlm_batch_size,
        max_examples=args.vlm_max_examples,
        max_text_tokens=args.vlm_max_text_tokens,
        num_workers=args.vlm_num_workers,
        epochs=args.vlm_epochs,
        learning_rate=args.vlm_learning_rate,
        weight_decay=args.vlm_weight_decay,
        warmup_steps=args.vlm_warmup_steps,
        checkpoint_interval=args.vlm_checkpoint_interval,
        eval_max_batches=args.vlm_eval_max_batches,
        alpha_eval_max_batches=args.vlm_alpha_eval_max_batches,
        plateau_patience=args.vlm_plateau_patience,
        seed=args.seed,
    )
    run_vlm(vlm_args)


if __name__ == "__main__":
    main()
