from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from src.vlm.ablation.config import AblationExperimentConfig, VARIANTS


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_wandb_mode(config: AblationExperimentConfig) -> tuple[bool, str]:
    if not config.wandb_enabled:
        return False, "disabled"
    env_mode = os.getenv("WANDB_MODE", "").strip().lower()
    if env_mode == "disabled" or _env_truthy("WANDB_DISABLED"):
        return False, "disabled"
    if env_mode in {"online", "offline"}:
        return True, env_mode
    if config.wandb_mode in {"online", "offline", "disabled"}:
        if config.wandb_mode == "disabled":
            return False, "disabled"
        return True, config.wandb_mode
    if os.getenv("WANDB_API_KEY"):
        return True, "online"
    return True, "offline"


def _flatten(payload: Mapping[str, Any], *, prefix: str = "", allow_strings: bool = False) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, prefix=name, allow_strings=allow_strings))
            continue
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, (int, float, bool)):
            flattened[name] = value
        elif allow_strings and isinstance(value, str):
            flattened[name] = value
    return flattened


def stable_run_id(variant: str, seed: int, config_hash: str) -> str:
    return f"vlm_ablation_{variant}_seed{seed}_{config_hash}"


def stable_run_name(variant: str, seed: int, config_hash: str, run_mode: str) -> str:
    return f"{variant}_seed{seed}_{run_mode}_{config_hash}"


class AblationWandbLogger:
    """Per variant/seed W&B logger mirroring the repo's OptionalWandbLogger style."""

    def __init__(
        self,
        *,
        config: AblationExperimentConfig,
        variant: str,
        seed: int,
        config_hash: str,
        run_dir: Path,
        extra_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.enabled = False
        self.mode = "disabled"
        self.project = config.wandb_project
        self.entity = config.wandb_entity
        self.run_id: str | None = None
        self.run_name: str | None = None
        self.url: str | None = None
        self.error: str | None = None
        self._run = None

        enabled, mode = resolve_wandb_mode(config)
        if not enabled:
            return

        try:
            import wandb
        except Exception as error:  # noqa: BLE001
            self.error = f"wandb import failed: {error}"
            return

        residual = VARIANTS[variant]
        run_id = stable_run_id(variant, seed, config_hash)
        run_name = stable_run_name(variant, seed, config_hash, config.run_mode)
        init_kwargs: dict[str, Any] = {
            "project": config.wandb_project,
            "entity": config.wandb_entity or None,
            "name": run_name,
            "mode": mode,
            "dir": str(run_dir),
            "job_type": "train",
            "group": f"vlm_ablation_{config.run_mode}_{config_hash}",
            "tags": sorted(
                {
                    "vlm-ablation",
                    config.run_mode,
                    variant,
                    f"seed{seed}",
                    residual["encoder"],
                    residual["decoder"],
                    *list(config.wandb_tags),
                }
            ),
            "config": {
                **config.to_dict(),
                "variant": variant,
                "seed": seed,
                "config_hash": config_hash,
                "encoder_residual": residual["encoder"],
                "decoder_residual": residual["decoder"],
                **dict(extra_config or {}),
            },
        }
        if config.wandb_resume != "never":
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = config.wandb_resume
        init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}

        try:
            run = wandb.init(**init_kwargs)
        except Exception as error:  # noqa: BLE001
            if mode != "online":
                self.error = f"wandb init failed: {error}"
                return
            init_kwargs["mode"] = "offline"
            try:
                run = wandb.init(**init_kwargs)
                mode = "offline"
                self.error = f"wandb online init failed, fell back to offline: {error}"
            except Exception as offline_error:  # noqa: BLE001
                self.error = f"wandb init failed: {error}; offline fallback failed: {offline_error}"
                return

        self._run = run
        self.enabled = True
        self.mode = mode
        self.run_id = getattr(run, "id", run_id)
        self.run_name = getattr(run, "name", run_name)
        try:
            self.url = run.get_url() if hasattr(run, "get_url") else None
        except Exception:  # noqa: BLE001
            self.url = None

    def metadata(self) -> dict[str, Any]:
        return {
            "wandb_enabled": self.enabled,
            "wandb_mode": self.mode,
            "wandb_project": self.project,
            "wandb_entity": self.entity,
            "wandb_run_id": self.run_id,
            "wandb_run_name": self.run_name,
            "wandb_url": self.url,
            "wandb_error": self.error,
        }

    def log(self, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        if self._run is None:
            return
        metrics = _flatten(payload)
        metrics.pop("step", None)
        if metrics:
            self._run.log(metrics, step=step)

    def update_summary(self, payload: Mapping[str, Any]) -> None:
        if self._run is None:
            return
        for key, value in _flatten(payload, allow_strings=True).items():
            self._run.summary[key] = value

    def log_image(self, key: str, path: Path, *, step: int | None = None) -> None:
        if self._run is None:
            return
        try:
            import wandb
        except Exception:  # noqa: BLE001
            return
        payload = {key: wandb.Image(str(path))}
        self._run.log(payload, step=step)

    def finish(self, *, status: str = "completed") -> None:
        if self._run is None:
            return
        exit_code = 0 if status == "completed" else 1
        try:
            self._run.finish(exit_code=exit_code)
        except TypeError:
            self._run.finish()
        except Exception:  # noqa: BLE001
            return
        finally:
            self._run = None
