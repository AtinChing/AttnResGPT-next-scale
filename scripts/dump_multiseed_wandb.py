"""Dump filtered multi-seed W&B runs to JSON. Uses WANDB_API_KEY from .env."""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# Load .env without printing secrets
for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import wandb

ENTITY = "atin5551-uc-davis"
PROJECT = "attnres-next-scale"
PATH = f"{ENTITY}/{PROJECT}"
MULTISEED_GPT_SEEDS = {123, 456, 789}
MIN_FINAL_STEP_GPT = 2900
MIN_FINAL_STEP_VLM = 1  # VLM uses epochs; require eval logged

GPT_RE = re.compile(
    r"^tinystories_(?P<size>\w+)_(?P<arch>baseline|attnres)_ctx(?P<ctx>\d+)_steps(?P<steps>\d+)_seed(?P<seed>\d+)(?P<suffix>_rerun\d+)?$"
)
VLM_RE = re.compile(
    r"^vlm_(?P<arch>baseline|attnres)_flickr30k_b\d+_seed(?P<seed>\d+)(?P<suffix>_crashed|_rerun\d+)?$"
)
LOW_STEP_MARKERS = ("steps2", "steps200", "ctx128", "ctx256")


def _get_nested(cfg: dict, path: str, default=None):
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _block_profile(summary: dict, prefix: str) -> dict[str, dict[str, float]]:
    """prefix e.g. 'mean_activation_norms' or 'gradient_norms'."""
    pattern = re.compile(rf"^{re.escape(prefix)}/blocks\.(\d+)\.(attn|mlp)$")
    blocks: dict[int, dict[str, float]] = defaultdict(dict)
    for key, val in summary.items():
        m = pattern.match(key)
        if m and val is not None:
            blocks[int(m.group(1))][m.group(2)] = float(val)
    ordered: dict[str, dict[str, float]] = {}
    for idx in sorted(blocks):
        ordered[f"blocks.{idx}"] = dict(sorted(blocks[idx].items()))
    return ordered


def _attnres_flags(cfg: dict) -> dict[str, Any]:
    return {
        "enabled": _get_nested(cfg, "model.attnres.enabled"),
        "rmsnorm_keys": _get_nested(cfg, "model.attnres.rmsnorm_keys"),
        "zero_init_queries": _get_nested(cfg, "model.attnres.zero_init_queries"),
        "window_size": _get_nested(cfg, "model.attnres.window_size"),
        "final_readout": _get_nested(cfg, "model.attnres.final_readout"),
    }


def _is_smoke_or_false_start(name: str, steps_in_name: int | None) -> bool:
    lower = name.lower()
    if any(m in lower for m in LOW_STEP_MARKERS):
        return True
    if steps_in_name is not None and steps_in_name < 1000:
        return True
    if "_crashed" in lower or "seedtest" in lower:
        return True
    return False


def _gpt_run_record(run: Any) -> dict[str, Any] | None:
    m = GPT_RE.match(run.name)
    if not m:
        return None
    seed = int(m.group("seed"))
    if seed not in MULTISEED_GPT_SEEDS:
        return None
    steps_budget = int(m.group("steps"))
    if _is_smoke_or_false_start(run.name, steps_budget):
        return None
    if run.state != "finished":
        return None

    s = dict(run.summary)
    val_loss = s.get("val_loss")
    if val_loss is None:
        return None
    final_step = s.get("final_step") or s.get("_step")
    if final_step is None or int(final_step) < MIN_FINAL_STEP_GPT:
        return None

    cfg = dict(run.config)
    return {
        "wandb_name": run.name,
        "wandb_id": run.id,
        "wandb_url": run.url,
        "model_size": m.group("size"),
        "context_length": int(m.group("ctx")),
        "architecture": m.group("arch"),
        "training_steps_budget": steps_budget,
        "seed": seed,
        "rerun_suffix": m.group("suffix") or "",
        "core": {
            "seed": seed,
            "val_loss": float(val_loss),
            "perplexity": float(s["perplexity"]) if s.get("perplexity") is not None else None,
            "second_half_loss": float(s["second_half_loss"])
            if s.get("second_half_loss") is not None
            else None,
            "final_step": int(final_step),
            "total_parameters": s.get("total_parameters") or s.get("parameter_count_total"),
            "trainable_parameters": s.get("trainable_parameters")
            or s.get("parameter_count_trainable"),
        },
        "attnres_config": _attnres_flags(cfg),
        "depth_utilization": {
            "mean_embedding_contribution": _f(s, "mean_embedding_contribution"),
            "mean_early_contribution": _f(s, "mean_early_contribution"),
            "mean_late_contribution": _f(s, "mean_late_contribution"),
            "mean_depth_attention_entropy": _f(s, "mean_depth_attention_entropy"),
        },
        "per_layer_profiles": {
            "mean_activation_norms": _block_profile(s, "mean_activation_norms"),
            "gradient_norms": _block_profile(s, "gradient_norms"),
        },
        "_sort_key": (m.group("arch"), seed, m.group("suffix") or "", run.created_at),
    }


def _vlm_run_record(run: Any) -> dict[str, Any] | None:
    m = VLM_RE.match(run.name)
    if not m:
        return None
    seed = int(m.group("seed"))
    if seed not in MULTISEED_GPT_SEEDS:
        return None
    if _is_smoke_or_false_start(run.name, None):
        return None
    if run.state != "finished":
        return None

    s = dict(run.summary)
    eval_loss = s.get("eval/loss")
    if eval_loss is None:
        return None
    final_step = s.get("final_step") or s.get("epoch") or s.get("_step")
    if final_step is None:
        return None

    return {
        "wandb_name": run.name,
        "wandb_id": run.id,
        "wandb_url": run.url,
        "architecture": m.group("arch"),
        "seed": seed,
        "rerun_suffix": m.group("suffix") or "",
        "vlm": {
            "seed": seed,
            "eval_loss": float(eval_loss),
            "eval_perplexity": float(s["eval/perplexity"])
            if s.get("eval/perplexity") is not None
            else None,
            "final_step": int(final_step) if str(final_step).isdigit() else final_step,
            "alpha_vision_embedding_mean": _f(s, "alpha/vision_embedding_mean")
            or _f(s, "alpha/vision_mean_embedding_final_site"),
            "alpha_language_embedding_mean": _f(s, "alpha/language_embedding_mean"),
            "alpha_vision_entropy_mean": _f(s, "alpha/vision_entropy_mean")
            or _f(s, "alpha/vision_language_entropy"),
            "alpha_language_entropy_mean": _f(s, "alpha/language_entropy_mean")
            or _f(s, "alpha/vision_language_entropy"),
        },
        "_sort_key": (m.group("arch"), seed, m.group("suffix") or "", run.created_at),
    }


def _f(s: dict, key: str) -> float | None:
    v = s.get(key)
    return float(v) if v is not None else None


def _pick_best_per_seed(records: list[dict]) -> list[dict]:
    """One record per seed: prefer _rerun* suffix, then latest created_at."""

    def rank(r: dict) -> tuple:
        suffix = r.get("rerun_suffix") or ""
        return (1 if suffix else 0, suffix, str(r["_sort_key"]))

    by_seed: dict[int, dict] = {}
    for rec in records:
        seed = rec["seed"]
        existing = by_seed.get(seed)
        if existing is None or rank(rec) > rank(existing):
            by_seed[seed] = rec
    out = []
    for r in by_seed.values():
        cleaned = {k: v for k, v in r.items() if k != "_sort_key"}
        out.append(cleaned)
    return sorted(out, key=lambda r: r["seed"])


def _group_gpt(records: list[dict]) -> dict[str, Any]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        gkey = (rec["model_size"], rec["context_length"], rec["architecture"])
        groups[gkey].append(rec)

    result = {}
    for (size, ctx, arch), recs in sorted(groups.items()):
        gname = f"{size}_ctx{ctx}_{arch}"
        result[gname] = {
            "model_size": size,
            "context_length": ctx,
            "architecture": arch,
            "seeds_found": sorted({r["seed"] for r in recs}),
            "n_seeds": len(recs),
            "runs": [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ("model_size", "context_length", "architecture", "training_steps_budget")
                }
                for r in sorted(recs, key=lambda x: x["seed"])
            ],
        }
    return result


def _compare_seed_sets(gpt_groups: dict) -> list[dict]:
    """Pair baseline vs attnres within same (size, ctx)."""
    by_size_ctx: dict[tuple, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for g in gpt_groups.values():
        key = (g["model_size"], g["context_length"])
        by_size_ctx[key][g["architecture"]].update(g["seeds_found"])

    flags = []
    for (size, ctx), arch_seeds in sorted(by_size_ctx.items()):
        base = arch_seeds.get("baseline", set())
        attn = arch_seeds.get("attnres", set())
        if not base or not attn:
            continue
        paired = base == attn
        flags.append(
            {
                "model_size": size,
                "context_length": ctx,
                "baseline_seeds": sorted(base),
                "attnres_seeds": sorted(attn),
                "seeds_match_for_paired_comparison": paired,
                "only_baseline": sorted(base - attn),
                "only_attnres": sorted(attn - base),
            }
        )
    return flags


def main() -> None:
    api = wandb.Api()
    all_runs = list(api.runs(PATH, per_page=200))

    gpt_raw: list[dict] = []
    vlm_raw: list[dict] = []
    skipped: list[dict] = []

    for run in all_runs:
        rec = _gpt_run_record(run)
        if rec is not None:
            gpt_raw.append(rec)
            continue
        rec_v = _vlm_run_record(run)
        if rec_v is not None:
            vlm_raw.append(rec_v)
            continue
        # track near-miss multiseed names for transparency
        if "seed123" in run.name or "seed456" in run.name or "seed789" in run.name:
            if "ctx512" in run.name or "flickr30k" in run.name:
                skipped.append(
                    {
                        "name": run.name,
                        "state": run.state,
                        "val_loss": dict(run.summary).get("val_loss"),
                        "eval_loss": dict(run.summary).get("eval/loss"),
                        "final_step": dict(run.summary).get("final_step"),
                    }
                )

    # dedupe per arch+seed within each group key
    gpt_by_group: dict[tuple, list[dict]] = defaultdict(list)
    for rec in gpt_raw:
        gkey = (rec["model_size"], rec["context_length"], rec["architecture"])
        gpt_by_group[gkey].append(rec)

    gpt_deduped: list[dict] = []
    for gkey, recs in gpt_by_group.items():
        # dedupe within architecture
        arch = gkey[2]
        picked = _pick_best_per_seed(recs)
        gpt_deduped.extend(picked)

    vlm_by_arch: dict[str, list[dict]] = defaultdict(list)
    for rec in vlm_raw:
        vlm_by_arch[rec["architecture"]].append(rec)
    vlm_deduped: list[dict] = []
    for arch, recs in vlm_by_arch.items():
        vlm_deduped.extend(_pick_best_per_seed(recs))

    gpt_groups = _group_gpt(gpt_deduped)
    seed_flags = _compare_seed_sets(gpt_groups)

    vlm_out: dict[str, Any] = {}
    for arch in ("baseline", "attnres"):
        arch_recs = [r for r in vlm_deduped if r["architecture"] == arch]
        if arch_recs:
            vlm_out[arch] = {
                "architecture": arch,
                "seeds_found": sorted({r["seed"] for r in arch_recs}),
                "n_seeds": len(arch_recs),
                "runs": [{k: v for k, v in r.items() if k != "architecture"} for r in arch_recs],
            }

    vlm_seed_flag = None
    if vlm_out:
        b = set(vlm_out.get("baseline", {}).get("seeds_found", []))
        a = set(vlm_out.get("attnres", {}).get("seeds_found", []))
        vlm_seed_flag = {
            "baseline_seeds": sorted(b),
            "attnres_seeds": sorted(a),
            "seeds_match_for_paired_comparison": b == a and len(b) > 0,
            "only_baseline": sorted(b - a),
            "only_attnres": sorted(a - b),
        }

    payload = {
        "project": PATH,
        "filter_notes": {
            "gpt_seeds_expected": sorted(MULTISEED_GPT_SEEDS),
            "gpt_min_final_step": MIN_FINAL_STEP_GPT,
            "excluded": "smoke (steps2/steps200/low ctx), crashed/killed/failed, no val_loss/eval loss, final_step below budget",
            "dedup": "one run per (architecture, seed); prefer _rerun* over original",
        },
        "gpt_groups": gpt_groups,
        "gpt_paired_seed_check": seed_flags,
        "vlm_multiseed": vlm_out,
        "vlm_paired_seed_check": vlm_seed_flag,
        "skipped_near_miss_runs": skipped,
    }

    out_path = Path("outputs/multiseed_wandb_dump.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\n# Wrote {out_path}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
