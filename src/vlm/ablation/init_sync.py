from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.attnres import DepthAttentionResidual
from src.models.vlm_attnres import TinyAttnResVLM


def _is_attnres_param(name: str) -> bool:
    lowered = name.lower()
    return (
        "pre_attn_res" in lowered
        or "pre_mlp_res" in lowered
        or "final_residual" in lowered
        or name.endswith(".query")
        or ".key_norm." in lowered
    )


def shared_parameter_pairs(reference: TinyAttnResVLM, candidate: TinyAttnResVLM) -> list[tuple[str, str]]:
    """Map shared parameter names between a baseline-compatible reference and a variant."""
    ref_names = {name for name, _ in reference.named_parameters() if not _is_attnres_param(name)}
    cand_names = {name for name, _ in candidate.named_parameters() if not _is_attnres_param(name)}
    shared = sorted(ref_names & cand_names)
    return [(name, name) for name in shared]


def copy_shared_weights(reference: TinyAttnResVLM, candidate: TinyAttnResVLM) -> int:
    ref_state = reference.state_dict()
    cand_state = candidate.state_dict()
    copied = 0
    for ref_name, cand_name in shared_parameter_pairs(reference, candidate):
        if ref_name in ref_state and cand_name in cand_state:
            if cand_state[cand_name].shape == ref_state[ref_name].shape:
                cand_state[cand_name] = ref_state[ref_name].clone()
                copied += 1
    candidate.load_state_dict(cand_state, strict=False)
    return copied


@torch.no_grad()
def validate_shared_initialization(
    reference: TinyAttnResVLM,
    candidate: TinyAttnResVLM,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    mismatches: list[str] = []
    checked = 0
    ref_params = dict(reference.named_parameters())
    cand_params = dict(candidate.named_parameters())
    for ref_name, cand_name in shared_parameter_pairs(reference, candidate):
        left = ref_params[ref_name]
        right = cand_params[cand_name]
        checked += 1
        if left.shape != right.shape or not torch.allclose(left, right, atol=atol, rtol=rtol):
            mismatches.append(ref_name)
    if mismatches:
        raise ValueError(
            "Shared initialization mismatch for parameters: " + ", ".join(mismatches[:20])
        )
    return {
        "checked": checked,
        "mismatches": mismatches,
        "ok": True,
        "reference_encoder": reference.encoder_residual,
        "reference_decoder": reference.decoder_residual,
        "candidate_encoder": candidate.encoder_residual,
        "candidate_decoder": candidate.decoder_residual,
    }


def assert_encoder_decoder_attnres_separate(model: TinyAttnResVLM) -> dict[str, Any]:
    encoder_ids = {id(module) for module in model.iter_encoder_depth_residuals()}
    decoder_ids = {id(module) for module in model.iter_decoder_depth_residuals()}
    overlap = encoder_ids & decoder_ids
    if overlap:
        raise ValueError("Encoder and decoder DepthAttentionResidual modules must be separate")
    encoder_params = {id(param) for param in model.encoder.parameters()}
    decoder_params = {id(param) for param in model.decoder.parameters()}
    if encoder_params & decoder_params:
        raise ValueError("Encoder and decoder parameters must not be shared")
    return {
        "encoder_residual_modules": len(encoder_ids),
        "decoder_residual_modules": len(decoder_ids),
        "ok": True,
    }


def count_attnres_parameters(model: nn.Module) -> int:
    total = 0
    for name, param in model.named_parameters():
        if _is_attnres_param(name) or isinstance(param, nn.Parameter) and name.endswith("query"):
            # Count DepthAttentionResidual owned params via module walk as well.
            pass
    for module in model.modules():
        if isinstance(module, DepthAttentionResidual):
            total += sum(param.numel() for param in module.parameters())
    return total
