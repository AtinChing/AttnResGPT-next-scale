"""Guards for the lm-eval request-reorder fix (Collator + get_original)."""
from __future__ import annotations

import inspect
from pathlib import Path

from src.eval.lm_eval_gpt import AttnResGPTLM


def test_loglikelihood_uses_collator_get_original() -> None:
    source = inspect.getsource(AttnResGPTLM._loglikelihood_tokens)
    assert "Collator(" in source
    assert "get_original(" in source
    assert "get_batched(" in source


def test_benchmark_panel_tasks_cover_requested_suite() -> None:
    from src.eval.benchmark_tasks import PANEL_TASKS, TASK_SPECS

    expected = {
        "hellaswag",
        "lambada_openai",
        "piqa",
        "winogrande",
        "arc_easy",
        "arc_challenge",
        "openbookqa",
        "boolq",
        "sciq",
    }
    assert set(PANEL_TASKS) == expected
    for task in PANEL_TASKS:
        assert task in TASK_SPECS
        assert "chance" in TASK_SPECS[task]
        assert "metric" in TASK_SPECS[task]


def test_attnres_lm_defaults_to_auto_device_without_cuda() -> None:
    sig = inspect.signature(AttnResGPTLM.__init__)
    device_param = sig.parameters["device"]
    assert device_param.default is None
