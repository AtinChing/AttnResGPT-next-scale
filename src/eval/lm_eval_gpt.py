from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval import utils as lm_eval_utils
from lm_eval.api.model import TemplateLM
from lm_eval.models.utils import Collator
from lm_eval.models.utils_hf import pad_and_concat

from src.data.tokenizer import build_tokenizer
from src.training.eval import load_checkpoint_model
from src.utils.config import Config, load_config
from src.utils.runtime import amp_dtype_from_string, get_device


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


class AttnResGPTLM(TemplateLM):
    """lm-evaluation-harness adapter for this repo's GPT checkpoints."""

    backend = "causal"

    def __init__(
        self,
        *,
        config: Config,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
        batch_size: int = 8,
        max_length: int | None = None,
        mixed_precision: bool | None = None,
        amp_dtype: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self._device = torch.device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length or config.model.max_seq_len)
        self.mixed_precision = (
            config.training.mixed_precision if mixed_precision is None else mixed_precision
        )
        self.amp_dtype = amp_dtype_from_string(
            amp_dtype or config.training.amp_dtype,
        )
        self._use_autocast = self._device.type == "cuda" and self.mixed_precision

        tokenizer = build_tokenizer(config.data.tokenizer_name)
        config.model.vocab_size = tokenizer.vocab_size
        self.tokenizer = tokenizer.backend
        self._tokenizer_name = tokenizer.name

        self.model = load_checkpoint_model(config, self.checkpoint_path, self._device)
        self.model.eval()

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.global_step = int(checkpoint.get("global_step", -1))
        self.tokens_seen = int(checkpoint.get("cumulative_tokens_seen", -1))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def eot_token_id(self) -> int:
        if self.tokenizer.eos_token_id is not None:
            return int(self.tokenizer.eos_token_id)
        if self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        raise ValueError("Tokenizer must define eos_token_id or pad_token_id")

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    def tok_encode(self, string: str, add_special_tokens: bool | None = None) -> list[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        return list(
            self.tokenizer(
                string,
                add_special_tokens=add_special_tokens,
            )["input_ids"]
        )

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        autocast = torch.autocast(
            device_type=self._device.type,
            dtype=self.amp_dtype,
            enabled=self._use_autocast,
        )
        with autocast if self._use_autocast else nullcontext():
            logits, _ = self.model(inps, return_aux=False)
        return logits.float()

    def _select_cont_toks(
        self,
        logits: torch.Tensor,
        *,
        contlen: int,
        inplen: int,
    ) -> torch.Tensor:
        return logits[inplen - contlen : inplen]

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        res: list[tuple[float, bool]] = []
        batch_size = override_bs or self.batch_size

        def _collate(
            req: tuple[tuple[str, str], list[int], list[int]],
        ) -> tuple[int, tuple[int, ...]]:
            context_enc, continuation_enc = req[1], req[2]
            return -(len(context_enc) + len(continuation_enc)), tuple(context_enc + continuation_enc)

        # Sort for efficient batching, then restore the caller's request order.
        # Without get_original(), multi-task harness runs assign scores to the wrong examples.
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(n=batch_size)

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="loglikelihood",
        )
        for chunk in chunks:
            inps: list[torch.Tensor] = []
            cont_toks_list: list[list[int]] = []
            inplens: list[int] = []
            padding_len_inp: int | None = None

            for _request_pair, context_enc, continuation_enc in chunk:
                assert context_enc and continuation_enc
                total_length = len(context_enc) + len(continuation_enc)
                if total_length > self.max_length + 1:
                    trim = total_length - self.max_length - 1
                    context_enc = context_enc[trim:]
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self._device,
                )
                inplen = int(inp.numel())
                padding_len_inp = inplen if padding_len_inp is None else max(padding_len_inp, inplen)
                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            assert padding_len_inp is not None
            batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")
            multi_logits = F.log_softmax(self._model_call(batched_inps), dim=-1)

            for _request_pair, logits_row, inplen, cont_toks in zip(
                chunk,
                multi_logits,
                inplens,
                cont_toks_list,
                strict=True,
            ):
                contlen = len(cont_toks)
                ctx_len = inplen + (logits_row.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits_row, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)
                cont_tensor = torch.tensor(
                    cont_toks,
                    dtype=torch.long,
                    device=self._device,
                ).unsqueeze(0)
                greedy_match = (logits.argmax(dim=-1) == cont_tensor).all().item()
                token_logprobs = torch.gather(logits, 2, cont_tensor.unsqueeze(-1)).squeeze(-1)
                res.append((float(token_logprobs.sum().item()), bool(greedy_match)))
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self,
        requests: list["Instance"],
        disable_tqdm: bool = False,
    ) -> list[float]:
        all_windows: list[tuple[int, tuple[Any, ...]]] = []
        request_window_counts: list[int] = []

        for req_idx, (string,) in enumerate(
            tqdm(
                [req.args for req in requests],
                disable=disable_tqdm,
                desc="tokenizing rolling windows",
            )
        ):
            rolling_token_windows = list(
                map(
                    lm_eval_utils.make_disjoint_window,
                    lm_eval_utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            windows = [(None,) + window for window in rolling_token_windows]
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        all_nlls: list[tuple[int, tuple[float, bool]]] = []
        for index in range(0, len(all_windows), self.batch_size):
            batch = all_windows[index : index + self.batch_size]
            batch_indices, batch_windows = zip(*batch, strict=True)
            batch_nlls = self._loglikelihood_tokens(
                list(batch_windows),
                disable_tqdm=True,
                override_bs=len(batch_windows),
            )
            all_nlls.extend(zip(batch_indices, batch_nlls, strict=True))

        loglikelihoods: list[float] = []
        cursor = 0
        for window_count in request_window_counts:
            request_nlls = all_nlls[cursor : cursor + window_count]
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            cursor += window_count
            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), request_total)

        return loglikelihoods

    def generate_until(self, requests: list["Instance"], disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError(
            "AttnResGPTLM only supports loglikelihood tasks (hellaswag, lambada_openai, arc_easy)."
        )
