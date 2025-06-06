# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from functools import lru_cache, partial
from typing import Optional, Union

import torch

from vllm_logits.sampling_params import LogitsProcessor
from vllm_logits.transformers_utils.tokenizer import AnyTokenizer


class AllowedTokenIdsLogitsProcessor:
    """Logits processor for constraining generated tokens to a
    specific set of token ids."""

    def __init__(self, allowed_ids: Iterable[int]):
        self.allowed_ids: Optional[list[int]] = list(allowed_ids)
        self.mask: Optional[torch.Tensor] = None

    def __call__(self, token_ids: list[int],
                 logits: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            self.mask = torch.ones((logits.shape[-1], ),
                                   dtype=torch.bool,
                                   device=logits.device)
            self.mask[self.allowed_ids] = False
            self.allowed_ids = None
        logits.masked_fill_(self.mask, float("-inf"))
        return logits


@lru_cache(maxsize=32)
def _get_allowed_token_ids_logits_processor(
    allowed_token_ids: frozenset[int],
    vocab_size: int,
) -> LogitsProcessor:
    if not allowed_token_ids:
        raise ValueError("Empty allowed_token_ids provided")
    if not all(0 <= tid < vocab_size for tid in allowed_token_ids):
        raise ValueError("allowed_token_ids contains "
                         "out-of-vocab token id")
    return AllowedTokenIdsLogitsProcessor(allowed_token_ids)


def logit_bias_logits_processor(
    logit_bias: dict[int, float],
    token_ids: list[int],
    logits: torch.Tensor,
) -> torch.Tensor:
    for token_id, bias in logit_bias.items():
        logits[token_id] += bias
    return logits


def get_logits_processors(
    logit_bias: Optional[Union[dict[int, float], dict[str, float]]],
    allowed_token_ids: Optional[list[int]],
    tokenizer: AnyTokenizer,
) -> list[LogitsProcessor]:
    logits_processors: list[LogitsProcessor] = []
    if logit_bias:
        try:
            # Convert token_id to integer
            # Clamp the bias between -100 and 100 per OpenAI API spec
            clamped_logit_bias: dict[int, float] = {
                int(token_id): min(100.0, max(-100.0, bias))
                for token_id, bias in logit_bias.items()
            }
        except ValueError as exc:
            raise ValueError(
                "Found token_id in logit_bias that is not "
                "an integer or string representing an integer") from exc

        # Check if token_id is within the vocab size
        for token_id, bias in clamped_logit_bias.items():
            if token_id < 0 or token_id >= len(tokenizer):
                raise ValueError(f"token_id {token_id} in logit_bias contains "
                                 "out-of-vocab token id")

        logits_processors.append(
            partial(logit_bias_logits_processor, clamped_logit_bias))

    if allowed_token_ids is not None:
        logits_processors.append(
            _get_allowed_token_ids_logits_processor(
                frozenset(allowed_token_ids), len(tokenizer)))

    return logits_processors
