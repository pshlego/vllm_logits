# SPDX-License-Identifier: Apache-2.0

from vllm_logits.attention.backends.flash_attn import FlashAttentionBackend


class DummyAttentionBackend(FlashAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "Dummy_Backend"
