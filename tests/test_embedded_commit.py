# SPDX-License-Identifier: Apache-2.0

import vllm_logits


def test_embedded_commit_defined():
    assert hasattr(vllm_logits, "__version__")
    assert hasattr(vllm_logits, "__version_tuple__")
    assert vllm_logits.__version__ != "dev"
    assert vllm_logits.__version_tuple__ != (0, 0, "dev")
