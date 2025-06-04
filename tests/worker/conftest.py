# SPDX-License-Identifier: Apache-2.0
import pytest


@pytest.fixture(scope="function", autouse=True)
def use_v0_only(monkeypatch):
    """
    This module tests V0 internals, so set VLLM_LOGITS_USE_V1=0.
    """
    monkeypatch.setenv('VLLM_LOGITS_USE_V1', '0')