# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest

from vllm_logits.envs import get_vllm_logits_port


def test_get_vllm_logits_port_not_set():
    """Test when VLLM_LOGITS_PORT is not set."""
    with patch.dict(os.environ, {}, clear=True):
        assert get_vllm_logits_port() is None


def test_get_vllm_logits_port_valid():
    """Test when VLLM_LOGITS_PORT is set to a valid integer."""
    with patch.dict(os.environ, {"VLLM_LOGITS_PORT": "5678"}, clear=True):
        assert get_vllm_logits_port() == 5678


def test_get_vllm_logits_port_invalid():
    """Test when VLLM_LOGITS_PORT is set to a non-integer value."""
    with (patch.dict(os.environ, {"VLLM_LOGITS_PORT": "abc"}, clear=True),
          pytest.raises(ValueError, match="must be a valid integer")):
        get_vllm_logits_port()


def test_get_vllm_logits_port_uri():
    """Test when VLLM_LOGITS_PORT is set to a URI."""
    with (patch.dict(os.environ, {"VLLM_LOGITS_PORT": "tcp://localhost:5678"},
                     clear=True),
          pytest.raises(ValueError, match="appears to be a URI")):
        get_vllm_logits_port()
