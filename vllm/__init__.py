# SPDX-License-Identifier: Apache-2.0
"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""
# The version.py should be independent library, and we always import the
# version library first.  Such assumption is critical for some customization.
from .version import __version__, __version_tuple__  # isort:skip

# The environment variables override should be imported before any other
# modules to ensure that the environment variables are set before any
# other modules are imported.
import vllm_logits.env_override  # isort:skip  # noqa: F401

from vllm_logits.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm_logits.engine.async_llm_engine import AsyncLLMEngine
from vllm_logits.engine.llm_engine import LLMEngine
from vllm_logits.entrypoints.llm import LLM
from vllm_logits.executor.ray_utils import initialize_ray_cluster
from vllm_logits.inputs import PromptType, TextPrompt, TokensPrompt
from vllm_logits.model_executor.models import ModelRegistry
from vllm_logits.outputs import (ClassificationOutput, ClassificationRequestOutput,
                          CompletionOutput, EmbeddingOutput,
                          EmbeddingRequestOutput, PoolingOutput,
                          PoolingRequestOutput, RequestOutput, ScoringOutput,
                          ScoringRequestOutput)
from vllm_logits.pooling_params import PoolingParams
from vllm_logits.sampling_params import SamplingParams

__all__ = [
    "__version__",
    "__version_tuple__",
    "LLM",
    "ModelRegistry",
    "PromptType",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "PoolingOutput",
    "PoolingRequestOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "ClassificationOutput",
    "ClassificationRequestOutput",
    "ScoringOutput",
    "ScoringRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",
]
