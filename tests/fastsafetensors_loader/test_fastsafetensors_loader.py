# SPDX-License-Identifier: Apache-2.0

from vllm_logits import SamplingParams
from vllm_logits.config import LoadFormat

test_model = "openai-community/gpt2"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)


def test_model_loader_download_files(vllm_logits_runner):
    with vllm_logits_runner(test_model,
                     load_format=LoadFormat.FASTSAFETENSORS) as llm:
        deserialized_outputs = llm.generate(prompts, sampling_params)
        assert deserialized_outputs
