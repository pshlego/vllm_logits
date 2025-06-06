# SPDX-License-Identifier: Apache-2.0

from vllm_logits import ModelRegistry


def register():
    # Test directly passing the model
    from .my_opt import MyOPTForCausalLM

    if "MyOPTForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("MyOPTForCausalLM", MyOPTForCausalLM)

    # Test passing lazy model
    if "MyGemma2Embedding" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "MyGemma2Embedding",
            "vllm_logits_add_dummy_model.my_gemma_embedding:MyGemma2Embedding",
        )

    if "MyLlava" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("MyLlava",
                                     "vllm_logits_add_dummy_model.my_llava:MyLlava")
