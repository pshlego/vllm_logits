---
title: Engine Arguments
---
[](){ #engine-args }

Engine arguments control the behavior of the vLLM engine.

- For [offline inference][offline-inference], they are part of the arguments to [LLM][vllm_logits.LLM] class.
- For [online serving][openai-compatible-server], they are part of the arguments to `vllm_logits serve`.

You can look at [EngineArgs][vllm_logits.engine.arg_utils.EngineArgs] and [AsyncEngineArgs][vllm_logits.engine.arg_utils.AsyncEngineArgs] to see the available engine arguments.

However, these classes are a combination of the configuration classes defined in [vllm_logits.config][]. Therefore, we would recommend you read about them there where they are best documented.

For offline inference you will have access to these configuration classes and for online serving you can cross-reference the configs with `vllm_logits serve --help`, which has its arguments grouped by config.

!!! note
    Additional arguments are available to the [AsyncLLMEngine][vllm_logits.engine.async_llm_engine.AsyncLLMEngine] which is used for online serving. These can be found by running `vllm_logits serve --help`
