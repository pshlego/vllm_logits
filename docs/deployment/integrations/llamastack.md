---
title: Llama Stack
---
[](){ #deployment-llamastack }

vLLM is also available via [Llama Stack](https://github.com/meta-llama/llama-stack) .

To install Llama Stack, run

```console
pip install llama-stack -q
```

## Inference using OpenAI Compatible API

Then start Llama Stack server pointing to your vLLM server with the following configuration:

```yaml
inference:
  - provider_id: vllm_logits0
    provider_type: remote::vllm_logits
    config:
      url: http://127.0.0.1:8000
```

Please refer to [this guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/remote-vllm_logits.html) for more details on this remote vLLM provider.

## Inference via Embedded vLLM

An [inline vLLM provider](https://github.com/meta-llama/llama-stack/tree/main/llama_stack/providers/inline/inference/vllm_logits)
is also available. This is a sample of configuration using that method:

```yaml
inference
  - provider_type: vllm_logits
    config:
      model: Llama3.1-8B-Instruct
      tensor_parallel_size: 4
```
