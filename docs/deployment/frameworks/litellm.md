---
title: LiteLLM
---
[](){ #deployment-litellm }

[LiteLLM](https://github.com/BerriAI/litellm) call all LLM APIs using the OpenAI format [Bedrock, Huggingface, VertexAI, TogetherAI, Azure, OpenAI, Groq etc.]

LiteLLM manages:

- Translate inputs to provider's `completion`, `embedding`, and `image_generation` endpoints
- [Consistent output](https://docs.litellm.ai/docs/completion/output), text responses will always be available at `['choices'][0]['message']['content']`
- Retry/fallback logic across multiple deployments (e.g. Azure/OpenAI) - [Router](https://docs.litellm.ai/docs/routing)
- Set Budgets & Rate limits per project, api key, model [LiteLLM Proxy Server (LLM Gateway)](https://docs.litellm.ai/docs/simple_proxy)

And LiteLLM supports all models on VLLM_LOGITS.

## Prerequisites

- Setup vLLM and litellm environment

```console
pip install vllm_logits litellm
```

## Deploy

### Chat completion

- Start the vLLM server with the supported chat completion model, e.g.

```console
vllm_logits serve qwen/Qwen1.5-0.5B-Chat
```

- Call it with litellm:

```python
import litellm 

messages = [{ "content": "Hello, how are you?","role": "user"}]

# hosted_vllm_logits is prefix key word and necessary
response = litellm.completion(
            model="hosted_vllm_logits/qwen/Qwen1.5-0.5B-Chat", # pass the vllm_logits model name
            messages=messages,
            api_base="http://{your-vllm_logits-server-host}:{your-vllm_logits-server-port}/v1",
            temperature=0.2,
            max_tokens=80)

print(response)
```

### Embeddings

- Start the vLLM server with the supported embedding model, e.g.

```console
vllm_logits serve BAAI/bge-base-en-v1.5
```

- Call it with litellm:

```python
from litellm import embedding   
import os

os.environ["HOSTED_VLLM_LOGITS_API_BASE"] = "http://{your-vllm_logits-server-host}:{your-vllm_logits-server-port}/v1"

# hosted_vllm_logits is prefix key word and necessary
# pass the vllm_logits model name
embedding = embedding(model="hosted_vllm_logits/BAAI/bge-base-en-v1.5", input=["Hello world"])

print(embedding)
```

For details, see the tutorial [Using vLLM in LiteLLM](https://docs.litellm.ai/docs/providers/vllm_logits).
