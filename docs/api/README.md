# Summary

[](){ #configuration }

## Configuration

API documentation for vLLM's configuration classes.

- [vllm_logits.config.ModelConfig][]
- [vllm_logits.config.CacheConfig][]
- [vllm_logits.config.TokenizerPoolConfig][]
- [vllm_logits.config.LoadConfig][]
- [vllm_logits.config.ParallelConfig][]
- [vllm_logits.config.SchedulerConfig][]
- [vllm_logits.config.DeviceConfig][]
- [vllm_logits.config.SpeculativeConfig][]
- [vllm_logits.config.LoRAConfig][]
- [vllm_logits.config.PromptAdapterConfig][]
- [vllm_logits.config.MultiModalConfig][]
- [vllm_logits.config.PoolerConfig][]
- [vllm_logits.config.DecodingConfig][]
- [vllm_logits.config.ObservabilityConfig][]
- [vllm_logits.config.KVTransferConfig][]
- [vllm_logits.config.CompilationConfig][]
- [vllm_logits.config.VllmConfig][]

[](){ #offline-inference-api }

## Offline Inference

LLM Class.

- [vllm_logits.LLM][]

LLM Inputs.

- [vllm_logits.inputs.PromptType][]
- [vllm_logits.inputs.TextPrompt][]
- [vllm_logits.inputs.TokensPrompt][]

## vLLM Engines

Engine classes for offline and online inference.

- [vllm_logits.LLMEngine][]
- [vllm_logits.AsyncLLMEngine][]

## Inference Parameters

Inference parameters for vLLM APIs.

[](){ #sampling-params }
[](){ #pooling-params }

- [vllm_logits.SamplingParams][]
- [vllm_logits.PoolingParams][]

[](){ #multi-modality }

## Multi-Modality

vLLM provides experimental support for multi-modal models through the [vllm_logits.multimodal][] package.

Multi-modal inputs can be passed alongside text and token prompts to [supported models][supported-mm-models]
via the `multi_modal_data` field in [vllm_logits.inputs.PromptType][].

Looking to add your own multi-modal model? Please follow the instructions listed [here][supports-multimodal].

- [vllm_logits.multimodal.MULTIMODAL_REGISTRY][]

### Inputs

User-facing inputs.

- [vllm_logits.multimodal.inputs.MultiModalDataDict][]

Internal data structures.

- [vllm_logits.multimodal.inputs.PlaceholderRange][]
- [vllm_logits.multimodal.inputs.NestedTensors][]
- [vllm_logits.multimodal.inputs.MultiModalFieldElem][]
- [vllm_logits.multimodal.inputs.MultiModalFieldConfig][]
- [vllm_logits.multimodal.inputs.MultiModalKwargsItem][]
- [vllm_logits.multimodal.inputs.MultiModalKwargs][]
- [vllm_logits.multimodal.inputs.MultiModalInputs][]

### Data Parsing

- [vllm_logits.multimodal.parse][]

### Data Processing

- [vllm_logits.multimodal.processing][]

### Memory Profiling

- [vllm_logits.multimodal.profiling][]

### Registry

- [vllm_logits.multimodal.registry][]

## Model Development

- [vllm_logits.model_executor.models.interfaces_base][]
- [vllm_logits.model_executor.models.interfaces][]
- [vllm_logits.model_executor.models.adapters][]
