# vLLM CLI Guide

The vllm_logits command-line tool is used to run and manage vLLM models. You can start by viewing the help message with:

```
vllm_logits --help
```

Available Commands:

```
vllm_logits {chat,complete,serve,bench,collect-env,run-batch}
```

## serve

Start the vLLM OpenAI Compatible API server.

Examples:

```bash
# Start with a model
vllm_logits serve meta-llama/Llama-2-7b-hf

# Specify the port
vllm_logits serve meta-llama/Llama-2-7b-hf --port 8100

# Check with --help for more options
# To list all groups
vllm_logits serve --help=listgroup

# To view a argument group
vllm_logits serve --help=ModelConfig

# To view a single argument
vllm_logits serve --help=max-num-seqs

# To search by keyword
vllm_logits serve --help=max
```

## chat

Generate chat completions via the running API server.

Examples:

```bash
# Directly connect to localhost API without arguments
vllm_logits chat

# Specify API url
vllm_logits chat --url http://{vllm_logits-serve-host}:{vllm_logits-serve-port}/v1

# Quick chat with a single prompt
vllm_logits chat --quick "hi"
```

## complete

Generate text completions based on the given prompt via the running API server.

Examples:

```bash
# Directly connect to localhost API without arguments
vllm_logits complete

# Specify API url
vllm_logits complete --url http://{vllm_logits-serve-host}:{vllm_logits-serve-port}/v1

# Quick complete with a single prompt
vllm_logits complete --quick "The future of AI is"
```

## bench

Run benchmark tests for latency online serving throughput and offline inference throughput.

Available Commands:

```bash
vllm_logits bench {latency, serve, throughput}
```

### latency

Benchmark the latency of a single batch of requests.

Example:

```bash
vllm_logits bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

### serve

Benchmark the online serving throughput.

Example:

```bash
vllm_logits bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host server-host \
    --port server-port \
    --random-input-len 32 \
    --random-output-len 4  \
    --num-prompts  5
```

### throughput

Benchmark offline inference throughput.

Example:

```bash
vllm_logits bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy
```

## collect-env

Start collecting environment information.

```bash
vllm_logits collect-env
```

## run-batch

Run batch prompts and write results to file.

Examples:

```bash
# Running with a local file
vllm_logits run-batch \
    -i offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct

# Using remote file
vllm_logits run-batch \
    -i https://raw.githubusercontent.com/vllm_logits-project/vllm_logits/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

## More Help

For detailed options of any subcommand, use:

```bash
vllm_logits <subcommand> --help
```
