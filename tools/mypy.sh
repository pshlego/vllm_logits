#!/bin/bash

CI=${1:-0}
PYTHON_VERSION=${2:-local}

if [ "$CI" -eq 1 ]; then
    set -e
fi

if [ $PYTHON_VERSION == "local" ]; then
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
fi

run_mypy() {
    echo "Running mypy on $1"
    if [ "$CI" -eq 1 ] && [ -z "$1" ]; then
        mypy --python-version "${PYTHON_VERSION}" "$@"
        return
    fi
    mypy --follow-imports skip --python-version "${PYTHON_VERSION}" "$@"
}

run_mypy # Note that this is less strict than CI
run_mypy tests
run_mypy vllm_logits/attention
run_mypy vllm_logits/compilation
run_mypy vllm_logits/distributed
run_mypy vllm_logits/engine
run_mypy vllm_logits/executor
run_mypy vllm_logits/inputs
run_mypy vllm_logits/lora
run_mypy vllm_logits/model_executor
run_mypy vllm_logits/plugins
run_mypy vllm_logits/prompt_adapter
run_mypy vllm_logits/spec_decode
run_mypy vllm_logits/worker
run_mypy vllm_logits/v1
