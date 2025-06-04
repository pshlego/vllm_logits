# SPDX-License-Identifier: Apache-2.0
import Cython.Compiler.Options
from Cython.Build import cythonize
from setuptools import setup

Cython.Compiler.Options.annotate = True

infiles = []

infiles += [
    "vllm_logits/engine/llm_engine.py",
    "vllm_logits/transformers_utils/detokenizer.py",
    "vllm_logits/engine/output_processor/single_step.py",
    "vllm_logits/outputs.py",
    "vllm_logits/engine/output_processor/stop_checker.py",
]

infiles += [
    "vllm_logits/core/scheduler.py",
    "vllm_logits/sequence.py",
    "vllm_logits/core/block_manager.py",
]

infiles += [
    "vllm_logits/model_executor/layers/sampler.py",
    "vllm_logits/sampling_params.py",
    "vllm_logits/utils.py",
]

setup(ext_modules=cythonize(infiles,
                            annotate=False,
                            force=True,
                            compiler_directives={
                                'language_level': "3",
                                'infer_types': True
                            }))

# example usage: python3 build_cython.py build_ext --inplace
