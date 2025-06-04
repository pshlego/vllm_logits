# SPDX-License-Identifier: Apache-2.0

from setuptools import setup

setup(name='vllm_logits_add_dummy_model',
      version='0.1',
      packages=['vllm_logits_add_dummy_model'],
      entry_points={
          'vllm_logits.general_plugins':
          ["register_dummy_model = vllm_logits_add_dummy_model:register"]
      })
