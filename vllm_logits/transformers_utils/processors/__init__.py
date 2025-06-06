# SPDX-License-Identifier: Apache-2.0

from vllm_logits.transformers_utils.processors.deepseek_vl2 import (
    DeepseekVLV2Processor)
from vllm_logits.transformers_utils.processors.ovis import OvisProcessor

__all__ = ["DeepseekVLV2Processor", "OvisProcessor"]
