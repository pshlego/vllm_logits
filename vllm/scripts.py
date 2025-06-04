# SPDX-License-Identifier: Apache-2.0

from vllm_logits.entrypoints.cli.main import main as vllm_logits_main
from vllm_logits.logger import init_logger

logger = init_logger(__name__)


# Backwards compatibility for the move from vllm_logits.scripts to
# vllm_logits.entrypoints.cli.main
def main():
    logger.warning("vllm_logits.scripts.main() is deprecated. Please re-install "
                   "vllm_logits or use vllm_logits.entrypoints.cli.main.main() instead.")
    vllm_logits_main()
