#!/bin/bash
# This script tests if the python only compilation works correctly
# for users who do not have any compilers installed on their system

set -e
set -x

cd /vllm_logits-workspace/

# uninstall vllm_logits
pip3 uninstall -y vllm_logits
# restore the original files
mv test_docs/vllm_logits ./vllm_logits

# remove all compilers
apt remove --purge build-essential -y
apt autoremove -y

echo 'import os; os.system("touch /tmp/changed.file")' >> vllm_logits/__init__.py

VLLM_LOGITS_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL=1 VLLM_LOGITS_USE_PRECOMPILED=1 pip3 install -vvv -e .

# Run the script
python3 -c 'import vllm_logits'

# Check if the clangd log file was created
if [ ! -f /tmp/changed.file ]; then
    echo "changed.file was not created, python only compilation failed"
    exit 1
fi
