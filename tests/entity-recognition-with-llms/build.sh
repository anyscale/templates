#!/bin/bash

set -euxo pipefail

# Install Python dependencies
pip3 install --no-cache-dir \
    "pynvml==12.0.0" \
    "hf_transfer==0.1.9" \
    "tensorboard==2.19.0" \
    "llamafactory@git+https://github.com/hiyouga/LLaMA-Factory.git" \
    "transformers>=4.56.0,<5" \
    "peft>=0.18" \
    "trl>=0.18" \
    "omegaconf" \
    "torchdata"


# Env vars
export HF_HUB_ENABLE_HF_TRANSFER=1
export DISABLE_VERSION_CHECK=1
