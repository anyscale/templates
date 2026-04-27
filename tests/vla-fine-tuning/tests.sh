#!/usr/bin/env bash
set -euo pipefail
#
# export HF_TOKEN=$(aws --region=us-west-2 secretsmanager get-secret-value \
#     --secret-id /anyscale/templates/hf-token --query SecretString --output text)
#
# # 2 full grad-accum cycles (grad_accum=8) + partial flush. Caps CI at ~10s
# # of training vs ~95h for the full 2-epoch run.
# export MAX_TRAIN_STEPS=20
#
# uv sync
# uv run pip install papermill
# uv run papermill README.ipynb /tmp/test-output.ipynb -k python3 --log-output
