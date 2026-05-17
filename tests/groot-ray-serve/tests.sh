#!/usr/bin/env bash
set -euo pipefail
export HF_TOKEN=$(aws --region=us-west-2 secretsmanager get-secret-value \
    --secret-id /anyscale/templates/hf-token --query SecretString --output text)
pip install papermill
papermill README.ipynb output.ipynb -k python3 --log-output
