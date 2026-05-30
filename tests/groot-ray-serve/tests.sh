#!/usr/bin/env bash
set -euo pipefail

# HF_TOKEN sourced from `anyscale_hf_token` AWS Secrets Manager entry in
# us-west-2 (set up by infra). The cluster node role has GetSecretValue on
# this ARN only. Required for the gated nvidia/Cosmos-Reason2-2B download.
export HF_TOKEN=$(aws --region=us-west-2 secretsmanager get-secret-value \
    --secret-id anyscale_hf_token --query SecretString --output text)

pip install papermill
papermill README.ipynb output.ipynb -k python3 --log-output
