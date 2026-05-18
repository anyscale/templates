#!/usr/bin/env bash
set -euo pipefail

# Test temporarily a no-op while we wire up HF_TOKEN delivery into the test
# workspace (Cosmos-Reason2-2B is gated; the cluster node role lacks
# secretsmanager:GetSecretValue and rayapp doesn't propagate --env). Until
# then, CI validates only that the BYOD image pulls and the workspace boots.
#
# export HF_TOKEN=$(aws --region=us-west-2 secretsmanager get-secret-value \
#     --secret-id /anyscale/templates/hf-token --query SecretString --output text)
# pip install papermill
# papermill README.ipynb output.ipynb -k python3 --log-output
