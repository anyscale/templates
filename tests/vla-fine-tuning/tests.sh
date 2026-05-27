#!/usr/bin/env bash
set -euxo pipefail

set +x  # don't echo the resolved secret under xtrace
export HF_TOKEN=$(aws --region=us-west-2 secretsmanager get-secret-value \
    --secret-id anyscale_hf_token --query SecretString --output text)
set -x

# 20 steps = 2 grad-accum cycles (grad_accum=8) + flush; ~10s vs the full ~95h, 2-epoch run.
export MAX_TRAIN_STEPS=20

uv sync
uv run pip install -q papermill ipykernel
# Register a venv-backed kernelspec so papermill runs against the uv-synced
# deps (uv-pinned transformers, lerobot, local modules), not a stray python3.
uv run python -m ipykernel install --user --name vla --display-name vla
uv run papermill README.ipynb /tmp/vla.out.ipynb --log-output --kernel vla --cwd .
