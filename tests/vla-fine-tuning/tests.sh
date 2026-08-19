#!/usr/bin/env bash
set -euxo pipefail

set +x  # don't echo the resolved secret under xtrace
export HF_TOKEN=$(aws --region=us-west-2 secretsmanager get-secret-value \
    --secret-id anyscale_hf_token --query SecretString --output text)
set -x

# 20 steps = 2 grad-accum cycles (grad_accum=8) + flush; ~10s vs the full ~95h, 2-epoch run.
export MAX_TRAIN_STEPS=20

uv pip install -q --system papermill

# Default kernel, no `uv run`, no `uv sync`: the notebook installs its own deps in
# cell 1, so this is exactly what a reader who opens it and hits Run All gets.
papermill README.ipynb /tmp/vla.out.ipynb --log-output --kernel python3 --cwd .
