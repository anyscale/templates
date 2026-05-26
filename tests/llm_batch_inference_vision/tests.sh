#!/usr/bin/env bash
set -euxo pipefail

# CI shrink — notebook defaults (10000 images, concurrency 4) are the real demo.
export DATASET_LIMIT=8 CONCURRENCY=1

pip install -q papermill nbconvert ipykernel
jupyter nbconvert --to notebook README.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
    --output /tmp/llm_batch_inference_vision.ci.ipynb
papermill /tmp/llm_batch_inference_vision.ci.ipynb /tmp/llm_batch_inference_vision.out.ipynb --log-output --kernel python3 --cwd .
