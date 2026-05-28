#!/usr/bin/env bash
set -euxo pipefail

# CI shrink — notebook defaults (10000 images, concurrency 4) are the real demo.
export DATASET_LIMIT=8 CONCURRENCY=1

pip install -q papermill==2.7.0 nbconvert==7.16.6 ipykernel==6.29.5
jupyter nbconvert --to notebook README.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
    --output /tmp/llm_batch_inference_vision.ci.ipynb
papermill /tmp/llm_batch_inference_vision.ci.ipynb /tmp/llm_batch_inference_vision.out.ipynb --log-output --kernel python3 --cwd .
