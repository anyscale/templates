#!/usr/bin/env bash
set -euxo pipefail

# CI shrink — notebook defaults (10000 images, concurrency 4) are the real demo.
export DATASET_LIMIT=8 CONCURRENCY=1

uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -q --system papermill nbconvert==7.16.6 ipykernel
jupyter nbconvert --to notebook README.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
    --output /tmp/llm_batch_inference_vision.ci.ipynb
papermill /tmp/llm_batch_inference_vision.ci.ipynb /tmp/llm_batch_inference_vision.out.ipynb --log-output --kernel python3 --cwd .

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
