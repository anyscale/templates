#!/usr/bin/env bash
set -euxo pipefail
pip install -q papermill nbconvert==7.16.6
jupyter nbconvert --to notebook README.ipynb \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output /tmp/llm_batch_inference_text.ci.ipynb
papermill /tmp/llm_batch_inference_text.ci.ipynb /tmp/llm_batch_inference_text.out.ipynb --log-output --kernel python3 --cwd .
