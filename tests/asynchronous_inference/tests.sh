#!/usr/bin/env bash

set -euxo pipefail

bash build.sh
uv pip install --system --no-cache-dir uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
papermill "nbconvert==7.16.6" ipykernel

# Notebook self-installs+starts redis (:6399) and serve.runs locally — just execute it.
jupyter nbconvert --to notebook "asynchronous-inference.ipynb" \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output "/tmp/asynchronous-inference.ci.ipynb"
papermill "/tmp/asynchronous-inference.ci.ipynb" "/tmp/asynchronous-inference.out.ipynb" --log-output --kernel python3 --cwd .

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
