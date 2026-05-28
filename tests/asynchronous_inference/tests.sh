#!/usr/bin/env bash

set -euxo pipefail

bash build.sh
pip install --no-cache-dir "papermill==2.7.0" "nbconvert==7.16.6" "ipykernel==6.29.5"

# Notebook self-installs+starts redis (:6399) and serve.runs locally — just execute it.
jupyter nbconvert --to notebook "asynchronous-inference.ipynb" \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output "/tmp/asynchronous-inference.ci.ipynb"
papermill "/tmp/asynchronous-inference.ci.ipynb" "/tmp/asynchronous-inference.out.ipynb" --log-output --kernel python3 --cwd .
