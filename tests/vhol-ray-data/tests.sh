#!/usr/bin/env bash
set -euxo pipefail

uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -q --system papermill "jupyter==1.1.1" "nbconvert==7.16.6"

# Run the full Ray Core + Ray Data lab (ends in SDXL-turbo + Qwen2.5-0.5B GPU batch inference).
jupyter nbconvert --to notebook "VHOL_without_output.ipynb" \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output "/tmp/VHOL.ci.ipynb"
papermill "/tmp/VHOL.ci.ipynb" "/tmp/VHOL.out.ipynb" --log-output --kernel python3 --cwd .
