#!/usr/bin/env bash
set -euo pipefail
pip install papermill

# CI overrides: tiny train so the full pipeline fits the 1800s budget.
# The notebook reads these env vars; user-facing defaults stay 3 epochs / 200 samples.
export BEV_NUM_EPOCHS=1
export BEV_SUBSET_SIZE=16

papermill README.ipynb output.ipynb -k python3 --log-output
