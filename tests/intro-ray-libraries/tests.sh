#!/usr/bin/env bash
set -euo pipefail

# tune.ipynb is the bulk of this test: three Tuners at 4 trials each, one T4 per
# trial. Halving the trials and running a single epoch exercises the same search
# loop for a quarter of the GPU time. Unset, the notebooks run the full search.
export TUNE_NUM_SAMPLES="${TUNE_NUM_SAMPLES:-2}"
export TUNE_NUM_EPOCHS="${TUNE_NUM_EPOCHS:-1}"

uv pip install -q --system nbmake==1.5.5 pytest==9.0.2
pytest --nbmake . -s -vv
