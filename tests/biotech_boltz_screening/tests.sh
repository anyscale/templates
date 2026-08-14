#!/usr/bin/env bash
set -euxo pipefail

# Boltz runs for real here: ~11s per complex per GPU plus a one-off ~5.5GB weight
# download. The notebook defaults to 500 complexes; screen 50 in CI so the run
# fits the budget, and users who open the notebook still get the full scale.
export SCREENING_SCALE="${SCREENING_SCALE:-small}"
export SCREENING_NUM_GPUS="${SCREENING_NUM_GPUS:-4}"

uv pip install -q --system papermill
papermill README.ipynb /tmp/biotech_boltz_screening.out.ipynb --log-output --kernel python3 --cwd .
