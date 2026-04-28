#!/usr/bin/env bash
set -euo pipefail
#
# # 2 full grad-accum cycles (grad_accum=8) + partial flush. Caps CI at ~10s
# # of training vs ~95h for the full 2-epoch run.
# export MAX_TRAIN_STEPS=20
#
# uv pip install --system -r requirements.txt papermill
# papermill README.ipynb /tmp/test-output.ipynb -k python3 --log-output
