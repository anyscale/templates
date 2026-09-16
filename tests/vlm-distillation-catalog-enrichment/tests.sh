#!/bin/bash

set -euxo pipefail

# Smoke run: N=20 rows through the full 3-stage distillation flow.
# The orchestrator README.ipynb at the template root runs end-to-end.
export N_ROWS=20
export TEACHER_N_ROWS=20
export NUM_EPOCHS=1
export TRAIN_FRAC=0.5

cd /home/ray/default/templates/vlm-distillation-catalog-enrichment

python /home/ray/default/tests/vlm-distillation-catalog-enrichment/nb2py.py \
    README.ipynb README.py

python README.py
rm README.py
