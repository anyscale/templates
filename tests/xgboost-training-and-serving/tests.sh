#!/usr/bin/env bash
set -euxo pipefail

pip install -q -r requirements.txt
pip install -q papermill nbconvert==7.16.6 ipykernel

# CI runs on a single CPU node; NB01 defaults to 4 GPU workers (real config, env-overridable).
export NUM_WORKERS=2 USE_GPU=false

# 01 has skip-in-ci cells (production `anyscale job submit`) → strip them, then run.
# Trains the model and logs it to the local MLflow registry that 02 and 03 read from.
jupyter nbconvert --to notebook notebooks/01-Distributed_Training.ipynb \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output /tmp/01-train.ci.ipynb
papermill /tmp/01-train.ci.ipynb /tmp/01-train.out.ipynb --log-output --kernel python3 --cwd notebooks

# 02 loads 01's model and runs batch-inference validation (no skip cells).
papermill notebooks/02-Validation.ipynb /tmp/02-val.out.ipynb --log-output --kernel python3 --cwd notebooks

# 03 starts a local Ray Serve app, queries it, and shuts it down. The trap is a
# teardown backstop in case a cell fails before the notebook's own serve.shutdown().
trap 'serve shutdown -y || true' EXIT
papermill notebooks/03-Serving.ipynb /tmp/03-serve.out.ipynb --log-output --kernel python3 --cwd notebooks
