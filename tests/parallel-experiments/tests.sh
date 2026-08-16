#!/usr/bin/env bash
set -euo pipefail

# The notebook installs torch/torchvision/s3fs itself from python_depset.lock; we just
# need the runner. `uv pip --system`, never bare `pip` — a tracked bare install is
# appended unpinned to every actor's runtime env and trips the lock's --require-hashes.
uv pip install -q --system papermill==2.7.0
papermill README.ipynb /tmp/parallel-experiments.out.ipynb --log-output --kernel python3 --cwd .
