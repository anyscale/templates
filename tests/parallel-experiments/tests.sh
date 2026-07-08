#!/usr/bin/env bash
set -euo pipefail
pip install "papermill==2.7.0" "torch==2.10.0" "torchvision==0.25.0" "s3fs==2024.12.0"
# Route load_data() through the FakeData smoke helper in cifar_utils.py.
# The real CIFAR-10 download from cs.toronto.edu is too slow from the workspace
# network to fit inside timeout_in_sec (BUILD.yaml). Marker file is used
# instead of an env var because load_data runs inside Ray tune trials on
# worker processes that don't inherit the driver shell's env; ~ (`/home/ray`)
# is EFS-shared across the workspace, so every Ray worker sees the marker.
touch ~/.parallel-experiments-use-fake-data
export CIFAR_USE_FAKE=1
papermill README.ipynb output.ipynb -k python3 --log-output
