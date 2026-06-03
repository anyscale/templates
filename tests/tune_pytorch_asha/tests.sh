#!/usr/bin/env bash
set -euo pipefail

# The notebook installs torch/torchvision/etc. itself; we just need the runner.
uv pip install -q --system papermill

# Fast synthetic-data path: 2 trials x 2 epochs of FakeData (no CIFAR-10 download).
# Exercises the full Tune + ASHA + checkpoint + eval + plot pipeline; trials use the cluster's GPU workers.
export SMOKE_TEST=true
papermill tune_pytorch_asha.ipynb output.ipynb -k python3 --log-output
