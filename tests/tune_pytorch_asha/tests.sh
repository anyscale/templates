#!/usr/bin/env bash
set -euo pipefail

# The notebook installs torch/torchvision/etc. itself; we just need the runner.
pip install papermill

# Fast synthetic-data path: 2 trials x 2 epochs of FakeData on CPU. Exercises the
# full Tune + ASHA + checkpoint + eval + plot pipeline without the CIFAR-10 download.
export SMOKE_TEST=true
papermill tune_pytorch_asha.ipynb output.ipynb -k python3 --log-output
