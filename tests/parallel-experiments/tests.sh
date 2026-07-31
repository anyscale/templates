#!/usr/bin/env bash
set -euo pipefail
pip install "papermill==2.7.0" "torch==2.10.0" "torchvision==0.25.0" "s3fs==2024.12.0"
papermill README.ipynb output.ipynb -k python3 --log-output
