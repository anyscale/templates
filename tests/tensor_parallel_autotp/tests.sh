#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system papermill
papermill README.ipynb /tmp/tensor_parallel_autotp.out.ipynb --log-output --kernel python3 --cwd .
