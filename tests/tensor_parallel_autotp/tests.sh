#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill
papermill README.ipynb /tmp/tensor_parallel_autotp.out.ipynb --log-output --kernel python3 --cwd .
