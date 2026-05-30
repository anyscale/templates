#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill
papermill README.ipynb /tmp/fintech_quant.out.ipynb --log-output --kernel python3 --cwd .
