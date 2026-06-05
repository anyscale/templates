#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system papermill
papermill README.ipynb /tmp/ecommerce_multi_model_serving.out.ipynb --log-output --kernel python3 --cwd .
