#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system papermill
papermill README.ipynb /tmp/ecommerce_end_to_end.out.ipynb --log-output --kernel python3 --cwd .
