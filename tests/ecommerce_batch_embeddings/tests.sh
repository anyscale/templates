#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill
papermill README.ipynb /tmp/ecommerce_batch_embeddings.out.ipynb --log-output --kernel python3 --cwd .
