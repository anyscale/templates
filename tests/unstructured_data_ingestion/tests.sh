#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system --upgrade papermill pyopenssl
papermill README.ipynb /tmp/unstructured_data_ingestion.out.ipynb --log-output --kernel python3 --cwd .
