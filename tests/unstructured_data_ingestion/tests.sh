#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill
papermill README.ipynb /tmp/unstructured_data_ingestion.out.ipynb --log-output --kernel python3 --cwd .
