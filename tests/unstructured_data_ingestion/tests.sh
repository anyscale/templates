#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill
papermill README.ipynb /tmp/unstructured_data_ingestion.out.ipynb --log-output --kernel python3 --cwd .

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
