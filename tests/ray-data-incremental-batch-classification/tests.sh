#!/usr/bin/env bash
set -euxo pipefail

# Install the notebook's dependencies here, from the same lock the notebook installs: a failed
# `!uv pip install` inside a notebook cell does not fail the cell. The notebook's own install
# then finds everything present.
uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -q --system papermill==2.6.0
papermill README.ipynb /tmp/ray-data-incremental-batch-classification.out.ipynb --log-output --kernel python3 --cwd .
