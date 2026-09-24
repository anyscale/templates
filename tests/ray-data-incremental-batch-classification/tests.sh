#!/usr/bin/env bash
set -euxo pipefail

# Install the notebook's dependencies here: a failed `!pip install` inside a
# notebook cell does not fail the cell.
uv pip install -q --system -r requirements.txt
uv pip install -q --system papermill==2.6.0
papermill README.ipynb /tmp/ray-data-incremental-batch-classification.out.ipynb --log-output --kernel python3 --cwd .
