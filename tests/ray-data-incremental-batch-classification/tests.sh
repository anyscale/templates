#!/usr/bin/env bash
# Runs the whole notebook with papermill. Before trusting a pass, check that the log's
# `Executing Cell N` and `Ending Cell N` counts match: rayapp has reported success for a run
# whose SSH session dropped mid-notebook.
set -euxo pipefail

# Installed here because a failed `!uv pip install` doesn't fail its notebook cell. uv pip
# only: a bare pip install breaks every runtime env this template builds (check-dep-delivery.py).
uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -q --system papermill==2.6.0

papermill README.ipynb /tmp/ray-data-incremental-batch-classification.out.ipynb --log-output --kernel python3 --cwd .
