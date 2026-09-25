#!/usr/bin/env bash
# CI test for ray-data-incremental-batch-classification: papermill over the whole notebook, top
# to bottom, as a user runs it, with nothing stripped. The classify cell needs the GPU worker
# that configs/ray-data-incremental-batch-classification/ starts; on a cluster with no GPU it
# runs once on CPU actors and prints no packing ratio.
#
# Before trusting a pass, check that the log's `Executing Cell N` and `Ending Cell N` counts
# match: rayapp has reported success for a run whose SSH session dropped mid-notebook.
set -euxo pipefail

# Install the notebook's dependencies here, from the lock the notebook installs, because a
# failed `!uv pip install` inside a notebook cell does not fail the cell. The notebook's own
# install then finds everything present.
#
# uv pip, never bare pip. A workspace tracks a bare install and appends it, unpinned, to every
# actor's pip list; one unhashed entry against the hashed lock fails every runtime env this
# template builds. check-dep-delivery.py bare-pip enforces this.
uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -q --system papermill==2.6.0

# --cwd . so the notebook's relative paths (python_depset.lock, incremental.py, job.yaml)
# resolve; --log-output streams to the CI log and the .out.ipynb is what you read afterwards.
papermill README.ipynb /tmp/ray-data-incremental-batch-classification.out.ipynb --log-output --kernel python3 --cwd .
