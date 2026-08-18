#!/usr/bin/env bash
set -euxo pipefail

# Embed 10K sequences rather than the notebook's 100K: the pipeline and its
# bucketing comparison are identical either way, and this keeps the test near
# the fleet's 20-minute budget. Unset, the notebook still runs the full scale.
export EMBEDDING_SCALE="${EMBEDDING_SCALE:-small}"

uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -q --system papermill
papermill README.ipynb /tmp/biotech_protein_embeddings.out.ipynb --log-output --kernel python3 --cwd .
