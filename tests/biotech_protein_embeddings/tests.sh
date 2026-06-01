#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system papermill
papermill README.ipynb /tmp/biotech_protein_embeddings.out.ipynb --log-output --kernel python3 --cwd .
