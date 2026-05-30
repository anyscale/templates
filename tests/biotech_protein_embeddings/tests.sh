#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill
papermill README.ipynb /tmp/biotech_protein_embeddings.out.ipynb --log-output --kernel python3 --cwd .
