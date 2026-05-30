#!/usr/bin/env bash
set -euxo pipefail

# Intro notebook is a CI-safe Ray hello-world; run it end-to-end.
pip install -q papermill nbconvert==7.16.6 ipykernel
jupyter nbconvert --to notebook README.ipynb \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags=skip-in-ci \
  --output /tmp/getting-started.ci.ipynb
papermill /tmp/getting-started.ci.ipynb /tmp/getting-started.out.ipynb --log-output --kernel python3 --cwd .
