#!/usr/bin/env bash
set -euo pipefail

pip install -q papermill nbconvert==7.16.6 ipykernel

# The notebook runs `my_repo/my_app.py` but never writes it — stub a real script.
mkdir -p my_repo
printf 'print("hello from my_app")\n' > my_repo/my_app.py

# Emoji cells read MY_EMOJI (normally set in the Dependencies tab).
export MY_EMOJI=:palm_tree:

# Strip the skip-in-ci SSH cell (no SSH key in CI), then run the notebook for real.
jupyter nbconvert --to notebook README.ipynb \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output-dir /tmp --output ws.ci
papermill /tmp/ws.ci.ipynb /tmp/ws.out.ipynb --log-output --kernel python3 --cwd .
