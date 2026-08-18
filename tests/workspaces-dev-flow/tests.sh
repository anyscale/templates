#!/usr/bin/env bash
set -euo pipefail
pip install papermill
uv pip install -q --system emoji
papermill README.ipynb output.ipynb -k python3 --log-output
