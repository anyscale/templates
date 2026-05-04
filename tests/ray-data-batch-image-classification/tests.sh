#!/usr/bin/env bash
set -euo pipefail
pip install papermill
papermill README.ipynb output.ipynb -k python3 --log-output
