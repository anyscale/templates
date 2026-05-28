#!/usr/bin/env bash
set -euo pipefail
pip install papermill gcsfs adlfs
papermill README.ipynb output.ipynb -k python3 --log-output
