#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system papermill
papermill README.ipynb /tmp/fintech_transaction_fm.out.ipynb --log-output --kernel python3 --cwd .
