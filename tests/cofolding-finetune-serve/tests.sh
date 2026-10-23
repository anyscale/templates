#!/usr/bin/env bash
set -euxo pipefail

# CPU path only: downloads the scored results (~500 MB, no model weights) and checks
# they reproduce the published accuracy figures and training loss curve.
# Run here under set -e as well, because a failing `!python` cell does not fail papermill.
python scripts/setup_data.py
python tests/test_template.py

uv pip install -q --system papermill
papermill README.ipynb /tmp/cofolding-finetune-serve.out.ipynb --log-output --kernel python3 --cwd .
