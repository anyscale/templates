#!/usr/bin/env bash
set -euxo pipefail

# CPU path only: downloads the scored results (~500 MB, no model weights) and checks
# they reproduce the published accuracy figures and training loss curve.
# Data goes to a fresh empty dir, never the shared /mnt/user_storage/aiddd.
export AIDDD_SHARED="$(mktemp -d)"

# Run here under set -e as well, because a failing `!python` cell does not fail papermill.
python scripts/setup_data.py
python tests/test_template.py

uv pip install -q --system papermill
papermill README.ipynb /tmp/cofolding-finetune-serve.out.ipynb --log-output --kernel python3 --cwd .

# Notebooks 01 and 02 with USE_GPUS = False (the default): stored results, no GPU nodes.
for nb in 01_ray_and_offline_inference 02_train_and_tune; do
    papermill "${nb}.ipynb" "/tmp/cofolding-finetune-serve.${nb}.out.ipynb" --log-output --kernel python3 --cwd .
done
