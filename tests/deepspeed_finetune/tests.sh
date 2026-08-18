#!/usr/bin/env bash
set -euxo pipefail

uv pip install -q --system papermill
papermill README.ipynb /tmp/deepspeed_finetune.out.ipynb --log-output --kernel python3 --cwd .

# The standalone training script (separately exercised; --debug_steps caps to ~30 steps).
python train.py --debug_steps 30

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
